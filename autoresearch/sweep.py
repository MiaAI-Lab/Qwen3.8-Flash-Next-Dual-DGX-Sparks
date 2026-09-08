#!/usr/bin/env python3
"""Autoresearch sweep driver for the dual-Spark Qwen3.8-Flash-Next deployment.

Runs a list of trials. For each: apply .env overrides -> relaunch -> wait for
health (with a HARD timeout) -> benchmark -> record -> restore.

Safety, learned the hard way from a torch.compile mode-3 hang that took both
nodes down for 100 minutes:
  * every launch has a hard deadline; a trial that misses it is killed and
    recorded as FAILED rather than left to hang;
  * the baseline .env is restored after every trial, pass or fail, so the
    server never stays down in a broken config;
  * a trial whose benchmark is polluted by other clients is recorded DIRTY
    and not counted (see --expect-tokens).

Usage:
  ./sweep.py --list
  ./sweep.py --only moe_flashinfer_b12x,linear_b12x
  ./sweep.py --all
  ./sweep.py --restore          # just put the baseline back and start it
"""
import argparse, atexit, json, os, re, shutil, signal, subprocess, sys, time, urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "autoresearch")
ENV = os.path.join(ROOT, ".env")
BASELINE_ENV = os.path.join(HERE, "baseline.env")
RESULTS = os.path.join(HERE, "results.jsonl")
HEALTH = "http://localhost:8888/health"
METRICS = "http://localhost:8888/metrics"

LAUNCH_TIMEOUT = 1500      # 25 min: weight load is ~7 min; anything past this is a hang
BENCH_TOKENS = 400
BENCH_TASKS = 4


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd, timeout=None, check=False):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout, check=check)


def load_trials():
    """Minimal YAML reader for the trials.yaml shape we control."""
    import yaml  # PyYAML may be absent; fall back below
    with open(os.path.join(HERE, "trials.yaml")) as fh:
        return yaml.safe_load(fh)


def set_env_keys(path, overrides):
    """Set KEY=VALUE in a .env, replacing an existing assignment or appending."""
    with open(path) as fh:
        lines = fh.readlines()
    for key, val in overrides.items():
        pat = re.compile(rf"^{re.escape(key)}=")
        quoted = f'{key}="{val}"\n'
        for i, line in enumerate(lines):
            if pat.match(line):
                lines[i] = quoted
                break
        else:
            lines.append(quoted)
    with open(path, "w") as fh:
        fh.writelines(lines)


def healthy():
    try:
        with urllib.request.urlopen(HEALTH, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def stop_server():
    sh(f"cd {ROOT} && ./stop.sh", timeout=300)


def start_server(deadline):
    """Launch and wait for health. Returns (ok, seconds, reason)."""
    t0 = time.time()
    proc = subprocess.Popen(f"cd {ROOT} && ./start.sh --launch",
                            shell=True, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    while time.time() - t0 < deadline:
        if healthy():
            return True, time.time() - t0, "ok"
        # container died and start.sh gave up
        if proc.poll() is not None and not healthy():
            time.sleep(10)
            if healthy():
                return True, time.time() - t0, "ok"
            return False, time.time() - t0, "start.sh exited without a healthy server"
        time.sleep(15)
    proc.kill()
    return False, time.time() - t0, f"timeout after {deadline}s (probable hang)"


def scrape_spec():
    out = {}
    try:
        with urllib.request.urlopen(METRICS, timeout=20) as r:
            body = r.read().decode()
    except Exception:
        return out
    wanted = ("vllm:spec_decode_num_drafts_total",
              "vllm:spec_decode_num_draft_tokens_total",
              "vllm:spec_decode_num_accepted_tokens_total",
              "vllm:generation_tokens_total")
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^([a-z_:]+)\{([^}]*)\}\s+([0-9.eE+-]+)$", line.strip())
        if m and m.group(1) in wanted:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def wait_idle(max_wait=180):
    """Wait until no requests are running, so the benchmark starts clean."""
    t0 = time.time()
    while time.time() - t0 < max_wait:
        r = sh("docker logs --tail 3 vllm-fn 2>&1 | grep -oE 'Running: [0-9]+ reqs' | tail -1")
        m = re.search(r"(\d+)", r.stdout or "")
        if m and m.group(1) == "0":
            return
        time.sleep(5)


def benchmark(temp="0.0"):
    wait_idle()
    before = scrape_spec()
    r = sh(f"cd {ROOT} && python3 bench/decodebench.py --decode {BENCH_TOKENS} "
           f"--contexts 1000 --temps {temp}", timeout=1800)
    after = scrape_spec()

    rows = {}
    for line in (r.stdout or "").splitlines():
        m = re.search(r"\s(prose|code|entropy|copy)\s+[\d,]+\s+\d+\s+[\d.]+\s+([\d.]+)\s*$", line)
        if m:
            rows[m.group(1)] = float(m.group(2))

    def d(k):
        return after.get(k, 0.0) - before.get(k, 0.0)

    drafts, dtok, atok = d("vllm:spec_decode_num_drafts_total"), \
                         d("vllm:spec_decode_num_draft_tokens_total"), \
                         d("vllm:spec_decode_num_accepted_tokens_total")
    gen = d("vllm:generation_tokens_total")
    expected = BENCH_TOKENS * BENCH_TASKS
    return {
        "tok_s": rows,
        "mean_tok_s": round(sum(rows.values()) / len(rows), 2) if rows else None,
        "acceptance": round(atok / dtok, 4) if dtok else None,
        "accepted_per_draft": round(atok / drafts, 4) if drafts else None,
        "generation_tokens": int(gen),
        "clean": bool(gen and gen <= expected * 1.09),
        "expected_tokens": expected,
    }


def selected_backends():
    """Record what the engine actually chose, so a flag that silently did
    nothing is visible in the results rather than credited with a win."""
    r = sh("docker logs vllm-fn 2>&1 | grep -iE "
           "'moe_backend|linear_backend|MoE backend|Using .* MoE|draft vocab|"
           "argmax reduction|kv cache size' | tail -12")
    return (r.stdout or "").strip().splitlines()


def run_trial(name, env_over, extra_args, baseline_args):
    log(f"=== TRIAL {name} ===")
    shutil.copy(BASELINE_ENV, ENV)
    over = dict(env_over or {})
    over["EXTRA_VLLM_ARGS"] = extra_args if extra_args else baseline_args
    set_env_keys(ENV, over)

    stop_server()
    ok, secs, reason = start_server(LAUNCH_TIMEOUT)
    rec = {
        "trial": name,
        "ts": datetime.now(timezone.utc).isoformat(),
        "env_overrides": env_over or {},
        "extra_args": over["EXTRA_VLLM_ARGS"],
        "launch_ok": ok,
        "launch_seconds": round(secs, 1),
    }
    if not ok:
        rec["status"] = "FAILED"
        rec["reason"] = reason
        log(f"  FAILED: {reason}")
        tail = sh("docker logs vllm-fn 2>&1 | tail -40")
        rec["log_tail"] = (tail.stdout or "")[-4000:]
        stop_server()
    else:
        log(f"  healthy in {secs:.0f}s; benchmarking")
        rec["backends"] = selected_backends()
        rec.update(benchmark())
        rec["status"] = "OK" if rec.get("clean") else "DIRTY"
        log(f"  {rec['status']} mean={rec.get('mean_tok_s')} "
            f"acc={rec.get('acceptance')} {rec.get('tok_s')}")

    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def _restore_baseline_env():
    """Put the baseline .env back. Registered atexit and on SIGINT/SIGTERM so a
    killed sweep cannot strand a trial's .env -- otherwise the next manual launch
    silently inherits that trial's settings, which invalidates the comparison."""
    try:
        if os.path.exists(BASELINE_ENV):
            shutil.copy(BASELINE_ENV, ENV)
            log("baseline .env restored")
    except Exception as exc:
        log(f"WARNING: could not restore baseline .env: {exc}")


def _on_signal(signum, _frame):
    _restore_baseline_env()
    sys.exit(128 + signum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--repeat", type=int, default=1,
                    help="benchmark repetitions per trial (variance is real)")
    args = ap.parse_args()

    cfg = load_trials()
    base = cfg["baseline"]
    trials = cfg["trials"]

    if args.list:
        print(f"baseline: {base['env']} args={base.get('args','')}")
        for t in trials:
            print(f"  {t['name']:<26} {t.get('why','')}")
        return

    # Snapshot the baseline .env once, from the live .env plus declared baseline.
    if not os.path.exists(BASELINE_ENV):
        shutil.copy(ENV, BASELINE_ENV)
        set_env_keys(BASELINE_ENV, base["env"])
        set_env_keys(BASELINE_ENV, {"EXTRA_VLLM_ARGS": base.get("args", "")})
        log(f"captured baseline -> {BASELINE_ENV}")

    atexit.register(_restore_baseline_env)
    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, _on_signal)

    if args.restore:
        shutil.copy(BASELINE_ENV, ENV)
        stop_server()
        ok, secs, reason = start_server(LAUNCH_TIMEOUT)
        log(f"restore: {'ok' if ok else reason} in {secs:.0f}s")
        return

    picked = trials
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        picked = [t for t in trials if t["name"] in want]
    elif not args.all:
        ap.error("choose --all, --only NAME[,NAME...], --list or --restore")

    log(f"running {len(picked)} trial(s)")
    for t in picked:
        for _ in range(args.repeat):
            run_trial(t["name"], t.get("env"), t.get("args"), base.get("args", ""))

    # Always leave the box on the baseline, serving.
    log("sweep done; restoring baseline")
    shutil.copy(BASELINE_ENV, ENV)
    stop_server()
    ok, secs, reason = start_server(LAUNCH_TIMEOUT)
    log(f"baseline restored: {'ok' if ok else reason}")


if __name__ == "__main__":
    main()
