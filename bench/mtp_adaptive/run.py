# SPDX-License-Identifier: AGPL-3.0-or-later
"""Paired, no-reload fixed-K3 vs adaptive-K benchmark. Synthetic prompts only.

Mia/sparkDash protocol: temperature=0, top_p=1, thinking off, 600 tokens,
32-token warmup, C>1 stream suffix, post-first-token SSE decode timing.
Mia prompt strings are from sparkDash (MIT), src/shared/llmPrompts.js.
Both arms retain IndexShare/max-K4 graphs; NOT an unmodified-upstream A/B.
See THIRD_PARTY_NOTICES.md for the sparkDash prompt license.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shlex
import statistics
import subprocess
import threading
import time
import urllib.request

BASE = 'http://127.0.0.1:8888'
CONTROL = Path.home() / '.cache/vllm/mtp-adaptive/control.json'
MODEL = 'qwen3.8-flash-next'
KEY_ENV = 'VLLM_API_KEY'
WORKER = None
HERE = Path(__file__).resolve().parent
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PROMPTS = {
    'mia_prose': 'Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough.',
    'mia_code': 'Output only Python source code. No comments, no docstrings, no markdown fences. Write functions clamp_00 through clamp_49. Each function is exactly:\ndef clamp_NN(x, lo=0, hi=1):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\nChange only the function name suffix (00, 01, … 49). One blank line between functions. No other text.',
    'mia_json': 'Emit only a JSON array of fake GPU metrics rows. Each object needs host, gpuIndex, utilPct, tempC, powerW, memUsedMb. Invent many rows. No markdown. Keep expanding the array.',
    'mia_structured': 'Count from 1 to 200. Output only the numbers, separated by spaces. No other text.',
    'zh_prose': '请用连贯的中文解释为什么在写代码时，缓存命中率提高了，用户感受到的响应速度却不一定变快。结合排队、预填充、逐字生成和网络传输说明，给出具体例子，不要只列提纲。',
    'thinking': 'Reason carefully about designing a correct incremental build cache for a Python monorepo. Consider file changes, deleted dependencies, concurrent writers, and stale results. Then give an implementation plan.',
}
PROMPTS['zh_long'] = ''.join(f'Archive entry {i:05d}: A review of the software project recorded delivery dates, cache statistics, routine maintenance and observations from the engineering team.\n' for i in range(950)) + '\n\n' + PROMPTS['zh_prose']
MATRIX = [(name, c) for name in ('mia_prose', 'mia_code', 'mia_json', 'mia_structured') for c in (1, 2, 4, 8)]
MATRIX += [('zh_prose', 1), ('zh_prose', 4), ('zh_long', 1), ('thinking', 1)]


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def key():
    return os.environ.get(KEY_ENV, '')


def metrics():
    raw = HTTP.open(BASE + '/metrics', timeout=10).read().decode()
    out = {}
    for line in raw.splitlines():
        m = re.match(r'^(vllm:[\w:]+)(?:\{([^}]*)\})?\s+(\S+)$', line)
        if not m or m[1].endswith(('_bucket', '_created')):
            continue
        name, labels = m[1], m[2] or ''
        if 'per_pos' in name:
            p = re.search(r'position="(\d+)"', labels)
            if p:
                name += ':' + p[1]
        out[name] = out.get(name, 0) + float(m[3])
    return out


def idle():
    m = metrics()
    assert m.get('vllm:num_requests_running') == 0 and m.get('vllm:num_requests_waiting') == 0, 'Foreign traffic or unfinished request; pause benchmark'
    return m


def request(prompt, thinking, limit, api_key, index, barrier=None):
    body = dict(model=MODEL, messages=[dict(role='user', content=prompt)],
                temperature=0, top_p=1, max_tokens=limit, min_tokens=limit, ignore_eos=True,
                stop=[], stream=True, stream_options=dict(include_usage=True),
                chat_template_kwargs=dict(enable_thinking=thinking))
    # No seed override: matches sparkDash's greedy protocol.
    if thinking:
        body['chat_template_kwargs']['reasoning_effort'] = 'xhigh'
    req = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key})
    if barrier:
        barrier.wait(timeout=15)
    start = time.perf_counter()
    first = last = None
    usage = None
    done = False
    pieces, thoughts, arrivals = [], [], []
    # Use a separate opener per stream (no shared connection state).
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=360) as response:
        buffer = b''
        while chunk := response.read1(65536):
            buffer += chunk
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    continue
                obj = json.loads(data)
                assert not obj.get('error'), 'Error inside SSE stream'
                usage = obj.get('usage') or usage
                for choice in obj.get('choices', []):
                    delta = choice.get('delta') or {}
                    content = delta.get('content') or ''
                    thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                    if content or thought:
                        last = time.perf_counter()
                        first = last if first is None else first
                        arrivals.append(last)
                        pieces.append(content)
                        thoughts.append(thought)
    end = time.perf_counter()
    assert done and usage and first is not None and last > first, 'Incomplete stream'
    assert usage['completion_tokens'] == limit, 'Wrong completion length'
    text, thought = ''.join(pieces), ''.join(thoughts)
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    return dict(index=index, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                prompt_tokens=usage['prompt_tokens'], completion_tokens=limit,
                first=first, last=last, start=start, end=end,
                decode_tps=(limit - 1) / (last - first), ttft_s=first - start,
                wall_s=end - start, max_chunk_gap_s=max(gaps, default=0),
                text=text, reasoning=thought, text_sha256=hashlib.sha256((thought + text).encode()).hexdigest(),
                usage=usage)


def wave(name, concurrency, limit, api_key, rolling=False):
    before = idle()
    if rolling:
        tasks = [('mia_prose', 320), ('zh_prose', 640), ('mia_code', 512), ('mia_json', 256)] * 4
        jobs = [(PROMPTS[n] + f' (job {i + 1}/{len(tasks)})', False, l, api_key, i, None)
                for i, (n, l) in enumerate(tasks)]
    else:
        barrier = threading.Barrier(concurrency)
        jobs = [(PROMPTS[name] + (f' (stream {i+1}/{concurrency})' if concurrency > 1 else ''),
                 name == 'thinking', limit, api_key, i, barrier) for i in range(concurrency)]
    started = time.time()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(request, *job) for job in jobs]
        rows = [f.result() for f in cf.as_completed(futures)]
    rows.sort(key=lambda x: x['index'])
    expected = sum(r['completion_tokens'] for r in rows)
    for _ in range(60):
        after = metrics()
        if after.get('vllm:request_success_total', 0) - before.get('vllm:request_success_total', 0) >= len(rows):
            break
        time.sleep(.2)
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)}
    isolated = delta.get('vllm:generation_tokens_total') == expected and delta.get('vllm:request_success_total') == len(rows)
    drafts = delta.get('vllm:spec_decode_num_drafts_total', 0)
    drafted = delta.get('vllm:spec_decode_num_draft_tokens_total', 0)
    accepted = delta.get('vllm:spec_decode_num_accepted_tokens_total', 0)
    total_window = max(r['last'] for r in rows) - min(r['first'] for r in rows)
    result = dict(case=name, concurrency=concurrency, rolling=rolling, limit=limit, started=started,
                  ended=time.time(), streams=rows, isolated=isolated,
                  aggregate_decode_tps=sum(r['completion_tokens'] - 1 for r in rows) / total_window,
                  mean_stream_tps=statistics.mean(r['decode_tps'] for r in rows),
                  mean_ttft_s=statistics.mean(r['ttft_s'] for r in rows),
                  e2e_tps=expected / (max(r['end'] for r in rows) - min(r['start'] for r in rows)),
                  mean_k=drafted / drafts if drafts else None,
                  accepted_per_step=accepted / drafts if drafts else None,
                  delta={k: v for k, v in delta.items() if 'spec_decode' in k or k in
                         ('vllm:num_preemptions_total', 'vllm:generation_tokens_total', 'vllm:request_success_total',
                          'vllm:inter_token_latency_seconds_sum', 'vllm:inter_token_latency_seconds_count')})
    return result


def monitor(stop, out):
    # Whitelisted read-only fields. Never read configuration/env/key files.
    script = "nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv,noheader,nounits; awk '/^MemAvailable:/ {print $2}' /proc/meminfo"
    with (out / 'hardware.jsonl').open('a') as f:
        while not stop.is_set():
            row = {'time': time.time()}
            for node in (('head', 'worker') if WORKER else ('head',)):
                cmd = ['bash', '-c', script] if node == 'head' else ['ssh', '-n', WORKER, script]
                try:
                    p = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
                    row[node] = p.stdout.strip().splitlines() if p.returncode == 0 else ['unavailable']
                except subprocess.TimeoutExpired:
                    row[node] = ['timeout']
            f.write(json.dumps(row) + '\n')
            f.flush()
            stop.wait(3)


def mode(arm):
    idle()
    CONTROL.parent.mkdir(parents=True, exist_ok=True)
    atomic(CONTROL, {'mode': 'fixed' if arm == 'fixed3' else 'adaptive', 'k': 3})
    time.sleep(1.1)  # Controller refresh period; next scheduler call reads this.


def main():
    global BASE, CONTROL, MODEL, KEY_ENV, WORKER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New output directory; never overwritten')
    parser.add_argument('--base-url', default=BASE)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--control', type=Path, default=CONTROL, help='Host-side hot-control file')
    parser.add_argument('--api-key-env', default=KEY_ENV, help='Environment variable containing API key')
    parser.add_argument('--monitor-hardware', action='store_true')
    parser.add_argument('--worker', help='Optional SSH destination for passive hardware samples')
    args = parser.parse_args()
    BASE, CONTROL, MODEL, KEY_ENV, WORKER = args.base_url.rstrip('/'), args.control, args.model, args.api_key_env, args.worker
    os.umask(0o077)
    out = Path(args.out)
    out.mkdir()  # Deliberately refuse overwriting/replaying a completed run.
    api_key = key()
    idle()
    atomic(out / 'initial_control.json', json.loads(CONTROL.read_text()) if CONTROL.exists() else {'mode': 'adaptive'})
    protocol = dict(repeats=6, primary_matrix=MATRIX, tokens=600, temperature=0, top_p=1,
                    seed='unspecified (sparkDash protocol)', warmup_tokens=32,
                    baseline='IndexShare ON + current maxK4 runner + fixedK3',
                    candidate='same loaded service + v3 adaptive K1/2/3/4',
                    pair_orders=['AB', 'BA', 'AB', 'BA', 'BA', 'AB'],
                    statistic='all samples; paired geometric-mean ratio with bootstrap CI; trimmed range secondary',
                    scope='isolates controller only, not unmodified upstream; no retuning during run',
                    rolling='3 paired repetitions, 16 mixed jobs, 4 rolling slots; end-to-end throughput primary',
                    stop_on='API error, foreign traffic, preemption or STOP file; no model/config auto-rollback')
    atomic(out / 'protocol.json', protocol)
    atomic(out / 'prompts.json', PROMPTS)
    stop = threading.Event()
    mon = threading.Thread(target=monitor, args=(stop, out), daemon=True)
    if args.monitor_hardware:
        mon.start()
    completed = 0
    total = 6 * len(MATRIX) * 2 + 6
    state = dict(complete=False, completed=0, total=total, phase='preflight', started=time.time())
    def save_state(**kw):
        state.update(kw, updated=time.time(), control=json.loads(CONTROL.read_text()) if CONTROL.exists() else {'mode': 'adaptive'})
        atomic(out / 'state.json', state)
    def store_row(row, name):
        with (out / name).open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
        assert row['isolated'], 'Foreign traffic; result preserved but invalid'
        assert row['delta'].get('vllm:num_preemptions_total', 0) == 0, 'Preemption; stop'
    def one(arm, case, concurrency, rep, rolling=False):
        nonlocal completed
        assert not (out / 'STOP').exists(), 'User STOP marker'
        save_state(phase='running', arm=arm, case=case, concurrency=concurrency, rep=rep)
        mode(arm)
        warm = wave('mia_prose' if rolling else case, concurrency, 32, api_key)
        warm.update(arm=arm, rep=rep)
        store_row(warm, 'warmups.jsonl')
        row = wave(case, concurrency, 600, api_key, rolling)
        row.update(arm=arm, rep=rep)
        store_row(row, 'waves.jsonl')
        if arm == 'fixed3':
            assert abs(row['mean_k'] - 3) < 1e-9, 'Fixed arm not actually K=3'
        completed += 1
        save_state(completed=completed)
        print(json.dumps({k: row[k] for k in ('case', 'concurrency', 'arm', 'rep', 'aggregate_decode_tps',
                                               'mean_stream_tps', 'mean_ttft_s', 'mean_k')}) + f' [{completed}/{total}]', flush=True)
    try:
        save_state()
        # Deterministic cell shuffling; fixed, balanced AB/BA arm order.
        for rep in range(6):
            cells = MATRIX.copy()
            random.Random(20260922 + rep).shuffle(cells)
            arms = ('fixed3', 'adaptive') if rep in (0, 2, 5) else ('adaptive', 'fixed3')
            for case, concurrency in cells:
                for arm in arms:
                    one(arm, case, concurrency, rep + 1)
        for rep in range(3):
            arms = ('adaptive', 'fixed3') if rep == 1 else ('fixed3', 'adaptive')
            for arm in arms:
                one(arm, 'mixed_rolling', 4, rep + 1, rolling=True)
        idle()
        assert HTTP.open(BASE + '/health', timeout=10).status == 200
        save_state(complete=True, phase='complete', finished=time.time())
        print('DEEP_BENCH_COMPLETE_LAST_ARM_ADAPTIVE', flush=True)
    except Exception as exc:
        save_state(phase='stopped', failure=type(exc).__name__ + ': ' + str(exc))
        print('DEEP_BENCH_STOPPED', type(exc).__name__, str(exc), flush=True)
        raise
    finally:
        # Do not silently restore a mode or model on failure/completion.
        stop.set()
        if args.monitor_hardware:
            mon.join(timeout=14)


if __name__ == '__main__':
    main()
