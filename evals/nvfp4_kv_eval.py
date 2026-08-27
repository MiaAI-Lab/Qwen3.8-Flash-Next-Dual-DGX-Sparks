#!/usr/bin/env python3
"""NVFP4 KV-cache reliability eval against a live SGLang OpenAI API.

The CUDA-graph path is already verified bit-exact on SM121. This script
asks the question the ~9% relative K/V error actually raises: can the
model still retrieve a high-entropy passkey planted at known depths?

Suites
  quick  control + exact copy + NIAH at 1k/4k/16k + 4k radix follow-up
  full   quick + 32k/64k NIAH (three positions each)
  long   full + 128k (GB10-safe with chunk 1024; do not raise further)

Usage
  python3 evals/nvfp4_kv_eval.py
  python3 evals/nvfp4_kv_eval.py --suite full --json kv-eval.json
  ./start.sh kv-eval --suite quick --require-nvfp4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

ADJECTIVES = (
    "amber", "brine", "cedar", "dun", "ember", "flint", "gale", "haze",
    "ivory", "jade", "kestrel", "linen", "moss", "nimbus", "onyx", "pearl",
    "quartz", "russet", "slate", "topaz", "umber", "violet", "willow", "xenon",
)
NOUNS = (
    "barrel", "crate", "dock", "gantry", "hopper", "inlet", "jig", "keel",
    "latch", "mast", "nozzle", "pallet", "quay", "reel", "silo", "truss",
)
ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"

SUITES = {
    "quick": {"depths": (1024, 4096, 16384), "positions": (0.0, 0.5, 1.0)},
    "full": {"depths": (1024, 4096, 16384, 32768, 65536), "positions": (0.0, 0.5, 1.0)},
    "long": {
        "depths": (1024, 4096, 16384, 32768, 65536, 131072),
        "positions": (0.0, 0.5, 1.0),
    },
}

# Depths small enough that a single position (middle) is enough.
SHALLOW = 2048


@dataclass
class PoolInfo:
    kv_cache_dtype: str
    max_total_num_tokens: Optional[float]
    kv_cache_memory_gb: Optional[float]
    kv_available_tokens: Optional[float]
    context_length: Optional[int]
    model: str
    nvfp4: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    name: str
    kind: str
    passed: bool
    needle: str
    output: str
    depth_tokens: int
    position: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    latency_s: float
    ttft_s: Optional[float]
    prefill_tps: Optional[float]
    decode_tps: Optional[float]
    cache_hit: Optional[str] = None
    error: Optional[str] = None


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _color(enabled: bool, code: str, text: str) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def _http_json(
    url: str,
    *,
    payload: Optional[dict[str, Any]] = None,
    timeout: float = 30,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="GET" if payload is None else "POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _load_json(url: str, timeout: float = 10) -> Any:
    with _http_json(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        # sglang:name{labels} value
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{[^}]*\}\s+(\S+)$", line)
        if not m:
            m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(\S+)$", line)
        if not m:
            continue
        name, raw = m.group(1), m.group(2)
        try:
            val = float(raw)
        except ValueError:
            continue
        # last sample wins (single-rank server)
        out[name] = val
    return out


def detect_pool(base: str) -> PoolInfo:
    notes: list[str] = []
    info: dict[str, Any] = {}
    try:
        info = _load_json(f"{base}/get_server_info")
    except Exception as exc:  # noqa: BLE001 — any transport failure is a note
        notes.append(f"get_server_info failed: {exc}")
        try:
            models = _load_json(f"{base}/v1/models")
            model = models["data"][0]["id"]
            ctx = models["data"][0].get("max_model_len")
        except Exception as exc2:  # noqa: BLE001
            raise SystemExit(f"API not reachable at {base}: {exc2}") from exc2
        info = {"served_model_name": model, "context_length": ctx, "kv_cache_dtype": "unknown"}

    metrics: dict[str, float] = {}
    try:
        with _http_json(f"{base}/metrics", timeout=10) as resp:
            metrics = _parse_metrics(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        notes.append(f"metrics failed: {exc}")

    dtype = str(info.get("kv_cache_dtype") or "unknown")
    max_tokens = metrics.get("sglang:max_total_num_tokens")
    mem_gb = metrics.get("sglang:kv_cache_memory_usage_gb")
    avail = metrics.get("sglang:kv_available_tokens")
    ctx = info.get("context_length")
    model = str(info.get("served_model_name") or info.get("model_path") or "")

    nvfp4 = dtype.lower() == "nvfp4"
    if not nvfp4 and max_tokens is not None:
        # bf16 pool on this recipe was ~925k; nvfp4 is ~3x that at the same
        # mem-fraction. Treat a clearly larger pool as circumstantial evidence.
        if max_tokens >= 1_800_000:
            nvfp4 = True
            notes.append(
                f"kv_cache_dtype={dtype} but pool is {max_tokens:.0f} tokens "
                "(≥1.8M) — treating as NVFP4"
            )
        else:
            notes.append(
                f"kv_cache_dtype={dtype}, pool={max_tokens:.0f} tokens "
                "(bf16 on this recipe is ~925k). NVFP4 is probably off."
            )
    return PoolInfo(
        kv_cache_dtype=dtype,
        max_total_num_tokens=max_tokens,
        kv_cache_memory_gb=mem_gb,
        kv_available_tokens=avail,
        context_length=int(ctx) if ctx else None,
        model=model,
        nvfp4=nvfp4,
        notes=notes,
    )


def metric_snapshot(base: str) -> dict[str, float]:
    try:
        with _http_json(f"{base}/metrics", timeout=10) as resp:
            return _parse_metrics(resp.read().decode())
    except Exception:  # noqa: BLE001
        return {}


def make_passkey(rng: random.Random) -> str:
    parts = ["".join(rng.choice(ALPHABET) for _ in range(4)) for _ in range(3)]
    return "-".join(parts)


def filler_line(i: int) -> str:
    adj = ADJECTIVES[i % len(ADJECTIVES)]
    noun = NOUNS[i % len(NOUNS)]
    serial = hashlib.md5(str(i).encode()).hexdigest()[:8]
    return (
        f"Record {i:06d}: crate {adj}-{noun} serial {serial} sits idle "
        "and is unrelated to any passkey or secret code.\n"
    )


class TokenEstimator:
    """Estimate token counts. Prefers /tokenize; falls back to two-point chat."""

    def __init__(self, client: "ChatClient"):
        self.client = client
        self._mode = "unknown"
        self._tokens_per_line: Optional[float] = None
        self._chat_overhead: Optional[int] = None

    def warmup(self) -> None:
        sample = "".join(filler_line(10_000 + i) for i in range(8))
        n = self._try_tokenize(sample)
        if n is not None:
            self._mode = "tokenize"
            self._tokens_per_line = n / 8
            return
        a = self.client.prompt_tokens_of("".join(filler_line(20_000 + i) for i in range(8)))
        b = self.client.prompt_tokens_of("".join(filler_line(20_000 + i) for i in range(24)))
        if a is None or b is None or b <= a:
            self._mode = "heuristic"
            self._tokens_per_line = 22.0  # measured-ish for this filler on Qwen
            return
        self._mode = "chat-diff"
        self._tokens_per_line = (b - a) / 16
        self._chat_overhead = int(round(a - 8 * self._tokens_per_line))

    def _try_tokenize(self, text: str) -> Optional[int]:
        for path, body in (
            ("/tokenize", {"text": text}),
            ("/tokenize", {"prompt": text}),
            ("/v1/tokenize", {"prompt": text, "model": self.client.model}),
        ):
            try:
                with _http_json(
                    f"{self.client.base}{path}", payload=body, timeout=30
                ) as resp:
                    obj = json.loads(resp.read().decode())
            except Exception:  # noqa: BLE001
                continue
            for key in ("count", "tokens", "num_tokens", "input_tokens"):
                if isinstance(obj.get(key), int):
                    return int(obj[key])
            ids = obj.get("tokens") or obj.get("input_ids")
            if isinstance(ids, list):
                return len(ids)
        return None

    def lines_for(self, target_tokens: int) -> int:
        tpl = self._tokens_per_line or 22.0
        return max(1, int(round(target_tokens / tpl)))

    @property
    def mode(self) -> str:
        return self._mode


class ChatClient:
    def __init__(
        self,
        base: str,
        model: str,
        *,
        timeout: float,
        thinking: bool,
        max_tokens: int,
        temperature: float,
    ):
        self.base = base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.temperature = temperature

    def prompt_tokens_of(self, content: str) -> Optional[int]:
        try:
            result = self.complete(
                [{"role": "user", "content": content}],
                max_tokens=1,
                stream=False,
                timeout=min(120.0, self.timeout),
            )
        except Exception:  # noqa: BLE001
            return None
        return result.get("prompt_tokens")

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        stream: bool = True,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        t0 = time.perf_counter()
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            return self._stream(payload, t0, timeout or self.timeout)
        with _http_json(
            f"{self.base}/v1/chat/completions",
            payload=payload,
            timeout=timeout or self.timeout,
        ) as resp:
            obj = json.loads(resp.read().decode())
        latency = time.perf_counter() - t0
        return self._from_nonstream(obj, latency)

    def _stream(
        self, payload: dict[str, Any], t0: float, timeout: float
    ) -> dict[str, Any]:
        content: list[str] = []
        reasoning: list[str] = []
        usage: dict[str, Any] = {}
        ttft: Optional[float] = None
        finish = None
        try:
            with _http_json(
                f"{self.base}/v1/chat/completions",
                payload=payload,
                timeout=timeout,
            ) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    rpiece = delta.get("reasoning_content") or ""
                    if (piece or rpiece) and ttft is None:
                        ttft = time.perf_counter() - t0
                    if piece:
                        content.append(piece)
                    if rpiece:
                        reasoning.append(rpiece)
                    finish = choices[0].get("finish_reason") or finish
        except urllib.error.HTTPError:
            # some builds reject stream_options — retry once without streaming
            payload = dict(payload)
            payload.pop("stream", None)
            payload.pop("stream_options", None)
            with _http_json(
                f"{self.base}/v1/chat/completions",
                payload=payload,
                timeout=timeout,
            ) as resp:
                obj = json.loads(resp.read().decode())
            return self._from_nonstream(obj, time.perf_counter() - t0)
        latency = time.perf_counter() - t0
        text = "".join(content)
        reason = "".join(reasoning)
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        prefill_tps = (pt / ttft) if pt and ttft else None
        decode_window = (latency - ttft) if ttft is not None else None
        decode_tps = (
            (ct / decode_window) if ct and decode_window and decode_window > 0 else None
        )
        return {
            "text": text,
            "reasoning": reason,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "latency_s": latency,
            "ttft_s": ttft,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "finish_reason": finish,
        }

    @staticmethod
    def _from_nonstream(obj: dict[str, Any], latency: float) -> dict[str, Any]:
        msg = (obj.get("choices") or [{}])[0].get("message") or {}
        usage = obj.get("usage") or {}
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        return {
            "text": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "latency_s": latency,
            "ttft_s": None,
            "prefill_tps": (pt / latency) if pt and latency else None,
            "decode_tps": None,
            "finish_reason": (obj.get("choices") or [{}])[0].get("finish_reason"),
        }


def compact_alnum(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def needle_in(text: str, needle: str) -> bool:
    if not text or not needle:
        return False
    return compact_alnum(needle) in compact_alnum(text)


def combined_text(result: dict[str, Any]) -> str:
    return " ".join(
        p for p in (result.get("text") or "", result.get("reasoning") or "") if p
    )


def haystack(n_lines: int, start: int, needle: str, position: float) -> str:
    lines = [filler_line(start + i) for i in range(n_lines)]
    insert = (
        f"<<IMPORTANT>> The magic passkey is {needle}. "
        "Remember it; a question at the end will ask for it. <</IMPORTANT>>\n"
    )
    idx = int(round(position * n_lines))
    idx = min(max(idx, 0), n_lines)
    lines.insert(idx, insert)
    return "".join(lines)


def ask_passkey(hay: str) -> str:
    return (
        hay
        + "\nQuestion: What is the magic passkey written in the records above?\n"
        "Rules: Reply with only the passkey (the three hyphenated groups). "
        "Do not quote any crate serial.\n"
    )


def case_from_chat(
    name: str,
    kind: str,
    needle: str,
    result: dict[str, Any],
    *,
    depth: int,
    position: Optional[float],
    extra_text: str = "",
    cache_hit: Optional[str] = None,
    error: Optional[str] = None,
) -> CaseResult:
    blob = combined_text(result) + (" " + extra_text if extra_text else "")
    passed = bool(needle_in(blob, needle)) if not error else False
    return CaseResult(
        name=name,
        kind=kind,
        passed=passed,
        needle=needle,
        output=(result.get("text") or result.get("reasoning") or error or "")[:500],
        depth_tokens=depth,
        position=position,
        prompt_tokens=result.get("prompt_tokens"),
        completion_tokens=result.get("completion_tokens"),
        latency_s=float(result.get("latency_s") or 0.0),
        ttft_s=result.get("ttft_s"),
        prefill_tps=result.get("prefill_tps"),
        decode_tps=result.get("decode_tps"),
        cache_hit=cache_hit,
        error=error,
    )


def run(args: argparse.Namespace) -> int:
    color = sys.stdout.isatty() and not args.no_color
    client = ChatClient(
        args.base_url,
        args.model,
        timeout=args.timeout,
        thinking=args.thinking,
        max_tokens=args.max_tokens,
        temperature=0.0,
    )
    rng = random.Random(args.seed)

    print(_color(color, "1;37", f"NVFP4 KV eval  { _utc() }"))
    print(f"  endpoint  {client.base}")
    pool = detect_pool(client.base)
    if not args.model:
        client.model = pool.model or "Qwen3.8-Flash-Next-NVFP4"
    print(f"  model     {client.model}")
    print(
        f"  kv dtype  {pool.kv_cache_dtype}   "
        f"pool={_fmt(pool.max_total_num_tokens)} tokens   "
        f"mem={_fmt(pool.kv_cache_memory_gb)} GB   "
        f"free={_fmt(pool.kv_available_tokens)}"
    )
    if pool.context_length:
        print(f"  context   {pool.context_length}")
    for note in pool.notes:
        print("  note     ", note)

    if args.require_nvfp4 and not pool.nvfp4:
        print(
            _color(
                color,
                "1;31",
                "NVFP4 KV is not on this process (dtype "
                f"{pool.kv_cache_dtype}, pool {_fmt(pool.max_total_num_tokens)}). "
                "Fix .env (first assignment wins) and restart.",
            )
        )
        return 2
    if pool.nvfp4:
        print(_color(color, "1;32", "  pool looks like NVFP4 KV"))
    else:
        print(
            _color(
                color,
                "1;33",
                "  WARNING: this looks like bf16 KV — results are a baseline, "
                "not an NVFP4 measurement. Pass --require-nvfp4 to abort.",
            )
        )

    print("  calibrating token estimator…")
    est = TokenEstimator(client)
    est.warmup()
    print(f"  estimator {est.mode}  (~{(est._tokens_per_line or 0):.1f} tok/line)")

    depths: tuple[int, ...]
    positions: tuple[float, ...]
    if args.depths:
        depths = tuple(int(x) for x in args.depths.split(",") if x)
        positions = tuple(float(x) for x in (args.positions or "0,0.5,1").split(","))
    else:
        spec = SUITES[args.suite]
        depths = spec["depths"]  # type: ignore[assignment]
        positions = spec["positions"]  # type: ignore[assignment]

    ctx = pool.context_length or 900_000
    depths = tuple(d for d in depths if d + 256 < ctx)
    if not depths:
        print("no depths fit in context_length")
        return 2

    results: list[CaseResult] = []

    def report(case: CaseResult) -> None:
        results.append(case)
        mark = _color(color, "1;32", "PASS") if case.passed else _color(color, "1;31", "FAIL")
        pos = "" if case.position is None else f" @{case.position:.0%}"
        tps = ""
        if case.prefill_tps:
            tps += f"  prefill {case.prefill_tps:.0f} tok/s"
        if case.decode_tps:
            tps += f"  decode {case.decode_tps:.0f} tok/s"
        pt = f"{case.prompt_tokens} tok" if case.prompt_tokens else f"~{case.depth_tokens} tok"
        print(
            f"  {mark}  {case.name:28s}  {pt:>10s}{pos:5s}  "
            f"{case.latency_s:6.1f}s{tps}"
        )
        if not case.passed:
            got = (case.output or case.error or "").replace("\n", " ")[:160]
            print(f"         expected {case.needle}")
            print(f"         got      {got}")

    # --- control: needle only, no haystack (prompt/setup, not KV depth) ------
    code = make_passkey(rng)
    prompt = (
        f"The magic passkey is {code}.\n"
        "Question: What is the magic passkey? Reply with only the passkey.\n"
    )
    try:
        r = client.complete([{"role": "user", "content": prompt}])
        report(
            case_from_chat(
                "control/no-haystack", "control", code, r, depth=r.get("prompt_tokens") or 0,
                position=None,
            )
        )
    except Exception as exc:  # noqa: BLE001
        report(
            CaseResult(
                name="control/no-haystack",
                kind="control",
                passed=False,
                needle=code,
                output="",
                depth_tokens=0,
                position=None,
                prompt_tokens=None,
                completion_tokens=None,
                latency_s=0.0,
                ttft_s=None,
                prefill_tps=None,
                decode_tps=None,
                error=str(exc),
            )
        )
        print("control failed — not scoring retrieval (server/prompt issue)")
        return _finish(args, pool, results, color, setup_failed=True)

    # --- exact copy (decode path) --------------------------------------------
    token = "NVFP4-OK-" + make_passkey(rng).split("-")[0]
    prompt = (
        f"Reply with exactly this string and nothing else: {token}\n"
    )
    r = client.complete([{"role": "user", "content": prompt}])
    report(case_from_chat("decode/exact-copy", "decode", token, r, depth=0, position=None))

    # --- NIAH ----------------------------------------------------------------
    for depth in depths:
        pos_list: Iterable[float]
        if depth <= SHALLOW:
            pos_list = (0.5,)
        else:
            pos_list = positions
        start = rng.randint(10_000, 80_000)
        n_lines = est.lines_for(depth)
        for pos in pos_list:
            code = make_passkey(rng)
            hay = haystack(n_lines, start, code, pos)
            name = f"niah/{depth}@{pos:.0%}"
            try:
                r = client.complete([{"role": "user", "content": ask_passkey(hay)}])
                report(
                    case_from_chat(
                        name, "niah", code, r, depth=depth, position=pos
                    )
                )
            except Exception as exc:  # noqa: BLE001
                report(
                    CaseResult(
                        name=name,
                        kind="niah",
                        passed=False,
                        needle=code,
                        output="",
                        depth_tokens=depth,
                        position=pos,
                        prompt_tokens=None,
                        completion_tokens=None,
                        latency_s=0.0,
                        ttft_s=None,
                        prefill_tps=None,
                        decode_tps=None,
                        error=str(exc),
                    )
                )

    # --- multi-fact binding at ~8k ------------------------------------------
    bind_depth = 8192 if 8192 <= max(depths) or args.suite != "quick" else 4096
    if bind_depth <= max(depths) or args.suite != "quick":
        n_lines = est.lines_for(min(bind_depth, max(depths)))
        start = rng.randint(10_000, 80_000)
        lines = [filler_line(start + i) for i in range(n_lines)]
        facts = [
            (0.15, "Dr. Helene Voss works in Port Kestrel. Her badge is XV-4419.\n"),
            (0.50, "The night clerk at Pier 12 is named Ilya Strom. Ignore that.\n"),
            (0.85, "Port Kestrel's warehouse lead is the same Dr. Helene Voss.\n"),
        ]
        needle = "XV-4419"
        for frac, text in sorted(facts, key=lambda x: x[0], reverse=True):
            lines.insert(int(frac * n_lines), text)
        prompt = (
            "".join(lines)
            + "\nQuestion: What is the badge number of the person in Port Kestrel?\n"
            "Reply with only the badge.\n"
        )
        r = client.complete([{"role": "user", "content": prompt}])
        report(
            case_from_chat(
                "binding/port-kestrel",
                "binding",
                needle,
                r,
                depth=bind_depth,
                position=0.15,
            )
        )

    # --- radix follow-up: retrieve on turn 2 after a cached prefix ------------
    rad_depth = 4096 if 4096 in depths else depths[min(1, len(depths) - 1)]
    n_lines = est.lines_for(rad_depth)
    start = rng.randint(10_000, 80_000)
    code = make_passkey(rng)
    hay = haystack(n_lines, start, code, 0.5)
    turn1 = (
        hay
        + "\nDo not mention any passkey yet. Reply with the single word READY.\n"
    )
    before = metric_snapshot(client.base)
    r1 = client.complete([{"role": "user", "content": turn1}])
    messages = [
        {"role": "user", "content": turn1},
        {"role": "assistant", "content": r1.get("text") or "READY"},
        {
            "role": "user",
            "content": (
                "What is the magic passkey written in the earlier records? "
                "Reply with only the passkey."
            ),
        },
    ]
    r2 = client.complete(messages)
    after = metric_snapshot(client.base)
    cache_note = None
    dh0 = before.get("sglang:cache_hit_rate")
    dh1 = after.get("sglang:cache_hit_rate")
    if dh0 is not None and dh1 is not None:
        cache_note = f"cache_hit_rate {dh0:.3f} → {dh1:.3f}"
    report(
        case_from_chat(
            f"radix/follow-up/{rad_depth}",
            "radix",
            code,
            r2,
            depth=rad_depth,
            position=0.5,
            extra_text="",
            cache_hit=cache_note,
        )
    )
    if cache_note:
        print(f"         {cache_note}")

    return _finish(args, pool, results, color, setup_failed=False)


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "?"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.2f}"


def _finish(
    args: argparse.Namespace,
    pool: PoolInfo,
    results: list[CaseResult],
    color: bool,
    *,
    setup_failed: bool,
) -> int:
    niah = [c for c in results if c.kind == "niah"]
    niah_pass = sum(1 for c in niah if c.passed)
    by_depth: dict[int, list[bool]] = {}
    for c in niah:
        by_depth.setdefault(c.depth_tokens, []).append(c.passed)

    print()
    print(_color(color, "1;37", "Summary"))
    print(f"  pool     dtype={pool.kv_cache_dtype}  nvfp4={pool.nvfp4}  "
          f"tokens={_fmt(pool.max_total_num_tokens)}  mem={_fmt(pool.kv_cache_memory_gb)} GB")
    print(f"  cases    {sum(c.passed for c in results)}/{len(results)} passed")
    if niah:
        print(f"  niah     {niah_pass}/{len(niah)} passed")
        for depth in sorted(by_depth):
            ok = sum(by_depth[depth])
            n = len(by_depth[depth])
            print(f"           {depth:>6d} tok  {ok}/{n}")

    control_ok = all(c.passed for c in results if c.kind == "control")
    if setup_failed or not control_ok:
        verdict = "SETUP-FAIL"
        rc = 2
        why = "control/prompt failed — do not attribute this to NVFP4 KV"
    elif not niah:
        verdict = "NO-NIAH"
        rc = 2
        why = "no retrieval cases ran"
    elif niah_pass == len(niah) and all(c.passed for c in results):
        verdict = "RELIABLE"
        rc = 0
        why = "every planted passkey came back exactly"
    elif niah_pass == len(niah):
        verdict = "RELIABLE-RETRIEVAL"
        rc = 0
        why = "all NIAH passed; a non-retrieval case failed"
    elif niah_pass >= max(1, (len(niah) + 1) // 2):
        failed = sorted({c.depth_tokens for c in niah if not c.passed})
        verdict = "DEGRADED"
        rc = 1
        why = "retrieval broke at depths " + ", ".join(str(d) for d in failed)
    else:
        verdict = "UNRELIABLE"
        rc = 1
        why = "more than half of NIAH cases missed the passkey"

    tone = {"RELIABLE": "1;32", "RELIABLE-RETRIEVAL": "1;32", "DEGRADED": "1;33"}.get(
        verdict, "1;31"
    )
    print(f"  verdict  {_color(color, tone, verdict)}  — {why}")
    if not pool.nvfp4:
        print("           (pool is not NVFP4; treat this as a bf16 baseline)")

    report = {
        "started": _utc(),
        "endpoint": args.base_url,
        "suite": args.suite,
        "pool": asdict(pool),
        "verdict": verdict,
        "why": why,
        "cases": [asdict(c) for c in results],
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"  json     {args.json}")
    return rc


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Needle-in-haystack reliability eval for NVFP4 KV cache"
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8888")
    p.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    p.add_argument("--suite", choices=sorted(SUITES), default="quick")
    p.add_argument("--depths", help="comma-separated token depths, overrides --suite")
    p.add_argument("--positions", help="comma-separated fractions 0-1 (default 0,0.5,1)")
    p.add_argument("--timeout", type=float, default=1800.0, help="per-request seconds")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--thinking", action="store_true", help="leave Qwen thinking on")
    p.add_argument(
        "--require-nvfp4",
        action="store_true",
        help="exit 2 if the live pool does not look like NVFP4 KV",
    )
    p.add_argument("--json", help="write a JSON report to this path")
    p.add_argument("--no-color", action="store_true")
    return p.parse_args(argv)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
