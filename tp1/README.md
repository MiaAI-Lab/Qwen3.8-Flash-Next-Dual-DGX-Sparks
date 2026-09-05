# Qwen3.8-Flash-Next on ONE DGX Spark (TP=1)

Self-contained recipe. Nothing here depends on the 2-node files in the parent
directory.

```
cp .env.sample .env        # edit IMAGE / HF_TOKEN if needed
./start.sh                 # ~10-12 min to /health; serves on :8888
./stop.sh
```

Known-good profile (in `.env.sample`): 262,144 native context, MTP 3,
KV target 20 GiB (~686k tokens), host MemAvailable ~13 GiB under load,
decode ~25 tok/s prose / ~30 tok/s code, ~1,750 tok/s prefill at 200k.

Read `docs/HANDOFF-single-spark.md` §0 for how it fits (memory-mapped PLE
table, host-side offload handshake) and the safety rules. Key ones:

- Do not push host MemAvailable below ~10 GiB: exhausting the unified pool
  hangs the kernel, and the page cache is what keeps PLE lookups off NVMe.
- FP8 KV cache is unsupported by this model's attention backend; KV is bf16.
- `comfy-h3.service` must stay disabled (it launches a GPU co-tenant when
  port 8888 answers). `start.sh` refuses 8888 while it is active.

Layout:

- `start.sh` / `stop.sh` — launcher (derives the GPU budget from live memory,
  builds the packed PLE table on first run, starts `files/memwatch.sh`).
- `files/patch_ple_layer.py`, `files/patch_modelopt_mxfp8.py`,
  `files/patch_ple_offload.py` — regenerate the patched vLLM files from the
  `*.orig` copies on every launch; only edit the generators.
- `files/build_ple_packed_table.py` — one-time packed table builder
  (output: `~/.cache/vllm/ple_cache/`).
- `bench/` — long-context and decode benchmarks (`--port 8888`).
