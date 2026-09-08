# Autoresearch harness

Automated config sweep for the dual-Spark Qwen3.8-Flash-Next deployment.
Each trial = apply `.env` overrides -> relaunch -> wait for health -> benchmark
-> record -> restore baseline.

## Why it exists

Hand-driving this is ~12 minutes per configuration, and two failure modes
already bit us:

1. **A hung config takes the box down.** `torch.compile` mode 3 compiled for
   103 minutes without finishing and killed both nodes. Every launch here has a
   hard deadline (`LAUNCH_TIMEOUT`); a trial that misses it is killed, recorded
   `FAILED` with its log tail, and the baseline is restored.
2. **Silent measurement contamination.** Any co-resident client polling
   localhost:8888 that does not set `temperature` inherits the checkpoint's
   `generation_config.json` default of `temperature=1.0, top_k=20, top_p=0.95`.
   Mixed into a benchmark this drags acceptance down and the numbers look like a
   regression that isn't there.
   Every run checks the `generation_tokens` delta against the expected
   4 tasks x 400 tokens and marks the run `DIRTY` if other traffic intruded.

A third trap the harness guards against: **a flag that silently does nothing.**
`--all2all-backend` is inert at `dp_size=1`, so it looked like a tuning knob
and was not one. Each record captures `backends` from the startup log, so a
"win" can be traced to a setting that actually took effect.

## Use

    ./autoresearch/sweep.py --list
    ./autoresearch/sweep.py --only moe_flashinfer_b12x
    ./autoresearch/sweep.py --all --repeat 2
    ./autoresearch/sweep.py --restore    # put baseline back and serve

Results append to `autoresearch/results.jsonl`. The baseline `.env` is
snapshotted once to `autoresearch/baseline.env` and restored after every trial,
so the server never stays down in a broken config.

Add trials by editing `trials.yaml`. Keep `why:` honest -- it is the record of
what you expected, which is what makes a negative result worth reading later.

## Settled results

See `RESULTS.md`.
