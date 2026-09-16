#!/usr/bin/env bash
# Build the NVFP4-experts + FP8-dense hybrid checkpoint into the head HF cache.
# CPU only, streaming, ~1 GB RAM, ~2 minutes; safe to run next to a live server.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
SRC_REPO="${SRC_REPO:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
# MTP_FP8=true: also quantize the MTP draft layer (dense/HC per-channel FP8 + routed
# experts as 128x128 block FP8) into a SEPARATE repo dir / snapshot, so the plain
# FP8-dense checkpoint keeps serving untouched. Unchanged bf16 shards are hard-linked
# from it. Select with FP8_DENSE_MODEL_ID=MiaAI-Lab/Qwen3.8-Flash-Next-NVFP4-FP8dense-MTP.
MTP_FP8="${MTP_FP8:-false}"
BASE_DST_REPO="MiaAI-Lab/Qwen3.8-Flash-Next-NVFP4-FP8dense"
if [[ "$MTP_FP8" == "true" ]]; then
    DST_REPO="${DST_REPO:-${BASE_DST_REPO}-MTP}"
else
    DST_REPO="${DST_REPO:-$BASE_DST_REPO}"
fi
src_dir="$HF_CACHE_DIR/hub/models--${SRC_REPO%%/*}--${SRC_REPO##*/}"
rev=$(cat "$src_dir/refs/main")
dst_dir="$HF_CACHE_DIR/hub/models--${DST_REPO%%/*}--${DST_REPO##*/}"
MTP_ARGS=()
if [[ "$MTP_FP8" == "true" ]]; then
    dst_rev="fp8dense-mtp-${rev:0:8}"
    MTP_ARGS=(--mtp-dense --mtp-experts)
    base_snap="/hf/hub/models--${BASE_DST_REPO%%/*}--${BASE_DST_REPO##*/}/snapshots/fp8dense-${rev:0:8}"
    if [[ -d "$HF_CACHE_DIR/hub/models--${BASE_DST_REPO%%/*}--${BASE_DST_REPO##*/}/snapshots/fp8dense-${rev:0:8}" ]]; then
        MTP_ARGS+=(--link-unchanged-from "$base_snap")
    fi
else
    dst_rev="fp8dense-${rev:0:8}"
fi
mkdir -p "$dst_dir/refs" "$dst_dir/snapshots/$dst_rev"
echo -n "$dst_rev" > "$dst_dir/refs/main"
run() {
    docker run --rm --memory=1200m --memory-swap=1200m --cpus=4 --entrypoint python3 \
        -e CUDA_VISIBLE_DEVICES= -v "$HF_CACHE_DIR:/hf" -v "$SCRIPT_DIR:/work:ro" "$IMAGE" "$@"
}
run /work/make_fp8_dense_checkpoint.py --src "/hf/hub/models--${SRC_REPO%%/*}--${SRC_REPO##*/}/snapshots/$rev" \
    --dst "/hf/hub/models--${DST_REPO%%/*}--${DST_REPO##*/}/snapshots/$dst_rev" --resume "${MTP_ARGS[@]}" "$@"
docker run --rm --entrypoint chown -v "$HF_CACHE_DIR:/hf" "$IMAGE" -R "$(id -u):$(id -g)" \
    "/hf/hub/models--${DST_REPO%%/*}--${DST_REPO##*/}"
run /work/verify_fp8_dense_checkpoint.py --src "/hf/hub/models--${SRC_REPO%%/*}--${SRC_REPO##*/}/snapshots/$rev" \
    --dst "/hf/hub/models--${DST_REPO%%/*}--${DST_REPO##*/}/snapshots/$dst_rev"
if [[ "$MTP_FP8" == "true" ]]; then
    echo "Hybrid checkpoint ready: $DST_REPO  (set FP8_DENSE=true and FP8_DENSE_MODEL_ID=$DST_REPO in .env)"
else
    echo "Hybrid checkpoint ready: $DST_REPO  (set FP8_DENSE=true in .env)"
fi
