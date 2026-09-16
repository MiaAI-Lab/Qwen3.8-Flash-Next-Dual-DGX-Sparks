#!/usr/bin/env bash
# Build the NVFP4-experts + FP8-dense hybrid checkpoint into the head HF cache.
# CPU only, streaming, ~1 GB RAM, ~2 minutes; safe to run next to a live server.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
SRC_REPO="${SRC_REPO:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
DST_REPO="${DST_REPO:-MiaAI-Lab/Qwen3.8-Flash-Next-NVFP4-FP8dense}"
src_dir="$HF_CACHE_DIR/hub/models--${SRC_REPO%%/*}--${SRC_REPO##*/}"
# SRC_REPO defaults to the RadixArk build this converter was written and
# measured against, which is NOT the nvidia/... checkpoint start.sh serves by
# default -- so on most installs it is absent, or present only as a refs/main
# stub left by a cancelled download. Reading refs/main directly then dies deep
# inside the container ("no model.safetensors.index.json") with nothing naming
# the real cause. Resolve it the way start.sh does instead: same completeness
# rule, same preference for refs/main.
rev=""; rc=0
rev="$(python3 "$SCRIPT_DIR/../resolve_snapshot.py" "$src_dir" 2>/dev/null)" || rc=$?
if [[ "$rc" -ne 0 || -z "$rev" ]]; then
    echo "ERROR: no complete snapshot of $SRC_REPO in the HF cache." >&2
    echo "  Looked in: $src_dir" >&2
    case "$rc" in
        1) echo "  A snapshot exists but is missing indexed weight shards (partial download)." >&2 ;;
        *) echo "  No snapshot directory at all." >&2 ;;
    esac
    echo "  Fetch or resume it:  hf download $SRC_REPO" >&2
    echo "  Cache root:          HF_HOME=$HF_CACHE_DIR" >&2
    echo "  Or point SRC_REPO at another NVFP4 build already in the cache." >&2
    exit 1
fi
dst_dir="$HF_CACHE_DIR/hub/models--${DST_REPO%%/*}--${DST_REPO##*/}"
dst_rev="fp8dense-${rev:0:8}"
mkdir -p "$dst_dir/refs" "$dst_dir/snapshots/$dst_rev"
echo -n "$dst_rev" > "$dst_dir/refs/main"
run() {
    docker run --rm --memory=1200m --memory-swap=1200m --cpus=4 --entrypoint python3 \
        -e CUDA_VISIBLE_DEVICES= -v "$HF_CACHE_DIR:/hf" -v "$SCRIPT_DIR:/work:ro" "$IMAGE" "$@"
}
run /work/make_fp8_dense_checkpoint.py --src "/hf/hub/models--${SRC_REPO%%/*}--${SRC_REPO##*/}/snapshots/$rev" \
    --dst "/hf/hub/models--${DST_REPO%%/*}--${DST_REPO##*/}/snapshots/$dst_rev" --resume "$@"
docker run --rm --entrypoint chown -v "$HF_CACHE_DIR:/hf" "$IMAGE" -R "$(id -u):$(id -g)" \
    "/hf/hub/models--${DST_REPO%%/*}--${DST_REPO##*/}"
run /work/verify_fp8_dense_checkpoint.py --src "/hf/hub/models--${SRC_REPO%%/*}--${SRC_REPO##*/}/snapshots/$rev" \
    --dst "/hf/hub/models--${DST_REPO%%/*}--${DST_REPO##*/}/snapshots/$dst_rev"
echo "Hybrid checkpoint ready: $DST_REPO  (set FP8_DENSE=true in .env)"
