#!/usr/bin/env bash
# Build the single-Spark checkpoint: FP8-dense + quantized MTP (from files/fp8dense)
# with the PLE n-gram table requantized FP8 -> NVFP4, into a NEW HF repo dir.
# CPU only, streaming, <2 GiB RAM, ~30-60 min (47.7 GiB of FP8 -> 26.8 GiB NVFP4);
# safe to run next to a live server. Everything but the PLE shards is hard-linked
# from SRC_REPO, so the on-disk delta is ~27 GiB.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
SRC_REPO="${SRC_REPO:-MiaAI-Lab/Qwen3.8-Flash-Next-NVFP4-FP8dense-MTP}"
DST_REPO="${DST_REPO:-${SRC_REPO}-PLE4}"
CPUS="${CPUS:-6}"
MEM="${MEM:-4g}"
src_dir="$HF_CACHE_DIR/hub/models--${SRC_REPO%%/*}--${SRC_REPO##*/}"
[[ -f "$src_dir/refs/main" ]] || { echo "source repo not in cache: $src_dir (run files/fp8dense/build.sh with MTP_FP8=true first)"; exit 1; }
rev=$(cat "$src_dir/refs/main")
dst_dir="$HF_CACHE_DIR/hub/models--${DST_REPO%%/*}--${DST_REPO##*/}"
dst_rev="${rev/fp8dense-mtp-/fp8dense-mtp-ple4-}"
[[ "$dst_rev" != "$rev" ]] || dst_rev="ple4-${rev}"
mkdir -p "$dst_dir/refs" "$dst_dir/snapshots/$dst_rev"
echo -n "$dst_rev" > "$dst_dir/refs/main"
free_gib=$(df -BG --output=avail "$HF_CACHE_DIR" | tail -1 | tr -dc '0-9')
[[ "$free_gib" -ge 40 ]] || { echo "need >= 40 GiB free under $HF_CACHE_DIR, have ${free_gib}G"; exit 1; }
run() {
    docker run --rm --memory="$MEM" --memory-swap="$MEM" --cpus="$CPUS" --entrypoint python3 \
        -e CUDA_VISIBLE_DEVICES= -v "$HF_CACHE_DIR:/hf" -v "$FILES_DIR:/work:ro" "$IMAGE" "$@"
}
SRC_SNAP="/hf/hub/models--${SRC_REPO%%/*}--${SRC_REPO##*/}/snapshots/$rev"
DST_SNAP="/hf/hub/models--${DST_REPO%%/*}--${DST_REPO##*/}/snapshots/$dst_rev"
run -u /work/ple_nvfp4/make_ple_nvfp4_checkpoint.py --src "$SRC_SNAP" --dst "$DST_SNAP" --resume --threads "$CPUS" "$@"
docker run --rm --entrypoint chown -v "$HF_CACHE_DIR:/hf" "$IMAGE" -R "$(id -u):$(id -g)" \
    "/hf/hub/models--${DST_REPO%%/*}--${DST_REPO##*/}"
run -u /work/ple_nvfp4/verify_ple_nvfp4_checkpoint.py --src "$SRC_SNAP" --dst "$DST_SNAP"
echo "Single-Spark checkpoint ready: $DST_REPO  ($dst_dir/snapshots/$dst_rev)"
echo "Serve with: ./start-tp1.sh --profile tp1/resident.env   (TP1_MODEL_ID=$DST_REPO)"
