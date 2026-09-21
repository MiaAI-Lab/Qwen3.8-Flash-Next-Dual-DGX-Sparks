#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Sourced by start.sh. No work is done merely by sourcing this file.

mtp_adaptive_validate() {
    local knob
    for knob in MTP_ADAPTIVE MTP_INDEX_SHARE; do
        [[ "${!knob}" == true || "${!knob}" == false ]] || err "$knob must be true or false"
    done
    if [[ "$MTP_INDEX_SHARE" == true && "$MTP_NUM_SPECULATIVE_TOKENS" == 0 ]]; then
        err "MTP_INDEX_SHARE requires MTP_NUM_SPECULATIVE_TOKENS > 0"
    fi
    if [[ "$MTP_ADAPTIVE" == true ]]; then
        [[ "$TENSOR_PARALLEL_SIZE" == 2 ]] || err "MTP_ADAPTIVE is currently supported on TP2 only"
        [[ "$MTP_NUM_SPECULATIVE_TOKENS" == 4 ]] || err "MTP_ADAPTIVE requires MTP_NUM_SPECULATIVE_TOKENS=4 (maximum, not fixed depth)"
        [[ "$MTP_INDEX_SHARE" == true ]] || err "MTP_ADAPTIVE requires MTP_INDEX_SHARE=true (tested configuration)"
    fi
}

mtp_adaptive_prepare() {
    [[ "$MTP_ADAPTIVE" == true ]] || return 0
    local image_id worker_id cache sources output target
    image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE")
    [[ "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]] || err "Cannot identify adaptive MTP image"
    worker_id=$(ssh_worker "docker image inspect --format '{{.Id}}' '$IMAGE'")
    [[ "$image_id" == "$worker_id" ]] || err "MTP_ADAPTIVE requires the same image digest on both nodes"
    # Image-keyed originals: a mutable tag must never reuse a previous image's sources.
    cache="$SCRIPT_DIR/files/mtp_adaptive/generated/${image_id#sha256:}"
    sources=$(python3 "$SCRIPT_DIR/files/patch_mtp_adaptive.py" --list-sources) || err "Cannot enumerate adaptive MTP sources"
    while IFS='|' read -r output target; do
        mkdir -p "$cache/original/$(dirname "$target")"
        extract_from_image "$VLLM_PKG/$target" "$cache/original/$target"
    done <<< "$sources"
    python3 "$SCRIPT_DIR/files/patch_mtp_adaptive.py" \
        --source-root "$cache/original" --output-dir "$cache/overlay" >/dev/null || err "Adaptive MTP source/patch checks failed"
    while IFS='|' read -r output target; do
        add_overlay "$cache/overlay/$output" "$VLLM_PKG/$target"
    done <<< "$sources"
    add_overlay "$cache/overlay/qwen_mtp_adaptive.py" "$VLLM_PKG/v1/core/sched/qwen_mtp_adaptive.py"
    OVERLAY_ENV+=("-e QWEN_MTP_ADAPTIVE=1")
    ok "Adaptive MTP: K=1/2/3/4, maximum=4, IndexShare on; target verification unchanged"
}
