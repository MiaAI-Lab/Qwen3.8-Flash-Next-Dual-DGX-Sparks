#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../files/mtp_adaptive/launch.sh
source "$ROOT/files/mtp_adaptive/launch.sh"
err() { printf '%s\n' "$*" >&2; exit 1; }
MTP_ADAPTIVE=false MTP_INDEX_SHARE=false MTP_NUM_SPECULATIVE_TOKENS=3 TENSOR_PARALLEL_SIZE=2
mtp_adaptive_validate
# Prove the default-off path does not access Docker, SSH, or the filesystem.
docker() { err 'disabled path called docker'; }
ssh_worker() { err 'disabled path called ssh'; }
extract_from_image() { err 'disabled path extracted sources'; }
add_overlay() { err 'disabled path added overlay'; }
mtp_adaptive_prepare

if (MTP_ADAPTIVE=invalid; mtp_adaptive_validate) 2>/dev/null; then exit 1; fi
if (MTP_ADAPTIVE=true; mtp_adaptive_validate) 2>/dev/null; then exit 1; fi
if (MTP_INDEX_SHARE=true; MTP_NUM_SPECULATIVE_TOKENS=0; mtp_adaptive_validate) 2>/dev/null; then exit 1; fi
MTP_ADAPTIVE=true MTP_INDEX_SHARE=true MTP_NUM_SPECULATIVE_TOKENS=4
mtp_adaptive_validate
if (TENSOR_PARALLEL_SIZE=1; mtp_adaptive_validate) 2>/dev/null; then exit 1; fi
printf '%s\n' 'adaptive MTP launcher: defaults inert; invalid combinations refused'
