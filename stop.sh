#!/usr/bin/env bash
# Stop the Qwen3.8-Flash-Next server (both nodes).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/start.sh" stop "$@"
