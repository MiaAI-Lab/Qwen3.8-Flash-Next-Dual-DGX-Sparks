#!/usr/bin/env bash
# test_check_weights.sh — Shell-level tests for check-weights.sh --verify.
#
# These cover integration points that unit tests cannot reach: flag parsing,
# the manifest fetch step, the head verification path, the worker path (via a
# stub ssh/scp), and the blocking behavior on a corrupt shard. They use a tiny
# fake model directory plus a stub HF API server so no real model or network
# is needed.
#
# Run from the repo root:
#   bash tests/test_check_weights.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORK="$(mktemp -d)"
export WORK
trap '[[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true; rm -rf "$WORK" .env' EXIT

# --- fixture: tiny model in the standard hub layout -------------------------
HF_HOME="$WORK/hf"
MODEL_ROOT="$HF_HOME/hub/models--org--model"
SNAPSHOT="$MODEL_ROOT/snapshots/rev1"
mkdir -p "$SNAPSHOT" "$MODEL_ROOT/refs"
printf 'all the weights' > "$SNAPSHOT/model.safetensors"
printf 'rev1' > "$MODEL_ROOT/refs/main"

# Build a manifest describing exactly that fixture, using the verifier itself.
python3 - "$REPO_ROOT/verify-weights.py" "$SNAPSHOT" "$WORK/manifest.json" <<'PY'
import importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location("vw", sys.argv[1])
vw = importlib.util.module_from_spec(spec); spec.loader.exec_module(vw)
root = sys.argv[2]
files, _ = vw.iter_model_files(root)
manifest = {}
for rel, path in files.items():
    manifest[rel] = {"type": "file", "size": os.path.getsize(path),
                     "lfs": {"oid": "sha256:" + vw.file_sha256(path)}}
json.dump(manifest, open(sys.argv[3], "w"))
print(f"manifest written: {len(manifest)} files")
PY

# --- stub HF API server -----------------------------------------------------
cat > "$WORK/mock_api.py" <<'PY'
import http.server, json, os, sys
root, manifest_path = sys.argv[1], sys.argv[2]
manifest = json.load(open(manifest_path))
os.makedirs(root, exist_ok=True)

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps([{"type": "file", "path": k, "size": v["size"],
                            "lfs": v["lfs"]} for k, v in manifest.items()]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass

srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
with open(os.path.join(root, "port"), "w") as f:
    f.write(str(srv.server_address[1]))
srv.serve_forever()
PY
python3 "$WORK/mock_api.py" "$WORK" "$WORK/manifest.json" &
SERVER_PID=$!
for _ in $(seq 1 50); do
    [[ -f "$WORK/port" ]] && break
    sleep 0.1
done
[[ -f "$WORK/port" ]] || { echo "mock API server did not start"; exit 1; }
PORT="$(cat "$WORK/port")"
export HF_API_BASE="http://127.0.0.1:$PORT"

# --- stub ssh/scp: the worker's side lives on this same machine -------------
mkdir -p "$WORK/bin"
cat > "$WORK/bin/ssh" <<SSH
#!/usr/bin/env bash
# Stub ssh for check-weights.sh. The worker model path equals the head's, and
# /tmp/verify-weights.py + /tmp/verify-manifest.json are pre-copied by the test.
# Everything before the ssh host argument is ignored; the rest runs locally.
args=("\$@")
hostpos=-1
for i in "\${!args[@]}"; do
    case "\${args[\$i]}" in
        *@*|*.*.*.*) hostpos=\$i ;;
    esac
done
if [[ "\$*" == *'echo "\$HOME"'* || "\$*" == *'echo \$HOME'* ]]; then
    echo "\$HOME"
    exit 0
fi
if [[ "\$*" == *'test -d'* ]]; then
    exit 0
fi
if [[ "\$*" == *'du -sh'* ]]; then
    du -sh "$SNAPSHOT" 2>/dev/null | cut -f1
    exit 0
fi
if [[ "\$*" == *'find'* ]]; then
    find "$SNAPSHOT" -name '*.safetensors' -o -name '*.bin' 2>/dev/null | wc -l
    exit 0
fi
if (( hostpos >= 0 )); then
    exec bash -c "\${args[*]:hostpos+1}"
fi
exit 0
SSH
cat > "$WORK/bin/scp" <<SCP
#!/usr/bin/env bash
# Stub scp: the worker files are pre-copied by the test harness.
exit 0
SCP
chmod +x "$WORK/bin/ssh" "$WORK/bin/scp"

# Pre-copy to the "worker" (this machine) so the stubbed remote command works.
cp "$REPO_ROOT/verify-weights.py" /tmp/verify-weights.py
cp "$WORK/manifest.json" /tmp/verify-manifest.json

# --- .env for check-weights.sh ----------------------------------------------
cat > .env <<ENV
HEAD_IP=10.0.0.1
WORKER_IP=10.0.0.2
MODEL_ID=org/model
HF_HOME=$HF_HOME
ENV

export SNAPSHOT

PASS=0
run() {
    local name="$1"
    shift
    if timeout 15 "$@"; then
        PASS=$((PASS + 1))
        echo "ok - $name"
    else
        echo "FAIL/Timeout - $name (exit $?)"
        exit 1
    fi
}

echo "== check-weights.sh shell tests =="

# 1. No-argument fast check: presence + size, no manifest, no python verify.
run "no-arg fast check" env PATH="$WORK/bin:$PATH" ./check-weights.sh

# 2. --verify end to end: fetch via mock API, verify head, verify "worker".
run "--verify end to end (head + worker)" \
    env PATH="$WORK/bin:$PATH" ./check-weights.sh --verify >/dev/null

# 3. A corrupt shard makes --verify exit nonzero (blocking, not warning).
cp "$SNAPSHOT/model.safetensors" "$WORK/model.safetensors.bak"
printf 'corrupt!!!' > "$SNAPSHOT/model.safetensors"
run "corrupt shard is blocking (non-zero exit)" bash -c "
    set +e
    PATH=$WORK/bin:\$PATH ./check-weights.sh --verify >/dev/null 2>&1
    rc=\$?
    set -e
    [[ \$rc -ne 0 ]]
"
mv "$WORK/model.safetensors.bak" "$SNAPSHOT/model.safetensors"

# 3b. --dry-run succeeds, fetches the manifest, checks presence/size, no hash.
run "--dry-run succeeds (presence + size only)" \
    env PATH="$WORK/bin:$PATH" ./check-weights.sh --dry-run >/dev/null 2>&1

# 3c. --dry-run still flags a missing file (presence check, not just a plan).
mv "$SNAPSHOT/model.safetensors" "$WORK/model.safetensors.hidden"
run "--dry-run flags missing file (non-zero exit)" bash -c "
    set +e
    PATH=$WORK/bin:\$PATH ./check-weights.sh --dry-run >/dev/null 2>&1
    rc=\$?
    set -e
    [[ \$rc -ne 0 ]]
"
mv "$WORK/model.safetensors.hidden" "$SNAPSHOT/model.safetensors"

# 4. Unknown flag is rejected with exit 2.
run "unknown flag rejected (exit 2)" bash -c "
    set +e
    PATH=$WORK/bin:\$PATH ./check-weights.sh --bogus >/dev/null 2>&1
    rc=\$?
    set -e
    [[ \$rc -eq 2 ]]
"

# 5. Missing .env is still a hard error.
run "missing .env fails (exit 1)" bash -c "
    set +e
    mv .env \$WORK/.env.saved
    ./check-weights.sh >/dev/null 2>&1
    rc=\$?
    mv \$WORK/.env.saved .env
    set -e
    [[ \$rc -eq 1 ]]
"

# 6. Manifest fetch failure is blocking: an unreachable API must fail, not pass.
run "API fetch failure is blocking (non-zero exit)" bash -c "
    set +e
    PATH=$WORK/bin:\$PATH HF_API_BASE=http://127.0.0.1:1 ./check-weights.sh --verify >/dev/null 2>&1
    rc=\$?
    set -e
    [[ \$rc -ne 0 ]]
"

echo ""
echo "All $PASS shell tests passed."
