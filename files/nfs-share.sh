# files/nfs-share.sh — share the head HuggingFace cache over NFS (ConnectX).
# Sourced by start.sh and check-weights.sh. Requires: ssh_worker, info/ok/warn/err,
# SCRIPT_DIR, IFACE, WORKER_IP, HF_CACHE_DIR.
#
# Worker does not keep a local copy of the checkpoint. Docker on the worker
# mounts this share as /root/.cache/huggingface (dockerd is root, so no sudo).

NFS_IMAGE="${NFS_IMAGE:-vllm-fn-nfs:local}"
NFS_CONTAINER="${NFS_CONTAINER:-vllm-fn-nfs}"
NFS_VOLUME="${NFS_VOLUME:-vllm-fn-hf}"
NFS_DOCKERFILE_DIR="${SCRIPT_DIR}/files/nfs-server"

nfs_detect_server_ip() {
    if [[ -n "${NFS_SERVER_IP:-}" ]]; then
        return 0
    fi
    NFS_SERVER_IP=$(ip -4 -o addr show dev "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
    [[ -n "$NFS_SERVER_IP" ]] || err "Could not detect NFS_SERVER_IP from IFACE=$IFACE. Set NFS_SERVER_IP in .env (head ConnectX address, e.g. 10.0.22.1)."
}

nfs_clients() {
    local cidr net mask addr
    cidr=$(ip -4 -o addr show dev "$IFACE" 2>/dev/null | awk '{print $4}' | head -1)
    # 10.0.22.1/24 → 10.0.22.0/24 (Spark CX links are /24)
    if [[ "$cidr" == */* ]]; then
        addr="${cidr%/*}"
        mask="${cidr#*/}"
        net="${addr%.*}.0/${mask}"
        echo "${WORKER_IP},${net}"
    else
        echo "${WORKER_IP}"
    fi
}

nfs_ensure_server() {
    nfs_detect_server_ip
    local clients
    clients="$(nfs_clients)"

    [[ -d "$NFS_DOCKERFILE_DIR" ]] || err "Missing $NFS_DOCKERFILE_DIR"
    [[ -d "$HF_CACHE_DIR" ]] || err "HF cache not found at $HF_CACHE_DIR"

    # A live NFSv4 server is enough — do not docker rm a working share.
    # Kernel nfsd in a privileged container can leave rpcbind in D-state, which
    # makes `docker rm -f` hang. Recreate only when nothing is listening.
    if timeout 3 rpcinfo -t "$NFS_SERVER_IP" nfs 4 >/dev/null 2>&1; then
        ok "NFS share already up on ${NFS_SERVER_IP}:2049 → $HF_CACHE_DIR"
        return 0
    fi

    info "Building NFS image $NFS_IMAGE ..."
    docker build -q -t "$NFS_IMAGE" "$NFS_DOCKERFILE_DIR" >/dev/null

    docker rm -f "$NFS_CONTAINER" >/dev/null 2>&1 || true

    info "Exporting $HF_CACHE_DIR via NFS on $NFS_SERVER_IP (clients: $clients)"
    docker run -d --name "$NFS_CONTAINER" --restart unless-stopped \
        --privileged --network host \
        -v "$HF_CACHE_DIR:/export:ro" \
        -e "NFS_CLIENTS=$clients" \
        "$NFS_IMAGE" >/dev/null

    local i
    for i in 1 2 3 4 5 6 7 8 9 10; do
        if timeout 3 rpcinfo -t "$NFS_SERVER_IP" nfs 4 >/dev/null 2>&1; then
            ok "NFS server ready on ${NFS_SERVER_IP}:2049 (NFSv4)"
            return 0
        fi
        sleep 0.5
    done
    docker logs "$NFS_CONTAINER" >&2 || true
    err "NFS server did not become ready on $NFS_SERVER_IP. Check: docker logs $NFS_CONTAINER"
}

nfs_ensure_worker_volume() {
    local recreate="${1:-}"
    nfs_detect_server_ip
    if [[ "$recreate" != "recreate" ]] && ssh_worker "docker volume inspect '$NFS_VOLUME' >/dev/null 2>&1"; then
        ok "Worker volume $NFS_VOLUME already exists"
        return 0
    fi
    info "Creating worker NFS volume $NFS_VOLUME → ${NFS_SERVER_IP}:/"
    ssh_worker "docker volume rm '$NFS_VOLUME' >/dev/null 2>&1 || true"
    ssh_worker "docker volume create --driver local \
        --opt type=nfs \
        --opt o=addr=${NFS_SERVER_IP},nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600 \
        --opt device=:/ \
        '$NFS_VOLUME' >/dev/null"
    ok "Worker volume $NFS_VOLUME → nfs://${NFS_SERVER_IP}/"
}

nfs_worker_has_model() {
    local rel="$1"
    ssh_worker "docker run --rm -v '${NFS_VOLUME}:/hf:ro' alpine:latest test -d '/hf/${rel}'" >/dev/null 2>&1
}
