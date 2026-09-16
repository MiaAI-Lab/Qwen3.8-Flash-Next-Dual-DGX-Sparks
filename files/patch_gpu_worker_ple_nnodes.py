#!/usr/bin/env python3
"""Make PLE CPU-offload work with multi-node TP (one worker per node).

Why this exists
---------------
vLLM's stock guard `gpu_worker.py::_validate_ple_offload_config` rejects any
PLE CPU-offload config with `parallel_config.nnodes != 1`:

    ValueError: VLLM_PLE_CPU_OFFLOAD does not support the requested
    configuration. Unsupported settings: nnodes=2

The offload path is node-local by construction: every registration carries
CUDA IPC handles (output buffers) and file_system shared-memory tensor views
(inputs, done flag), and coordination is a per-node ipc address. The correct
topology for nnodes>1 is therefore ONE offload worker per node serving that
node's local ranks -- not a cross-node service. The two-node FP8 lane needs
exactly that: ~125.1 GiB of non-PLE FP8 weight does not fit one Spark
(~121.7 GiB unified pool), so TP2 is the only shape where full FP8 fits.

The stock single-node assumption shows up in three places, all fixed here:

  1. the `nnodes != 1` rejection -- removed;
  2. `spawn_ple_offload` only spawns on global rank 0, so a second node never
     gets a worker and its registration is sent to an ipc path with no
     listener (zmq queues it silently, the message is never delivered) --
     each node's first rank now spawns
     (`rank == node_rank_within_dp * local_world_size`);
  3. `num_workers = dp_size * tp_size` counts the whole world, but a
     node-local worker only ever receives its own node's registrations --
     now `local_world_size`.

The matching worker/connector/registration changes (per-registration rank,
local rank-set validation, per-DP-group leader for inputs and requests) live
in `files/patch_ple_offload.py`.

Inputs:  files/gpu_worker/gpu_worker.py.orig (extracted from the image)
Outputs: files/gpu_worker/gpu_worker.py      (bind-mounted over the package)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "gpu_worker", "gpu_worker.py.orig")
OUT = os.path.join(HERE, "gpu_worker", "gpu_worker.py")

EDITS = [
    # 1. Accept nnodes>1 for the node-local design.
    (
        "        if parallel_config.nnodes != 1:\n"
        "            unsupported.append(f\"nnodes={parallel_config.nnodes}\")\n",
        "        # nnodes > 1 is supported: one node-local offload worker per\n"
        "        # node over local CUDA IPC, shared memory and a per-node zmq\n"
        "        # ipc path. Replicas that span every node (DP>1) are NOT:\n"
        "        # a node-local worker only serves one dp0 replica per node.\n"
        "        if parallel_config.nnodes > 1 and parallel_config.data_parallel_size > 1:\n"
        "            unsupported.append(\n"
        "                f\"nnodes={parallel_config.nnodes} with \"\n"
        "                f\"DP={parallel_config.data_parallel_size}\"\n"
        "            )\n",
    ),
    # 2. Spawn from each node's first DP0 rank, not from global rank 0.
    (
        '        """Spawn one node-local PLE CPU worker from DP0/TP0."""\n',
        "        \"\"\"Spawn one node-local PLE CPU worker from each node's first\n"
        "        DP0 rank.\"\"\"\n",
    ),
    (
        "        if (\n"
        "            not self._ple_offload_enabled\n"
        "            or self.rank != 0\n"
        "            or self.parallel_config.data_parallel_rank != 0\n"
        "        ):\n"
        "            return\n",
        "        parallel_config = self.parallel_config\n"
        "        node_start = (\n"
        "            parallel_config.node_rank_within_dp\n"
        "            * parallel_config.local_world_size\n"
        "        )\n"
        "        if (\n"
        "            not self._ple_offload_enabled\n"
        "            or self.rank != node_start\n"
        "            or parallel_config.data_parallel_rank != 0\n"
        "        ):\n"
        "            return\n",
    ),
    # 3. A node's worker only ever receives that node's registrations.
    (
        "        dp_size = self.parallel_config.data_parallel_size\n"
        "        tp_size = self.parallel_config.tensor_parallel_size\n"
        "        num_workers = dp_size * tp_size\n",
        "        dp_size = self.parallel_config.data_parallel_size\n"
        "        tp_size = self.parallel_config.tensor_parallel_size\n"
        "        # One worker per node: it only receives its own node's ranks.\n"
        "        num_workers = self.parallel_config.local_world_size\n",
    ),
]

src = open(ORIG).read()
for old, new in EDITS:
    count = src.count(old)
    if count != 1:
        raise SystemExit(
            f"gpu_worker.py: anchor not unique/missing (count={count}):\n{old[:300]}"
        )
    src = src.replace(old, new, 1)
open(OUT, "w").write(src)
print("patched gpu_worker.py")
