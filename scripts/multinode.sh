#!/usr/bin/env bash
# Pipeline parallelism across two machines, one GPU each: Qwen3-8B split into two stages, stage 0 on
# the first machine (which also serves HTTP and runs the benchmark client), stage 1 on the second.
#
#   machine 0:  NODE_RANK=0 MASTER_ADDR=<machine 0's private IP> scripts/multinode.sh
#   machine 1:  NODE_RANK=1 MASTER_ADDR=<machine 0's private IP> scripts/multinode.sh
#
# Both machines need this repository and its .venv (scripts/gpu_suite.sh setup), and a private network
# between them: the same cloud VPC, or Tailscale. Never a public address: torchrun's rendezvous and the
# gloo and NCCL connections have no authentication, so this script refuses one. Machine 0 writes
# bench/results/distributed_8b/pp2-2nodes_*, next to the single-machine layouts it is compared with.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${NODE_RANK:?set NODE_RANK to 0 on the first machine and 1 on the second}"
: "${MASTER_ADDR:?set MASTER_ADDR to the private IP of machine 0}"
PY=.venv/bin/python
MODEL="${MODEL:-Qwen/Qwen3-8B-Base}"
MASTER_PORT="${MASTER_PORT:-29500}"
RESULTS="${RESULTS:-bench/results}"

$PY - "$MASTER_ADDR" <<'EOF'
import ipaddress, sys
ip = ipaddress.ip_address(sys.argv[1])
tailnet = ipaddress.ip_network("100.64.0.0/10")  # shared address space, which Tailscale assigns from
if not (ip.is_private or ip in tailnet):
    sys.exit(f"refusing MASTER_ADDR={ip}: not a private address. Use the machines' private network (VPC or Tailscale).")
EOF

# Keep gloo and NCCL on the interface that reaches MASTER_ADDR, which is the private network.
IFACE="${IFACE:-$(ip -o route get "$MASTER_ADDR" | sed -n 's/.* dev \([^ ]*\).*/\1/p')}"
export GLOO_SOCKET_IFNAME="$IFACE" NCCL_SOCKET_IFNAME="$IFACE"
echo "machine $NODE_RANK: rendezvous at $MASTER_ADDR:$MASTER_PORT over $IFACE"

$PY -c "from engine.model.loader import resolve_model_path; print(resolve_model_path('$MODEL'))"  # about 16 GB

SERVER=("$PY" -m torch.distributed.run --nnodes 2 --nproc-per-node 1 --node-rank "$NODE_RANK"
        --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
        -m engine.api.server --model "$MODEL" --pipeline-parallel-size 2 --max-model-len 4096
        --enable-chunked-prefill --max-batch-tokens 2048 --enable-prefix-caching --log-level warning)

if [ "$NODE_RANK" != 0 ]; then
  exec "${SERVER[@]}"  # the second stage; exits when machine 0 shuts the server down
fi

OUT="$RESULTS/distributed_8b"
mkdir -p "$OUT"
"${SERVER[@]}" > "$OUT/pp2-2nodes_server.log" 2>&1 &
PID=$!
trap 'kill -INT $PID 2>/dev/null || true; wait $PID 2>/dev/null || true' EXIT
for _ in $(seq 1 1800); do  # the second machine may still be downloading the model
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break
  kill -0 "$PID" 2>/dev/null || { echo "server exited; see $OUT/pp2-2nodes_server.log"; exit 1; }
  sleep 1
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "server did not become healthy"; exit 1; }
$PY -m bench.run_bench run --url http://127.0.0.1:8000 --model "$MODEL" --workload bench/data/sharegpt.jsonl \
  --rates 1,2,4,8,12,16 --duration 45 --label pp2-2nodes --out "$OUT"
