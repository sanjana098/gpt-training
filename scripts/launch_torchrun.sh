#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/launch_torchrun.sh <nnodes> <nproc_per_node> <rdzv_endpoint_host:port> [extra_args]

NNODES=${1:-2}
NPER=${2:-4}
RDZV=${3:-"127.0.0.1:29400"}
shift 3 || true

export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Tensor parallel size is read from Hydra config; ensure LOCAL_WORLD_SIZE is set
export LOCAL_WORLD_SIZE=$NPER

torchrun \
  --nnodes $NNODES \
  --nproc_per_node $NPER \
  --rdzv_backend c10d \
  --rdzv_endpoint $RDZV \
  --max_restarts 3 \
  src/scripts/train.py --config configs/config.yaml "$@"
