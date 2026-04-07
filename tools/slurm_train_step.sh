#!/usr/bin/env bash

set -euo pipefail

CONFIG=$1
WORK_DIR=$2
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=("${@:3}")
STEP_NODES=$(( (GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))

# Make distributed rendezvous deterministic per job step to reduce intermittent startup hangs.
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)}"
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export MASTER_PORT="${MASTER_PORT:-$((10000 + (SLURM_JOB_ID % 50000)))}"
fi
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}

echo "[slurm_train_step] MASTER_ADDR=${MASTER_ADDR:-unset} MASTER_PORT=${MASTER_PORT:-unset} GPUS=${GPUS} GPUS_PER_NODE=${GPUS_PER_NODE}"

declare -a SRUN_ARGS_ARR=()
if [[ -n "$SRUN_ARGS" ]]; then
    read -r -a SRUN_ARGS_ARR <<< "$SRUN_ARGS"
fi

export SRUN_CPUS_PER_TASK=${CPUS_PER_TASK}
PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" \
srun --nodes=${STEP_NODES} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --gpus-per-task=1 \
    --label \
    --kill-on-bad-exit=1 \
    --wait=0 \
    "${SRUN_ARGS_ARR[@]}" \
    python -u tools/train.py "$CONFIG" --work-dir="$WORK_DIR" --launcher="slurm" "${PY_ARGS[@]}"