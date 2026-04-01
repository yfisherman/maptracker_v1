#!/usr/bin/env bash

set -euo pipefail

CONFIG=$1
CHECKPOINT=$2
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:3}
SRUN_ARGS=${SRUN_ARGS:-""}
STEP_NODES=$(( (GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))

export SRUN_CPUS_PER_TASK=${CPUS_PER_TASK}
PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" \
srun --nodes=${STEP_NODES} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --gpus-per-task=1 \
    --kill-on-bad-exit=1 \
    --wait=0 \
    ${SRUN_ARGS} \
    python -u tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" ${PY_ARGS}