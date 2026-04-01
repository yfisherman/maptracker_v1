#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

RUN_ID="${RUN_ID-b1_stage3_gpu8_trainonly}"
PARTITION="${PARTITION-gpu}"
QOS="${QOS-gpu-short}"
TIME_LIMIT="${TIME_LIMIT-23:30:00}"
MAIL_USER="${MAIL_USER-yk3904@princeton.edu}"
CONSTRAINT="${CONSTRAINT-gpu80}"
TRAIN_GPUS="${TRAIN_GPUS-8}"
EVAL_GPUS="${EVAL_GPUS-2}"
GPUS_PER_NODE="${GPUS_PER_NODE-4}"
CPUS_PER_TASK="${CPUS_PER_TASK-5}"

CMD=(bash tools/experiments/submit_b1_b2_sbatch.sh
  --mode b1_only
  --run-id "$RUN_ID"
  --base-config plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py
  --base-checkpoint "$PROJECT_ROOT/work_dirs/pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth"
  --work-root "$PROJECT_ROOT/work_dirs"
  --time "$TIME_LIMIT"
  --mail-user "$MAIL_USER"
  --partition "$PARTITION"
  --qos "$QOS"
  --constraint "$CONSTRAINT"
  --train-gpus "$TRAIN_GPUS"
  --eval-gpus "$EVAL_GPUS"
  --gpus-per-node "$GPUS_PER_NODE"
  --cpus-per-task "$CPUS_PER_TASK"
  --skip-train-validation
  --skip-final-eval)

if [[ "${RESUME:-0}" == "1" ]]; then
  CMD+=(--resume)
fi

cd "$PROJECT_ROOT"
"${CMD[@]}"