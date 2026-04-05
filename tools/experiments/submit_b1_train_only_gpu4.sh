#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

RUN_ID="${RUN_ID-b1_stage3_gpu4_short_trainonly}"
PARTITION="${PARTITION-}"
QOS="${QOS-gpu-short}"
TIME_LIMIT="${TIME_LIMIT-06:00:00}"
MAIL_USER="${MAIL_USER-yk3904@princeton.edu}"
MAIL_TYPE="${MAIL_TYPE-BEGIN,END,FAIL}"
CONSTRAINT="${CONSTRAINT-a100&nomig}"
TRAIN_GPUS="${TRAIN_GPUS-4}"
EVAL_GPUS="${EVAL_GPUS-2}"
GPUS_PER_NODE="${GPUS_PER_NODE-2}"
CPUS_PER_TASK="${CPUS_PER_TASK-8}"

CFG_COMMON="num_gpus=4 data.samples_per_gpu=2 optimizer.lr=2.5e-4 lr_config.warmup_iters=1000 checkpoint_config.interval=3496 evaluation.interval=20976 runner.max_iters=167808"

CMD=(bash tools/experiments/submit_b1_b2_sbatch.sh
  --mode b1_only
  --run-id "$RUN_ID"
  --base-config plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py
  --base-checkpoint "$PROJECT_ROOT/work_dirs/pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth"
  --work-root "$PROJECT_ROOT/work_dirs"
  --time "$TIME_LIMIT"
  --mail-user "$MAIL_USER"
  --mail-type "$MAIL_TYPE"
  --qos "$QOS"
  --train-gpus "$TRAIN_GPUS"
  --eval-gpus "$EVAL_GPUS"
  --gpus-per-node "$GPUS_PER_NODE"
  --cpus-per-task "$CPUS_PER_TASK"
  --cfg-options-common "$CFG_COMMON"
  --b1-cfg-options "model.mvp_temporal_gate_cfg.corruption_trained_no_gate_baseline=True"
  --skip-train-validation
  --skip-final-eval)

if [[ -n "$PARTITION" ]]; then
  CMD+=(--partition "$PARTITION")
fi
if [[ -n "$CONSTRAINT" ]]; then
  CMD+=(--constraint "$CONSTRAINT")
fi

if [[ "${RESUME:-0}" == "1" ]]; then
  CMD+=(--resume)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CMD+=(--dry-run)
fi

cd "$PROJECT_ROOT"
"${CMD[@]}"