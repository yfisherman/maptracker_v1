#!/bin/bash
#SBATCH --job-name=mt_resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$SLURM_SUBMIT_DIR}
CONFIG=${CONFIG:-plugin/configs/maptracker/adroit/nusc_oldsplit_stage3_b2_gate_short.py}
RESUME_FROM=${RESUME_FROM:-}
WORK_ROOT=${WORK_ROOT:-/scratch/network/$USER/maptracker_runs/train}
RUN_NAME=${RUN_NAME:-resume_run}
WORK_DIR=${WORK_DIR:-$WORK_ROOT/$RUN_NAME}
MAPTRACKER_NUSC_ROOT=${MAPTRACKER_NUSC_ROOT:-/scratch/network/$USER/maptracker_data/datasets/nuscenes}
MAPTRACKER_STAGE2_CKPT=${MAPTRACKER_STAGE2_CKPT:-/scratch/network/$USER/maptracker_runs/stage2/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth}
SEED=${SEED:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}

if [[ -z "$RESUME_FROM" ]]; then
  echo "ERROR: set RESUME_FROM=/path/to/checkpoint before submitting resume_train.slurm" >&2
  exit 2
fi

mkdir -p "$WORK_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MAPTRACKER_NUSC_ROOT
export MAPTRACKER_STAGE2_CKPT

echo "[resume] repo=$REPO_ROOT"
echo "[resume] config=$CONFIG"
echo "[resume] resume_from=$RESUME_FROM"
echo "[resume] work_dir=$WORK_DIR"

srun python tools/train.py \
  "$CONFIG" \
  --launcher slurm \
  --work-dir "$WORK_DIR" \
  --resume-from "$RESUME_FROM" \
  --seed "$SEED" \
  $EXTRA_ARGS
