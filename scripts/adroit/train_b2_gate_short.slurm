#!/bin/bash
#SBATCH --job-name=mt_b2_gate
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
WORK_ROOT=${WORK_ROOT:-/scratch/network/$USER/maptracker_runs/train}
RUN_NAME=${RUN_NAME:-b2_gate_short}
WORK_DIR=${WORK_DIR:-$WORK_ROOT/$RUN_NAME}
MAPTRACKER_NUSC_ROOT=${MAPTRACKER_NUSC_ROOT:-/scratch/network/$USER/maptracker_data/datasets/nuscenes}
MAPTRACKER_STAGE2_CKPT=${MAPTRACKER_STAGE2_CKPT:-/scratch/network/$USER/maptracker_runs/stage2/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth}
SEED=${SEED:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$WORK_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MAPTRACKER_NUSC_ROOT
export MAPTRACKER_STAGE2_CKPT

echo "[train-b2] repo=$REPO_ROOT"
echo "[train-b2] config=$CONFIG"
echo "[train-b2] work_dir=$WORK_DIR"
echo "[train-b2] stage2_ckpt=$MAPTRACKER_STAGE2_CKPT"

srun python tools/train.py \
  "$CONFIG" \
  --launcher slurm \
  --work-dir "$WORK_DIR" \
  --seed "$SEED" \
  $EXTRA_ARGS
