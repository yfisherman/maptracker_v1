#!/bin/bash
#SBATCH --job-name=mt_eval_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$SLURM_SUBMIT_DIR}
CONFIG=${CONFIG:-plugin/configs/maptracker/adroit/nusc_oldsplit_stage3_eval_pretrained.py}
CHECKPOINT=${CHECKPOINT:-/scratch/network/$USER/maptracker_runs/pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth}
WORK_ROOT=${WORK_ROOT:-/scratch/network/$USER/maptracker_runs/eval}
RUN_NAME=${RUN_NAME:-baseline_eval}
WORK_DIR=${WORK_DIR:-$WORK_ROOT/$RUN_NAME}
MAPTRACKER_NUSC_ROOT=${MAPTRACKER_NUSC_ROOT:-/scratch/network/$USER/maptracker_data/datasets/nuscenes}
SEED=${SEED:-0}

mkdir -p "$WORK_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MAPTRACKER_NUSC_ROOT

echo "[eval] repo=$REPO_ROOT"
echo "[eval] config=$CONFIG"
echo "[eval] checkpoint=$CHECKPOINT"
echo "[eval] work_dir=$WORK_DIR"
echo "[eval] nuscenes_root=$MAPTRACKER_NUSC_ROOT"

srun python tools/test.py \
  "$CONFIG" \
  "$CHECKPOINT" \
  --launcher slurm \
  --work-dir "$WORK_DIR" \
  --eval \
  --eval-options save_semantic=True \
  --seed "$SEED"
