# Princeton Adroit staged-run runbook

This runbook prepares the existing MapTracker-derived MVP repo for staged Slurm runs on Princeton Adroit using `srun` and repo-local config copies instead of long ad hoc CLI overrides.

## Scope and assumptions

- Target workflow: **nuScenes old split**, because the repo’s official baseline evaluation command and pretrained checkpoint documentation are written against `maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py`.
- Target cluster assumptions from the task:
  - GPU jobs use Slurm.
  - Compute nodes have **no internet access**.
  - Large data and outputs should live under `/scratch/network/$USER`.
  - Use `srun`, not `mpirun`.
- The repo already contains the temporal-gating MVP and the repair fixes. The Adroit additions here only harden the run path and provide explicit short-run configs.

## What already works in the current repo

### Existing no-edit execution map

1. **Official baseline evaluation from pretrained checkpoint** already exists:
   - config: `plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py`
   - checkpoint expected by docs: `work_dirs/pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth`
2. **Resume support** already exists via `tools/train.py --resume-from ...`.
3. **Online read-path corruption is already wired into training**:
   - `MapTracker.forward_train()` samples one clip-level corruption mode and stale offset, injects them into frame metas, and keeps the choice consistent across the clip.
   - `VectorInstanceMemory.trans_memory_bank()` builds a corrupted local read view from the clean selected memory view.
   - The canonical memory bank is not overwritten by corruption.
4. **Gated vs no-gate parity logic already exists**:
   - stage-3 gate forward is used when `temporal_gate_cfg.enabled=True`
   - the corruption-trained no-gate baseline is runtime-guarded by `corruption_trained_no_gate_baseline=True`

### Remaining practical gaps before Adroit runs

These were the missing pieces for cluster execution, which are now covered by the added files in this patch:

- No Adroit-specific Slurm scripts.
- No explicit short-run configs for smoke/B1/B2.
- Original configs assume repo-local `./datasets/nuscenes` and repo-local checkpoint paths, which is awkward on `/scratch/network/$USER` without symlinks or env-aware config copies.

## Added files

### Slurm entrypoints

- `scripts/adroit/eval_baseline.slurm`
- `scripts/adroit/train_clean_smoke.slurm`
- `scripts/adroit/train_b1_nogate_short.slurm`
- `scripts/adroit/train_b2_gate_short.slurm`
- `scripts/adroit/resume_train.slurm`

### Explicit Adroit configs

- `plugin/configs/maptracker/adroit/nusc_oldsplit_stage3_eval_pretrained.py`
- `plugin/configs/maptracker/adroit/nusc_oldsplit_stage3_clean_smoke.py`
- `plugin/configs/maptracker/adroit/nusc_oldsplit_stage3_b1_nogate_short.py`
- `plugin/configs/maptracker/adroit/nusc_oldsplit_stage3_b2_gate_short.py`

## One-time staging on Adroit login node

Run these once on a login node with internet access already handled externally.

```bash
export SCRATCH_ROOT=/scratch/network/$USER/maptracker_data
mkdir -p "$SCRATCH_ROOT/datasets" "$SCRATCH_ROOT/checkpoints" /scratch/network/$USER/maptracker_runs
```

### Required staged inputs

1. **nuScenes data + processed PKLs** under:

```bash
/scratch/network/$USER/maptracker_data/datasets/nuscenes
```

Expected required files include:

- `nuscenes_map_infos_train.pkl`
- `nuscenes_map_infos_val.pkl`
- supporting raw nuScenes folders (`maps`, `samples`, `v1.0-trainval`, etc.)

2. **Official pretrained stage-3 checkpoint** for baseline evaluation:

```bash
/scratch/network/$USER/maptracker_runs/pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth
```

3. **Stage-2 checkpoint** for stage-3 smoke/B1/B2 training:

```bash
/scratch/network/$USER/maptracker_runs/stage2/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth
```

If your stage-2 checkpoint lives elsewhere, export `MAPTRACKER_STAGE2_CKPT=/your/path/latest.pth` at submit time.

## Batch-size and LR adjustment on Adroit

The official nuScenes old-split stage-3 config was authored for **8 GPUs × batch size 2 per GPU = global batch 16** with **LR = 5e-4**.

The Adroit short-run configs here use **1 GPU × batch size 1 = global batch 1** to be safe across A100 80GB, A100 MIG 20GB, and V100 debugging nodes.

### Exact LR scaling used here

We apply linear scaling:

```text
adroit_lr = official_lr * (adroit_global_batch / official_global_batch)
          = 5e-4 * (1 / 16)
          = 3.125e-5
```

That exact LR is baked into the three Adroit stage-3 short-run configs.

### Why not gradient accumulation here?

This repo does **not** currently provide a dedicated, validated gradient-accumulation path in its existing train configs or custom runner. Because of that, the safest Adroit preparation is:

- keep the stage logic unchanged,
- reduce batch size to 1,
- scale LR linearly to `3.125e-5`.

If you later add and validate accumulation explicitly, you could instead keep `LR=5e-4` and accumulate **16 optimizer steps worth of microbatches** to match the official global batch, but that is **not** assumed by this runbook.

## Stage A — Official baseline evaluation from pretrained checkpoint

### Command

```bash
cd /path/to/repo
sbatch -A <ACCOUNT> -p gpu scripts/adroit/eval_baseline.slurm
```

### Optional explicit submit-time overrides

```bash
cd /path/to/repo
MAPTRACKER_NUSC_ROOT=/scratch/network/$USER/maptracker_data/datasets/nuscenes \
CHECKPOINT=/scratch/network/$USER/maptracker_runs/pretrained_ckpts/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth \
WORK_DIR=/scratch/network/$USER/maptracker_runs/eval/baseline_eval \
sbatch -A <ACCOUNT> -p gpu scripts/adroit/eval_baseline.slurm
```

### Expected outputs

Under `WORK_DIR`:

- evaluation logs
- `submission_vector.json`
- `pos_predictions.pkl`
- semantic outputs when `save_semantic=True`

### Verify after completion

- Slurm job exits `0`.
- `WORK_DIR/submission_vector.json` exists.
- `WORK_DIR/pos_predictions.pkl` exists.
- terminal output includes printed evaluation metrics.

### Success vs failure signals

**Success**
- mAP is in the expected official range for the pretrained old-split checkpoint.
- output artifacts exist and are non-empty.

**Failure / investigate**
- immediate failure opening annotation PKLs or images → dataset root staging problem.
- checkpoint load failure → wrong checkpoint path or incompatible file.
- empty predictions or near-zero mAP → likely wrong split/checkpoint mismatch.

## Optional post-baseline C-mAP commands

Run these after baseline eval if you want the consistency metric too.

```bash
python tools/tracking/prepare_pred_tracks.py \
  plugin/configs/maptracker/adroit/nusc_oldsplit_stage3_eval_pretrained.py \
  --result_path /scratch/network/$USER/maptracker_runs/eval/baseline_eval/submission_vector.json \
  --cons_frames 3

python tools/tracking/calculate_cmap.py \
  plugin/configs/maptracker/adroit/nusc_oldsplit_stage3_eval_pretrained.py \
  --result_path /scratch/network/$USER/maptracker_runs/eval/baseline_eval/pos_predictions_3.pkl
```

## Stage B — Clean modified-model smoke test

Purpose: prove that the modified stage-3 path loads, trains, checkpoints, and validates on **clean-only** clips before spending time on corruption training.

### Command

```bash
cd /path/to/repo
MAPTRACKER_STAGE2_CKPT=/scratch/network/$USER/maptracker_runs/stage2/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth \
sbatch -A <ACCOUNT> -p gpu scripts/adroit/train_clean_smoke.slurm
```

### Config details

- config: `plugin/configs/maptracker/adroit/nusc_oldsplit_stage3_clean_smoke.py`
- load source: stage-2 checkpoint via `MAPTRACKER_STAGE2_CKPT`
- duration: `32` iterations
- validation/checkpoint interval: `16`
- corruption schedule: forced clean (`clean_validation_only=True`, `clean=1.0`)
- gate: enabled

### What to verify

- training starts and does not crash in the memory/gate path.
- a checkpoint is written around iter 16 and/or iter 32.
- validation runs once.
- logs contain gate fields rather than all-zero or missing gate telemetry.

### Success metrics / log fields

Look for these training fields in the log JSON / stdout:

- `loss`
- `gate_loss`
- `alpha_mean_clean`
- `alpha_mean_affected`
- `alpha_mean_preserved_recent`
- `affected_batch_fraction`

**Expected smoke behavior**
- `loss` is finite.
- `gate_loss` is finite.
- `affected_batch_fraction` should stay near `0` because the smoke config is clean-only.
- no NaNs / no divergence.

**Failure signals**
- NaN in total loss or gate loss.
- shape/assert errors in memory cross-attention.
- repeated dataloader or annotation open failures.

## Stage C — B1 corruption-trained no-gate baseline short fine-tune

Purpose: run the corruption schedule **without** gate forward, preserving the repo’s intended B1 parity control.

### Command

```bash
cd /path/to/repo
MAPTRACKER_STAGE2_CKPT=/scratch/network/$USER/maptracker_runs/stage2/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth \
WORK_DIR=/scratch/network/$USER/maptracker_runs/train/b1_nogate_short \
sbatch -A <ACCOUNT> -p gpu scripts/adroit/train_b1_nogate_short.slurm
```

### Config details

- config: `plugin/configs/maptracker/adroit/nusc_oldsplit_stage3_b1_nogate_short.py`
- duration: `400` iterations
- validation/checkpoint interval: `200`
- corruption schedule: `clean=0.6`, `c_full=0.2`, `c_tail=0.2`
- gate module config: disabled
- parity flag: `corruption_trained_no_gate_baseline=True`

### What to verify

- job runs through corrupted clips without crashing.
- checkpoints appear at iter 200 and iter 400.
- evaluation completes.
- the run is truly no-gate, not just no-gate-loss.

### Success metrics / log fields

**Expected**
- `loss` finite.
- `affected_batch_fraction` > 0 on at least some iterations, showing corruption is active.
- gate-related close/open supervision should not dominate the run.
- if alpha logs appear, they should reflect the runtime-disabled no-gate baseline behavior rather than learned selective suppression.

**Failure**
- `affected_batch_fraction` stays identically `0` across the whole run despite corruption schedule.
- runtime asserts complain about missing temporal gate modules or checkpoint mismatch.

## Stage D — B2 gated short fine-tune

Purpose: run the intended staged gated fine-tune with the same short corruption schedule used for B1.

### Command

```bash
cd /path/to/repo
MAPTRACKER_STAGE2_CKPT=/scratch/network/$USER/maptracker_runs/stage2/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth \
WORK_DIR=/scratch/network/$USER/maptracker_runs/train/b2_gate_short \
sbatch -A <ACCOUNT> -p gpu scripts/adroit/train_b2_gate_short.slurm
```

### Config details

- config: `plugin/configs/maptracker/adroit/nusc_oldsplit_stage3_b2_gate_short.py`
- duration: `400` iterations
- validation/checkpoint interval: `200`
- corruption schedule: `clean=0.6`, `c_full=0.2`, `c_tail=0.2`
- gate enabled
- stage logic preserved: stage-3 fine-tune from stage-2 checkpoint

### What to verify

- job runs through checkpoint + validation.
- corruption is active.
- gate metrics are non-degenerate.
- B2 differs from B1 only in gate-related settings.

### Success metrics / log fields

**Expected**
- `loss` finite.
- `gate_loss` finite and non-trivial.
- `affected_batch_fraction` > 0 on corrupted batches.
- `alpha_mean_affected` trends lower than `alpha_mean_preserved_recent` once the run stabilizes.

**Failure**
- `alpha_mean_affected` and `alpha_mean_preserved_recent` are identical/noisy with no separation.
- `gate_loss` is NaN or exactly zero for the whole run.
- corruption never activates.

## Stage E — Resume a paused fine-tune

### Command

Resume the B2 run from its latest saved checkpoint:

```bash
cd /path/to/repo
CONFIG=plugin/configs/maptracker/adroit/nusc_oldsplit_stage3_b2_gate_short.py \
WORK_DIR=/scratch/network/$USER/maptracker_runs/train/b2_gate_short \
RESUME_FROM=/scratch/network/$USER/maptracker_runs/train/b2_gate_short/latest.pth \
sbatch -A <ACCOUNT> -p gpu scripts/adroit/resume_train.slurm
```

Resume B1 instead by changing `CONFIG` and `RESUME_FROM`:

```bash
cd /path/to/repo
CONFIG=plugin/configs/maptracker/adroit/nusc_oldsplit_stage3_b1_nogate_short.py \
WORK_DIR=/scratch/network/$USER/maptracker_runs/train/b1_nogate_short \
RESUME_FROM=/scratch/network/$USER/maptracker_runs/train/b1_nogate_short/latest.pth \
sbatch -A <ACCOUNT> -p gpu scripts/adroit/resume_train.slurm
```

### What to verify

- resumed log prints checkpoint restore successfully.
- iteration count advances past the resumed checkpoint.
- next checkpoint is written on schedule.
- metrics continue from the prior run instead of resetting as a fresh `load_from` initialization.

## Short debug/subset pathway

A true dataset subset pathway is **not** added here because the repo does not already expose a clean, dedicated subset flag in the train configs. The safe debug pathway added instead is the **clean smoke config**:

- same stage-3 model path,
- same stage-2 checkpoint loading logic,
- same single-GPU Adroit assumptions,
- only `32` iterations,
- clean clips only.

That gives a fast, low-risk debug step without introducing new dataset semantics.

## Bottom-line answers to the repo-state questions

### Does the current code already perform online read-path corruption as intended?

**Yes, with the current repo state the online read-path corruption is already implemented for training.**

Specifically:

- clip-level corruption metadata is sampled in `MapTracker.forward_train()` and injected into frame metas before the history/current passes.
- `VectorInstanceMemory.trans_memory_bank()` builds a **local corrupted read view**.
- corruption is applied to the read tensors only.
- the clean/canonical bank remains intact.
- `c_full` and `c_tail` are already supported.

### Is extra dataset/config plumbing still missing?

**No extra dataset-side corruption plumbing appears necessary for the intended MVP training path.**

What *was* still missing was operational plumbing for Adroit:

- Slurm scripts,
- scratch-friendly config copies,
- stage-specific exact commands,
- explicit resume procedure,
- explicit LR scaling guidance for 1-GPU runs.

Those are what this runbook and the added files provide.
