# MapTracker Temporal Memory Gate — Full Research Project Context

This document is the canonical reference for future chat sessions. It covers the complete scope of the research project: motivation, architecture, implementation details, training protocol, evaluation, compliance status, and deferred scope.

---

## 1. Research Context & Motivation

This project extends **MapTracker** (ECCV 2024 Oral), a vector HD mapping architecture that formulates road element reconstruction as a tracking problem. MapTracker fuses **historical raster and vector memory latents** over time to produce temporally consistent lane/road geometry predictions across multi-camera autonomous driving sequences.

**Core vulnerability addressed:** Temporal memory systems accumulate "stale" information — historical embeddings that represent outdated map state contradicting current visual evidence. When stale priors survive into historical attention, they can override correct current perception, degrading performance during dynamic scene changes (construction zones, occlusions, wrong prior initialization).

---

## 2. Research Question

> "Can a pre-fusion, slotwise temporal memory gate inside MapTracker suppress stale historical vector memory before historical attention, improving contradiction recovery under controlled stale-memory corruption without materially hurting clean mapping performance?"

This is a targeted, falsifiable question. The experiment is controlled via a **corruption-trained no-gate baseline** that processes identical corrupted data without the gate active. Any performance gap on the contradiction recovery benchmark is attributable solely to the gate mechanism.

---

## 3. Datasets & Evaluation

**Datasets:** Argoverse2 (AV2) and NuScenes — both new-split and old-split configurations. Primary development target is AV2 new-split, 5-frame, span-10.

**Primary evaluation metrics:**
- **Vector AP** — per-class average precision on predicted polylines (divider, boundary, pedestrian crossing)
- **Raster segmentation IoU** — BEV semantic segmentation quality
- **C-mAP (Consistency mAP)** — tracking-level temporal consistency across frames (computed via `tools/tracking/calculate_cmap.py`)

**Gate-specific diagnostic metrics** (logged per training step):

| Metric | Meaning | Healthy Target |
|--------|---------|----------------|
| `alpha_mean_affected` | Mean gate scalar on corrupted slots | Low (<0.5) |
| `alpha_mean_preserved_recent` | Mean gate scalar on preserved recent slots | High (>0.8) |
| `alpha_mean_clean_recent` | Mean gate scalar on clean recent slots | ~1.0 |
| `affected_batch_fraction` | Fraction of batch items with eligible supervised slots | Coverage check |
| `historical_path_strength_ratio_clean` | `‖V_soft‖ / ‖V_clean‖` on clean clips | Detects over-suppression |
| `gate_loss`, `gate_loss_final_frame` | Total and final-frame gate loss | Training signal health |
| `L_close`, `L_open`, `L_clean` | Individual gate loss components | Diagnosing balance |

---

## 4. Architecture Overview

### 4.1 Base MapTracker (Unchanged)

- **Backbone:** BEVFormer-based multi-camera BEV encoder
- **Raster head:** `MapSegHead` — BEV semantic segmentation
- **Vector head:** `MapDetectorHead` → `MapTransformer` — query-based polyline detection with tracking
- **Memory bank:** `VectorInstanceMemory` — stores per-instance embeddings across frames, selected via strided memory fusion (up to `T` historical slots per query)
- **Propagation:** Queries propagated frame-to-frame via Hungarian matching; new queries initialized for unmatched detections

### 4.2 The Pre-Fusion Slotwise Temporal Gate

**File:** `plugin/models/transformer_utils/MapTransformer.py`
**Class:** `SlotwiseTemporalGate` (lines 24–109)

**Placement:** Per decoder layer. Executes **strictly after BEV cross-attention but before historical vector-memory attention** — the pre-fusion position.

**Mechanism:**

1. Takes as input:
   - `q_cur`: BEV-updated query embedding for the current slot (post-BEV cross-attention) — shape `[1, B, D]`
   - `mem_embeds`: historical key/value embeddings from the memory bank — shape `[T, B, D]`
   - `valid_mem`: validity mask `[N, T]`
   - `delta_t_int`: integer frame offset per slot `[N, T]`
   - `age_rank_norm`: normalized recency rank `[N, T]`
   - `eligible_mask`: which queries are propagated (not new/pad)

2. Constructs gate features (all LayerNorm-stabilized):
   - `u_cur`: LN(q_cur) broadcast over T
   - `u_key`, `u_val`: LN(mem_embeds)
   - `cos_key`: cosine similarity between q_cur and each historical key
   - `l2_val`: L2 distance between q_cur and each historical value
   - `delta_t`, `age_rank_norm`: temporal metadata
   - `valid_mem`: binary validity

3. Gate MLP: `Linear → GELU → Linear → Sigmoid` → `alpha ∈ [0,1]` — shape `[B, Q, T]`

4. **Value scaling only**: `gated_values = alpha * mem_embeds` — keys are untouched; no additive logit bias

5. **Packed contract enforced**: `assert q_len == 1` — the gate processes one query slot at a time

**Gate disabled behavior:** When `enabled=False`, returns `alpha=1` everywhere (identity pass-through), preserving full gradient flow and ensuring the no-gate baseline is truly parameter-free.

### 4.3 VEC Decoder Layer

**Class:** `MapTransformerLayer` (lines 225–497 of `MapTransformer.py`)

Three sequential attention operations per layer:
1. **Self-attention** on query slots
2. **BEV cross-attention** → produces `query_bev` (current perceptual state)
3. **Memory cross-attention** — the gating point:
   - Fetches `mem_embeds`, temporal metadata, and `eligible_mask` from memory bank
   - Calls `temporal_gate.forward()` → `mem_values` (gated), `alpha`
   - Runs cross-attention using original keys but **gated values**
   - Stores `alpha` and `V_soft` in memory bank for loss computation downstream

**Bounds guard:** `_assert_valid_track_idx_in_bounds()` enforces that `valid_track_idx` never exceeds the actual tracked query count, preventing silent index overflow.

---

## 5. Read-Path Synthetic Corruption Engine

**File:** `plugin/models/mapers/vector_memory.py` (lines 201–404)

### 5.1 Design Principle

Corruption is **read-path-only**: a local corrupted view is built for each forward pass from the canonical (clean) write buffer. The canonical bank is never modified. This ensures:
- Causal correctness (future frames see clean history)
- Controlled negative examples with known ground-truth labels
- Perfect parity with the no-gate baseline (same corrupted data, gate disabled)

### 5.2 Corruption Modes

| Mode | Behavior | Supervised Slots |
|------|----------|-----------------|
| `clean` | No corruption; selected slots used as-is | L_clean weak signal on most-recent |
| `c_full` | All valid historical slots replaced with slots from `stale_offset` frames earlier | All eligible → L_close |
| `c_tail` | Only older slots (beyond `c_tail_keep_recent=1`) replaced; most-recent slot clean | Older → L_close; recent → L_open |

### 5.3 Stale Source Selection

Fetches from the **propagated selected** portion of the canonical bank offset by `stale_offset` frames back. If the source does not exist (clip too short), the slot is marked **ineligible** for gate supervision.

### 5.4 Corruption Scheduling

- **Probabilities per clip:** `clean=0.60, c_full=0.20, c_tail=0.20`
- **Stale offsets:** Sampled from `[4, 8]` frames per clip
- **Onset:** Corruption injected starting at frame `corruption_onset=2` (not frames 0/1)
- **Sampling functions:** `_sample_corruption_mode()` and `_resolve_stale_offset()` in `MapTracker.py`

### 5.5 Metadata Produced Per Batch Element Per Frame

| Field | Shape | Meaning |
|-------|-------|---------|
| `batch_mem_embeds_dict` | `[T, N, D]` | Corrupted read view |
| `batch_mem_clean_embeds_dict` | `[T, N, D]` | Clean reference |
| `batch_key_padding_dict` | `[N, T]` | Key padding mask |
| `batch_valid_mem_dict` | `[N, T]` | Valid memory mask |
| `batch_delta_t_int_dict` | `[N, T]` | Frame offset (integer) |
| `batch_age_rank_norm_dict` | `[N, T]` | Normalized recency rank |
| `batch_slot_corrupt_mask_dict` | `[N, T]` | Binary corruption label |
| `batch_slot_corrupt_eligible_dict` | `[N, T]` | Supervision eligibility |

---

## 6. Loss Design

**File:** `plugin/models/mapers/MapTracker.py` (lines 426–520)

```
L_close = BCE(alpha[eligible & corrupted],  target=0)    λ_close = 1.0
L_open  = BCE(alpha[eligible & ~corrupted], target=1)    λ_open  = 0.5
L_clean = BCE(alpha[valid & most_recent on clean], 1)    λ_clean = 0.1

L_gate = λ_close · L_close + λ_open · L_open + λ_clean · L_clean
```

- **L_close:** Forces gate to suppress corrupted slots. Highest weight — primary training signal.
- **L_open:** Prevents over-suppression of preserved recent slots. Active on C-tail mode.
- **L_clean:** Weak regularizer on clean clips. Prevents gate from collapsing to always-closed.

**Total loss integration:**
```
L_total = L_BEV + L_track + λ_trans · L_trans + L_gate
```

Gate loss only accumulates when:
- `gate_supervision_enabled == True`
- `use_memory == True`
- `not corruption_trained_no_gate_baseline`

---

## 7. No-Gate Baseline Parity Control

The **corruption-trained no-gate baseline** is activated by setting `corruption_trained_no_gate_baseline=True` in config. It:
1. Keeps the full corruption engine active (identical corrupted inputs)
2. Calls `_enforce_no_gate_baseline_runtime()` at the start of each forward pass, setting `gate.enabled=False` across all decoder layers
3. Skips all gate loss computation

This produces a perfectly matched control: same corrupted data, same base model, no gate. Any performance gap between gate and no-gate on the contradiction recovery benchmark is attributable solely to the gate mechanism.

---

## 8. Training Stages

| Stage | Purpose | Gate | Corruption | Checkpoint Source |
|-------|---------|------|------------|-------------------|
| **Stage 1: BEV Pretrain** | Train BEV encoder + vector head, no memory | Disabled | N/A | Scratch / ImageNet init |
| **Stage 2: Gate Warmup** | Freeze backbone; warm up gate on corruption | Enabled | 60% clean / 20% C-full / 20% C-tail | Stage 1 ckpt |
| **Stage 3: Joint Finetune** | Unfreeze; co-adapt full model with gate | Enabled | Same | Stage 2 ckpt |

Config flags `freeze_stage='stage2_warmup'` / `unfreeze_stage='stage3_joint_finetune'` control parameter freezing.

---

## 9. MVP Configuration

**Primary config:** `plugin/configs/maptracker/av2_newsplit/maptracker_av2_newsplit_5frame_span10_stage2_warmup.py`

```python
mvp_temporal_gate_cfg = dict(
    gate_supervision_enabled=True,
    corruption_probs=dict(clean=0.6, c_full=0.2, c_tail=0.2),
    stale_offsets=[4, 8],
    corruption_onset=2,
    c_tail_keep_recent=1,
    gate_loss_weights=dict(lambda_close=1.0, lambda_open=0.5, lambda_clean=0.1),
    enable_clean_open_loss=True,
    clean_validation_only=False,
    corruption_trained_no_gate_baseline=False,
    freeze_stage='stage2_warmup',
    unfreeze_stage='stage3_joint_finetune',
)

temporal_gate_cfg = dict(
    enabled=True,   # Set False for no-gate baseline
    hidden_dims=64,
)
```

---

## 10. Testing & Validation

**Unit tests:** `tests/test_temporal_gate_mvp.py` (141 lines, 10 test cases)

| Test | Validates |
|------|-----------|
| `test_dimension_smoke` | Forward pass output shapes are correct |
| `test_no_history` | Empty memory handled gracefully |
| `test_one_alpha_parity` | Disabled gate returns alpha=1 (identity) |
| `test_zero_alpha_suppression` | Zero alpha fully suppresses values |
| `test_gate_raises_when_q_len_not_one` | Packed contract enforced |
| `test_track_idx_boundary_guard_raises` | Bounds overflow detected |
| `test_corrupted_read_isolation` | Canonical bank unmodified after corruption |
| `test_corrupted_read_uses_propagated_selected_source` | Stale source correctness |
| `test_corrupted_read_missing_source_is_ineligible` | Missing stale source → ineligible |
| `test_c_tail_selectivity` | Older slots get lower alpha under C-tail |

**Static validation:** `tools/validate_milestone4_gate.py` — checks logged variables include `alpha_mean_affected`, `affected_batch_fraction`, `batch_alpha_soft_dict`, `batch_v_mem_soft_dict`.

---

## 11. Key Implementation Files

| File | Lines | Role |
|------|-------|------|
| `plugin/models/mapers/MapTracker.py` | 1,326 | Main model, corruption scheduling, gate loss, baseline parity |
| `plugin/models/transformer_utils/MapTransformer.py` | 635 | Gate class, decoder layer, memory attention with gated values |
| `plugin/models/mapers/vector_memory.py` | 434 | Memory bank, corruption engine, metadata |
| `docs/mvp_spec.md` | 488 | Authoritative MVP specification |
| `tests/test_temporal_gate_mvp.py` | 141 | Unit tests |
| `tools/validate_milestone4_gate.py` | — | Static compliance validation |

---

## 12. MVP Compliance Status

### Implemented & Verified (unit tests / static validation)

- Pre-fusion gate placement (after BEV cross-attn, before memory cross-attn)
- Scalar alpha per query-slot from lightweight MLP with LayerNorm-stabilized features
- Value scaling only (no key modification, no additive logit bias)
- Read-path-only corruption with clean canonical write buffer
- C-full and C-tail modes with onset, stale offset, and eligibility tracking
- Corruption-trained no-gate baseline parity control
- L_close / L_open / L_clean gate losses with correct weights
- All diagnostic metrics logged
- `q_len == 1` packed contract assertion
- `valid_track_idx` bounds guard

### Not Yet Runtime-Validated

No end-to-end training runs have been completed in the current environment. The following remain empirically unverified:
- Actual alpha distributions during training
- Whether gate improves contradiction recovery on held-out corruption eval
- Whether clean-split vector AP and C-mAP are preserved

### Explicitly Deferred (Out of Scope for This MVP)

- Multi-head independent per-head gating
- Bilinear disagreement features between q_cur and memory
- BEV reliability branch features (c_ent, c_coh, c_rel)
- Additive attention logit bias (vs. value scaling)
- Geometry-localized failure routing
- Translation/rotation corruption modes (only embedding-level corruption implemented)

---

## 13. Documentation Files

| File | Purpose |
|------|---------|
| `docs/mvp_spec.md` | Full authoritative specification (488 lines) |
| `docs/codex_mvp_plan.md` | Repo-grounded implementation roadmap by milestone |
| `docs/codex_mvp_status.md` | Final compliance audit results |
| `docs/mvp_repair_walkthrough.md` | Bug analysis and repair strategy |
| `docs/project_context.md` | **This file** — full project context for future sessions |
| `docs/installation.md` | Environment and dependency setup |
| `docs/getting_started.md` | Training/evaluation instructions |
| `docs/data_preparation.md` | AV2 and NuScenes dataset preparation |
| `AGENTS.md` | Agent implementation instructions |

---

## 14. Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| Value scaling only (no key modification) | Simpler; avoids softmax logit-bias complexity; maintains key structure for attention patterns |
| Scalar alpha (not per-head or per-dim) | Lightweight; slotwise selectivity is the core claim; per-head gating is deferred |
| LayerNorm on gate features | Prevents scale domination; stabilizes MLP training across diverse embedding magnitudes |
| Read-path-only corruption | Ensures causal correctness; canonical bank integrity is required for multi-step rollout |
| C-tail mode | The key differentiator from naive full-bank suppression; tests that gate is selective, not just aggressive |
| L_open weight (0.5) < L_close weight (1.0) | Asymmetric: suppressing corruption is harder to learn than preserving recent; prioritize close |
| `q_len == 1` packed contract | Each query attends memory independently; simplifies gate arithmetic and avoids cross-query interference in the gate |
| corruption_onset=2 | Avoids penalizing frame 0/1 where memory is naturally sparse |

---

## 15. Open Questions for Future Work

1. **Empirical validation:** Do alpha distributions behave as expected during actual training? Does the gate actually suppress corrupted slots?
2. **Clean performance:** Is vector AP / C-mAP on clean validation data preserved (within an acceptable margin)?
3. **Contradiction recovery:** On a held-out stale-corruption eval set (not seen during training), does the gate outperform the no-gate baseline?
4. **Stage 3 necessity:** Does joint fine-tuning (Stage 3) improve over Stage 2 warmup alone?
5. **Sensitivity:** How sensitive is performance to λ_close, λ_open, λ_clean, corruption probabilities, and stale offsets?
6. **Generalization:** Does the AV2-trained gate transfer to NuScenes?
