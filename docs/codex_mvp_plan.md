# Temporal Gating MVP Implementation Plan (Repo-Grounded)

## Scope lock (must hold for all milestones)

This plan implements only the MVP defined in `docs/mvp_spec.md` and mapped to the current repo.

**In-scope MVP core:**
- Pre-fusion slotwise temporal gate inside VEC decoder memory path.
- Scalar `alpha` per query-slot (shared across heads).
- Value scaling only (no key scaling, no additive logits).
- Read-path-only corruption (canonical write bank remains clean).
- Corruption-trained no-gate baseline parity.

**Explicitly out of scope for first build:**
- Multi-head independent gates.
- Bilinear disagreement modules.
- BEV reliability branch.
- Additive logit bias.
- Geometry-localized routing.
- Translation/rotation corruption.
- Large ablation bureaucracy or speculative extensions.

## Repo-grounded deviation note (from spec vs code)

Current decoder memory attention in `plugin/models/transformer_utils/MapTransformer.py` (`MapTransformerLayer.forward`, `attn_index == 2`) consumes one memory tensor (`mem_embeds`) for both key/value via `MultiheadAttention`.

**Conservative MVP path:** do **not** force a broad K/V interface refactor first. Implement gating by constructing a gated value view (`V_mem_soft`) through a minimal compatibility wrapper in the same call path, while preserving current key usage and tensor layout. Only escalate to deeper K/V refactor if this minimal path proves impossible.

---

## Global stop-and-fix rule

After each milestone validation:
1. Run all listed validation commands.
2. If any command fails or acceptance criteria are not met, **stop**.
3. Repair only the current milestone scope until green.
4. Re-run validations for that milestone.
5. Move to next milestone only when current milestone is fully green.

---

## Milestone 1 — Decoder insertion scaffold at exact pre-fusion point

### Exact scope
- Add gate module scaffolding and per-layer instantiation in the VEC decoder stack.
- Insert callsite at exact pre-fusion memory path location:
  - `plugin/models/transformer_utils/MapTransformer.py`
  - `MapTransformerLayer.forward`
  - `layer == 'cross_attn'`, `attn_index == 2` branch, before memory-attention output is fused with `query_bev`.
- Keep behavior identity-safe when gate disabled (`alpha=1` equivalent).

### Files expected to change
- `plugin/models/transformer_utils/MapTransformer.py`
- `plugin/models/transformer_utils/__init__.py` (only if export needed)
- `plugin/configs/maptracker/*stage2_warmup.py` (minimal gate config knobs)
- `plugin/configs/maptracker/*stage3_joint_finetune.py` (minimal gate config knobs)

### Acceptance criteria
- Decoder builds with gate-enabled and gate-disabled config paths.
- Gate call occurs in pre-fusion memory branch (`attn_index == 2`) and not after `query_memory + query_bev`.
- Gate-disabled path is numerically equivalent in structure (no unintended loss wiring yet).

### Validation commands / smoke checks
- `rg -n "class MapTransformerLayer|attn_index == 2|query_memory \+ query_bev|SlotwiseTemporalGate|alpha" plugin/models/transformer_utils/MapTransformer.py`
- `python -m py_compile plugin/models/transformer_utils/MapTransformer.py`
- `rg -n "temporal_gate|gate" plugin/configs/maptracker`

### Out of scope (this milestone)
- Corruption protocol.
- New losses.
- Full metadata plumbing.
- Any baseline/experiment reruns.

---

## Milestone 2 — Minimal temporal feature plumbing for gating

### Exact scope
- Thread minimal required inputs from existing memory path into gate:
  - current query state (`q_cur` from layer query tensor),
  - memory slots (existing `mem_embeds`),
  - slot validity from `key_padding_mask` (as temporary `valid_mem` proxy),
  - temporal age from `relative_seq_idx` already computed in `VectorInstanceMemory.trans_memory_bank`.
- Add derived `age_rank_norm` from selected slot order / age in memory read path.
- Keep tensor contracts aligned with current per-batch valid-track slicing behavior.

### Files expected to change
- `plugin/models/mapers/vector_memory.py`
- `plugin/models/transformer_utils/MapTransformer.py`
- `plugin/models/mapers/MapTracker.py` (only if metadata needs forwarding hook)

### Acceptance criteria
- Gate receives real temporal features, not constants.
- No change to canonical memory write behavior in `update_memory`.
- Memory attention still respects existing `valid_track_idx` and `key_padding_mask` semantics.

### Validation commands / smoke checks
- `rg -n "relative_seq_idx|key_padding_mask|age_rank|valid_mem|batch_.*dict" plugin/models/mapers/vector_memory.py`
- `rg -n "valid_mem|delta_t|age_rank|key_padding_mask" plugin/models/transformer_utils/MapTransformer.py`
- `python -m py_compile plugin/models/mapers/vector_memory.py plugin/models/transformer_utils/MapTransformer.py`

### Out of scope (this milestone)
- Corruption and `slot_corrupt_mask`.
- Gate loss terms.
- Multi-head/per-head gate behavior.

---

## Milestone 3 — Read-path stale corruption engine (clean/C-full/C-tail)

### Exact scope
- Implement read-path-only corruption in vector memory selection/packaging flow.
- Add corruption mode controls and onset handling in training path.
- Emit slot labels/masks for supervision:
  - `slot_corrupt_mask`
  - eligibility mask for validly replaceable slots.
- Ensure corrupted tensors are local read views only and never overwrite canonical bank.

### Files expected to change
- `plugin/models/mapers/vector_memory.py`
- `plugin/models/mapers/MapTracker.py` (to set per-clip corruption mode/context)
- `plugin/configs/maptracker/*stage2_warmup.py`
- `plugin/configs/maptracker/*stage3_joint_finetune.py`

### Acceptance criteria
- Clean/C-full/C-tail modes selectable from config.
- Canonical buffers (`mem_bank`, `mem_bank_seq_id`, pose buffers) remain unchanged by corruption path.
- `slot_corrupt_mask` aligns with valid selected slots and onset policy.

### Validation commands / smoke checks
- `rg -n "slot_corrupt_mask|c_full|c_tail|corrupt|onset|eligible" plugin/models/mapers/vector_memory.py plugin/models/mapers/MapTracker.py`
- `python -m py_compile plugin/models/mapers/vector_memory.py plugin/models/mapers/MapTracker.py`
- `rg -n "corrupt|c_full|c_tail|stale" plugin/configs/maptracker`

### Out of scope (this milestone)
- Gate loss computation.
- No-gate baseline wiring.
- Any translation/rotation corruption.

---

## Milestone 4 — Gate loss wiring + training outputs/logging

### Exact scope
- Add gate supervision losses per MVP (`L_close`, `L_open`, optional weak clean-open term).
- Use explicit supervision masks from corruption engine and slot eligibility.
- Add `alpha` logging summaries (at least affected vs preserved slot stats).
- Keep existing losses intact:
  - map head cls/reg,
  - seg losses,
  - existing trans losses.

### Files expected to change
- `plugin/models/mapers/MapTracker.py` (top-level loss aggregation)
- `plugin/models/transformer_utils/MapTransformer.py` (return/expose alpha stats)
- `plugin/models/heads/MapDetectorHead.py` (only if intermediary pass-through is required)
- `plugin/configs/maptracker/*stage2_warmup.py`
- `plugin/configs/maptracker/*stage3_joint_finetune.py`

### Acceptance criteria
- Total loss includes gate term only when gate/corruption supervision is enabled.
- Training forward returns stable `log_vars` with gate metrics.
- Existing loss keys remain present and numerically valid.

### Validation commands / smoke checks
- `rg -n "L_close|L_open|lambda_close|lambda_open|gate_loss|alpha" plugin/models/mapers/MapTracker.py plugin/models/transformer_utils/MapTransformer.py`
- `python -m py_compile plugin/models/mapers/MapTracker.py plugin/models/heads/MapDetectorHead.py plugin/models/transformer_utils/MapTransformer.py`

### Out of scope (this milestone)
- Large-scale metric ablation tables.
- Architectural expansions beyond scalar slotwise gate.

---

## Milestone 5 — Corruption-trained no-gate baseline parity + runbook hardening

### Exact scope
- Add explicit switch for corruption-trained no-gate baseline (same corruption/data path, gate disabled).
- Ensure config parity between gated and no-gate experiments except gate-specific knobs.
- Document minimal run matrix for MVP claim.

### Files expected to change
- `plugin/configs/maptracker/*stage2_warmup.py`
- `plugin/configs/maptracker/*stage3_joint_finetune.py`
- `docs/codex_mvp_status.md`

### Acceptance criteria
- One config path runs corruption with gate off (baseline parity).
- One config path runs corruption with gate on.
- Differences between the two configs are limited to gate-related settings.

### Validation commands / smoke checks
- `rg -n "use_temporal_gate|gate_enabled|corruption|c_full|c_tail|lambda_close|lambda_open" plugin/configs/maptracker`
- `git diff -- plugin/configs/maptracker | sed -n '1,220p'`

### Out of scope (this milestone)
- Full benchmark report production.
- New datasets or evaluation protocols.

---

## Milestone handoff checklist

Before implementation starts, confirm:
- The pre-fusion insertion point remains exactly in `MapTransformerLayer.forward` memory branch (`attn_index == 2`).
- Memory read corruption is isolated from memory write buffers.
- Any shape adaptations are documented inline where track-index filtering and slot masks are applied.

