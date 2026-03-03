# MVP Status

- **Current status:** Prompt-scoped Milestone 3 complete for temporal gate behavior and pre-fusion integration in decoder memory attention.
- **Plan mismatch note:** `docs/codex_mvp_plan.md` labels Milestone 3 as stale corruption engine, while this prompt required SlotwiseTemporalGate completion and pre-fusion wiring checks. This update follows the prompt-scoped requirements with minimal code changes.

## What changed in this milestone

1. `plugin/models/transformer_utils/MapTransformer.py`
   - Updated `SlotwiseTemporalGate.forward(...)` to accept an explicit `eligible_mask` and enforce eligibility-only gating.
   - Ensured `alpha_soft` is explicitly zeroed for ineligible pairs (mask-driven, including key padding).
   - Kept gating as value scaling only; key tensors remain unchanged in memory cross-attention.
   - Preserved scalar alpha per eligible query-slot pair and applied value scaling by broadcasting alpha across embedding/head projections.
   - In pre-fusion `MapTransformerLayer.forward(...)` memory branch (`attn_index == 2`):
     - Applied gating after BEV-to-vector cross attention and before fusion (`query = query_memory + query_bev`).
     - Ensured new/non-tracked queries do not consume historical memory (only `valid_track_idx` are processed).
     - Ensured padded queries contribute no memory update (masked out via `query_key_padding_mask`, leaving zero memory contribution).
   - Exposed gate outputs for downstream use/logging on the memory object:
     - `memory_bank.batch_alpha_soft_dict[b_i]`
     - `memory_bank.batch_v_mem_soft_dict[b_i]`

## Validation run for this milestone

1. ✅ Syntax/compile check passed:
   - `python -m py_compile plugin/models/transformer_utils/MapTransformer.py`

2. ⚠️ Runtime validations blocked by missing dependency (`torch`) in this environment:
   - one-alpha parity check (`alpha = 1` reproduces ungated historical path)
   - zero-alpha suppression check (`alpha = 0` removes historical contribution)
   - no-history check for new/padded queries
   - dimension smoke test for gate path tensors

3. ✅ Strongest available static verification performed:
   - Confirmed pre-fusion insertion point and gating callsite in memory cross-attention path.
   - Confirmed explicit eligibility masking and ineligible-alpha zeroing logic.
   - Confirmed keys remain ungated while values are scaled.
