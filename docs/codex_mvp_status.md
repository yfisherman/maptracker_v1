# MVP Status

- **Current status:** Milestone 2 complete (temporal feature plumbing + local read-view corruption paths for clean/C-full/C-tail, without gate loss).
- **Milestone-2 scope note:** The implementation followed `docs/codex_mvp_plan.md` Milestone 2 tensor plumbing and additionally implemented the prompt-required read-path corruption outputs (`slot_corrupt_mask`, eligibility, clean/C-full/C-tail local views) while keeping canonical write behavior unchanged.

## What changed in Milestone 2

1. `plugin/models/mapers/vector_memory.py`
   - Extended read-path bookkeeping dictionaries to expose gate metadata and corruption labels:
     - `batch_valid_mem_dict`
     - `batch_delta_t_int_dict`
     - `batch_age_rank_norm_dict`
     - `batch_mem_clean_embeds_dict`
     - `batch_slot_corrupt_mask_dict`
     - `batch_slot_corrupt_eligible_dict`
     - `batch_mem_corrupt_mode_dict`
   - Added `_build_local_corrupted_read_view(...)` helper that builds local read tensors for:
     - `clean`
     - `c_full`
     - `c_tail`
   - Implemented missing-stale-source handling as **unsupervised/ineligible** (no fabricated replacement).
   - Kept corruption strictly local to read path: canonical `mem_bank` and write/update logic are not modified by corruption construction.
   - Added age/valid/temporal metadata construction in `trans_memory_bank(...)`:
     - `valid_mem` from `key_padding_mask`
     - `delta_t_int` from `relative_seq_idx`
     - `age_rank_norm` from selected slot order.

2. `plugin/models/transformer_utils/MapTransformer.py`
   - Extended `SlotwiseTemporalGate` input contract to accept and consume:
     - `valid_mem`
     - `delta_t_int`
     - `age_rank_norm`
     - `clean_mem_embeds` (exposed for gate-side use)
   - Updated memory cross-attention read path (`attn_index == 2`) to forward the new metadata tensors from `VectorInstanceMemory`.

## Validation run for Milestone 2

1. ✅ Syntax/compile check passed:
   - `python -m py_compile plugin/models/mapers/vector_memory.py plugin/models/transformer_utils/MapTransformer.py`

2. ⚠️ Targeted runtime tensor/mask checks were prepared but blocked by environment dependency:
   - `python - <<'PY' ...` (failed with `ModuleNotFoundError: No module named 'torch'`)

3. ✅ Static verification by code inspection confirms:
   - Corruption is read-path local and uses detached local views.
   - Missing stale sources remain ineligible (no fabricated source tensors).
   - Canonical memory write path remains in `update_memory(...)` and was not altered.

## Remaining for Milestone 3

- Integrate corruption mode/onset controls from training configuration paths (rather than metadata defaults).
- Thread corruption labels/eligibility through full training outputs as needed for downstream gate supervision wiring.
- Add end-to-end runtime validation in an environment with `torch` available (including explicit C-full/C-tail behavior checks and canonical-memory isolation assertions at runtime).
