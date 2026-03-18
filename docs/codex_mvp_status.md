# MVP Status (Final Compliance Audit)

## Final implemented scope (as currently in repo)

- **Implemented core path:** pre-fusion temporal gating callsite is in decoder memory cross-attention branch (`MapTransformerLayer.forward`, `attn_index == 2`) before fusion with BEV path (`query = query_memory + query_bev`).
- **Implemented gate form:** scalar `alpha` from a lightweight MLP; value scaling path only (no additive attention-logit bias).
- **Implemented corruption protocol:** read-path corruption modes (`clean`, `c_full`, `c_tail`) with onset and stale offset controls in `VectorInstanceMemory`; canonical memory bank remains clean and corruption is applied only to local read tensors.
- **Implemented supervision/logging:** `L_close`, `L_open`, optional `L_clean`, gate metrics (`alpha_mean_*`, `affected_batch_fraction`) and clean-path strength ratio logging.
- **Implemented parity control:** corruption-trained no-gate mode is config-wired, and gate loss is now explicitly disabled when `corruption_trained_no_gate_baseline=True` to avoid penalizing no-gate runs.

## Compliance findings vs `docs/mvp_spec.md` / `docs/codex_mvp_plan.md`

### Pass (implemented and aligned)

1. **Gate placement is pre-fusion** in decoder memory branch, not post-fusion.
2. **Corruption is read-only** (local corrupted view built from clean/canonical tensors; write buffers not overwritten by corruption path).
3. **C-full/C-tail are present** with onset and stale-offset handling.
4. **Value scaling only** is used (no key scaling or additive logit-bias machinery).
5. **Corruption-trained no-gate parity control exists**, and gate-loss contamination in no-gate mode is now blocked.

### Missing / partial for a fully defendable MVP claim

1. **Strict propagated/new/pad query masking semantics are only partial.**
   - Current gating eligibility relies on `valid_track_idx`, `query_key_padding_mask`, and memory validity, but does not thread explicit propagated/new/pad masks exactly as the spec’s canonical formulation.
2. **Packed-call-path contract should remain explicit and defended.**
   - The decoder memory branch currently uses the intended packed formulation where `q_len == 1`, so the slotwise value scaling path is exact for the implemented call site rather than an arbitrary multi-query approximation.
   - This remains a contract that should be enforced by assertions/tests to prevent future misuse outside the packed memory-branch path.
3. **Runtime validation is incomplete in this environment.**
   - Only static/syntactic and lightweight local checks were run here; no end-to-end training/eval jobs were executed.

### Overengineering / drift checks

- No BEV reliability branch, no bilinear module, no additive logit bias, no geometry-localized routing, and no translation/rotation corruption were introduced for MVP path.
- Existing repo complexity around stage tags/config flags remains, but no new speculative architecture was added in this audit.

## Required manual next steps

1. Run short stage2 and stage3 training smokes for **gated** and **corruption-trained no-gate** paths with identical corruption schedule and data path.
2. Confirm B1 vs B2 parity with matched seeds/settings and verify that only gate-specific toggles differ.
3. Validate contradiction-suite and clean-validation reporting in full runtime environment.
4. Keep the packed `q_len == 1` contract and track-boundary guard covered by targeted tests in the full runtime environment.

## Known caveats

- This environment did not run full training/evaluation.
- Some tests are dependency-gated by local runtime availability (e.g., full torch stack / project runtime context).
- Current implementation follows the conservative no-major-refactor path and relies on the packed decoder memory call path (`q_len == 1`) plus guardrails rather than a broader arbitrary-`Q` refactor.

## Deferred items (intentionally not implemented)

- Multi-head independent gating.
- Bilinear disagreement feature branch.
- BEV reliability branch (`c_ent/c_coh/c_rel`).
- Additive attention-logit bias.
- Geometry-localized failure routing (`geom_fail_mask`, transformed previous-query routing).
- Translation/rotation corruption modes.
- Natural inconsistency subset and broad ablation expansion.

## Validation state breakdown

### Implemented

- Decoder pre-fusion gate callsite, corruption engine, gate supervision/loss/logging, and config wiring.

### Statically / syntactically checked

- Python compilation for touched implementation files.
- Existing local MVP tests and validation utility invocation (subject to environment/runtime deps).

### Runtime-validated in this session

- Limited to local command-level checks listed below in this file’s update commit.

### Blocked by current environment

- Full long-horizon training runs and benchmark-quality evaluation sweeps (clean + contradiction suite) were not executed in this session.
