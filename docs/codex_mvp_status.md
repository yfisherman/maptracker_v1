# MVP Status

- **Current status:** Milestone 4 complete (gate loss wiring + training outputs/logging + MVP config/test wiring).
- **Scope note:** Kept the base MapTracker objective intact and added only MVP gate supervision terms (`L_close`, `L_open`, optional `L_clean`) plus gate diagnostics.

## Completed implementation work

1. **Gate supervision loss wiring (MapTracker):**
   - Added corruption-context sampling/wiring for clip-level `clean` / `c_full` / `c_tail` training metadata.
   - Added gate supervision aggregation with explicit masks from memory corruption outputs:
     - `L_close` on affected corrupted slots.
     - `L_open` on preserved recent slots.
     - optional weak `L_clean` on recent clean slots.
   - Added weighted `gate_loss` and conditional inclusion into total loss (`gate_supervision_enabled`).

2. **Required alpha and gate logging:**
   - `alpha_mean_affected`
   - `alpha_mean_preserved_recent`
   - `alpha_mean_clean_recent`
   - `affected_batch_fraction`
   - `historical_path_strength_ratio_clean`
   - plus scalar loss terms in logs: `gate_loss`, `L_close`, `L_open`, `L_clean`

3. **Minimal config/experiment wiring (stage2 + stage3 configs):**
   - Added `mvp_temporal_gate_cfg` fields for:
     - corruption probabilities
     - stale offsets
     - gate loss weights
     - freeze/unfreeze stage tags
     - clean-validation and contradiction-suite flags
     - corruption-trained no-gate baseline switch
   - Kept explicit stage wiring and temporal gate enable toggles in transformer layer configs.

4. **Tests and lightweight validation utility:**
   - Added MVP gate tests (dimension smoke, no-history, one-alpha parity, zero-alpha suppression, corrupted-read isolation, C-tail selectivity).
   - Added a lightweight static validator script for Milestone 4 wiring checks.

## Completed validations

- `python -m py_compile plugin/models/mapers/MapTracker.py plugin/models/transformer_utils/MapTransformer.py plugin/models/heads/MapDetectorHead.py tests/test_temporal_gate_mvp.py tools/validate_milestone4_gate.py`
- `python tools/validate_milestone4_gate.py`
- `python -m unittest tests/test_temporal_gate_mvp.py -v`

## Remaining manual training / evaluation steps

1. Run short gated training smoke on stage2 config with corruption enabled and verify:
   - non-zero `affected_batch_fraction` over training windows.
   - `L_close` and `L_open` both numerically active.
2. Run stage3 joint finetune smoke and verify alpha trends:
   - affected slots trend lower than preserved recent slots.
3. Run clean-validation pass and inspect `historical_path_strength_ratio_clean` for collapse/over-suppression.
4. Run contradiction-suite configured experiment path and compare gate/no-gate outputs.
5. Run corruption-trained no-gate baseline config path (same corruption schedule, gate disabled) for parity checks.

## Known limitations

- Runtime torch-based execution is blocked in this environment (`No module named 'torch'`), so dynamic tensor-behavior tests are added but currently skipped here.
- No long training jobs were launched in this prompt by design.
- The contradiction-suite and clean-validation flags are wired as config/runtime controls; end-to-end dataset/evaluator behavior must be validated in a full training/eval environment.
