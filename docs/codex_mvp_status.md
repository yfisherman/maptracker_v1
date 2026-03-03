# MVP Status

- **Current status:** Milestone 1 complete (decoder insertion scaffold at pre-fusion memory path).
- **Exact architectural path:** `plugin/models/transformer_utils/MapTransformer.py` → `MapTransformerLayer.forward` → `layer == 'cross_attn'` with `attn_index == 2` (memory branch) → `SlotwiseTemporalGate` scales memory **values** before memory attention output is fused via `query = query_memory + query_bev`.
- **Identity-safe behavior:** Gate-disabled path returns `alpha=1` and uses ungated `mem_embeds` as values, preserving prior structure.

## Files changed in Milestone 1

1. `plugin/models/transformer_utils/MapTransformer.py`
   - Added `SlotwiseTemporalGate` scaffold module.
   - Added per-layer gate instantiation in `MapTransformerLayer.__init__` via `temporal_gate_cfg`.
   - Added pre-fusion callsite in memory cross-attention branch (`attn_index == 2`) to gate value path only.
2. `plugin/configs/maptracker/nuscenes_newsplit/maptracker_nusc_newsplit_5frame_span10_stage2_warmup.py`
3. `plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage2_warmup.py`
4. `plugin/configs/maptracker/av2_newsplit/maptracker_av2_100x50_newsplit_5frame_span10_stage2_warmup.py`
5. `plugin/configs/maptracker/av2_newsplit/maptracker_av2_newsplit_5frame_span10_stage2_warmup.py`
6. `plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup.py`
7. `plugin/configs/maptracker/nuscenes_newsplit/maptracker_nusc_newsplit_5frame_span10_stage3_joint_finetune.py`
8. `plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py`
9. `plugin/configs/maptracker/av2_newsplit/maptracker_av2_100x50_newsplit_5frame_span10_stage3_joint_finetune.py`
10. `plugin/configs/maptracker/av2_newsplit/maptracker_av2_newsplit_5frame_span10_stage3_joint_finetune.py`
11. `plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py`

## Validation commands run

1. `rg -n "class MapTransformerLayer|attn_index == 2|query_memory \+ query_bev|SlotwiseTemporalGate|alpha" plugin/models/transformer_utils/MapTransformer.py`
2. `python -m py_compile plugin/models/transformer_utils/MapTransformer.py`
3. `rg -n "temporal_gate|gate" plugin/configs/maptracker`
