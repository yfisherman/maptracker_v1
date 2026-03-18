import os

_base_ = ['../nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py']

_nusc_root = os.environ.get('MAPTRACKER_NUSC_ROOT', './datasets/nuscenes')
_val_ann = os.path.join(_nusc_root, 'nuscenes_map_infos_val.pkl')

# Explicit Adroit-friendly config copy: keep official evaluation settings,
# but resolve dataset roots from environment so compute nodes can read from
# /scratch/network/$USER without ad hoc CLI overrides.
eval_config = dict(
    data_root=_nusc_root,
    ann_file=_val_ann,
)

match_config = dict(
    data_root=_nusc_root,
    ann_file=_val_ann,
)

data = dict(
    workers_per_gpu=4,
    val=dict(
        data_root=_nusc_root,
        ann_file=_val_ann,
        eval_config=eval_config,
    ),
    test=dict(
        data_root=_nusc_root,
        ann_file=_val_ann,
        eval_config=eval_config,
    ),
)
