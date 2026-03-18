import os

_base_ = ['../nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py']

_nusc_root = os.environ.get('MAPTRACKER_NUSC_ROOT', './datasets/nuscenes')
_train_ann = os.path.join(_nusc_root, 'nuscenes_map_infos_train.pkl')
_val_ann = os.path.join(_nusc_root, 'nuscenes_map_infos_val.pkl')
_stage2_ckpt = os.environ.get(
    'MAPTRACKER_STAGE2_CKPT',
    'work_dirs/maptracker_nusc_oldsplit_5frame_span10_stage2_warmup/latest.pth',
)

eval_config = dict(
    data_root=_nusc_root,
    ann_file=_val_ann,
)

match_config = dict(
    data_root=_nusc_root,
    ann_file=_val_ann,
)

# Safe 1-GPU smoke on Adroit: keep stage-3 logic, force clean clips only,
# shorten runtime, and linearly scale LR from official global batch 16 to 1.
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        data_root=_nusc_root,
        ann_file=_train_ann,
    ),
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

model = dict(
    mvp_temporal_gate_cfg=dict(
        gate_supervision_enabled=True,
        corruption_probs=dict(clean=1.0, c_full=0.0, c_tail=0.0),
        stale_offsets=[4, 8],
        corruption_onset=2,
        c_tail_keep_recent=1,
        gate_loss_weights=dict(lambda_close=1.0, lambda_open=0.5, lambda_clean=0.1),
        enable_clean_open_loss=True,
        clean_validation_only=True,
        run_contradiction_suite=False,
        corruption_trained_no_gate_baseline=False,
        freeze_stage='stage2_warmup',
        unfreeze_stage='stage3_joint_finetune',
    ),
    head_cfg=dict(
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    temporal_gate_cfg=dict(
                        enabled=True,
                        hidden_dims=64,
                    ),
                ),
            ),
        ),
    ),
)

optimizer = dict(lr=3.125e-5)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type='MyRunnerWrapper', max_iters=32)
checkpoint_config = dict(interval=16)
evaluation = dict(interval=16)
log_config = dict(
    interval=4,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)
load_from = _stage2_ckpt
