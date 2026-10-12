# cot-scene-token P-001: detection-only evaluation of the released LCT checkpoint
# (unilion_swin_384_seq_e2e.pth) on nuScenes val, to check the encoder before caching its BEV.
# Same model and test pipeline as unilion_swin_384_seq_e2e.py at b75c284, except:
#   - only the detection head is built and evaluated (occupancy, map, motion and planning off);
#   - the annotation loader does not read Occ3D labels, which this project does not use.
# The BEV features do not depend on which heads are built.
_base_ = ['../unilion_swin_384_seq_e2e.py']

model = dict(tasks=dict(det=True, occ=False, map=False, motion=False, planning=False))

evaluation = dict(
    eval_mode=dict(
        with_det=True,
        with_tracking=False,
        with_occ=False,
        with_map=False,
        with_motion=False,
        with_planning=False,
        tracking_threshold=0.2,
        motion_threshhold=0.2,
    ))

# Copied from the base test_pipeline; the only change is with_occ_3d=False.
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='ToEgo'),
    dict(
        type='PrepareImageInputs',
        is_train=False,
        data_config={{_base_.data_config}}),
    dict(type='LoadMultiTaskAnnotations3D', with_occ_3d=False),
    dict(
        type='BEVAug',
        bda_aug_conf={{_base_.bda_aug_conf}},
        classes={{_base_.class_names}}),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config={{_base_.grid_config}}),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='MultiTaskFormatBundle3D',
                class_names={{_base_.class_names}},
                with_label=False),
            dict(type='Collect3D', keys=['points',
                                         'img',
                                         'img_inputs',
                                         'gt_depth',
                                         'gt_bev_masks',
                                         'ego_status',
                                         'fut_boxes',
                                         'gt_ego_fut_trajs',
                                         'gt_ego_fut_masks',
                                         'gt_ego_fut_cmd'],
                 meta_keys=('filename', 'token', 'scene_token', 'timestamp', 'lidar2global'))
        ])
]

data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
