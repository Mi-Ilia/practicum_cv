_base_ = [
    '../../mmdetection/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py',
]

# только меняем число классов
model = dict(
    bbox_head=dict(
        num_classes=17
    )
)

import os
# Используем абсолютный путь для надежности
data_root = os.path.abspath('datasets/minecraft')

metainfo = dict(
    classes=(
        'bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog',
        'ghast', 'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider',
        'turtle', 'wolf', 'zombie'
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances')),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='BaseDetDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/train_annotations.voc.json',
        data_prefix=dict(img='', img_path=''),  # Указываем оба ключа для применения data_root
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseDetDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val_annotations.voc.json',
        data_prefix=dict(img='', img_path=''),  # Указываем оба ключа для применения data_root
        test_mode=True,
        pipeline=test_pipeline)
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseDetDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test_annotations.voc.json',
        data_prefix=dict(img='', img_path=''),  # Указываем оба ключа для применения data_root
        test_mode=True,
        pipeline=test_pipeline)
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=None,
    metric='bbox',
)

test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=24,     
    val_interval=1
)

param_scheduler = [
    dict(type='ConstantLR', factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1
    )
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=10, norm_type=2),
    loss_scale=512.0,
)

# Настройки логирования
log_level = 'INFO'

log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True
)

default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=50,
    ),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP',  
        rule='greater',             
    ),
)