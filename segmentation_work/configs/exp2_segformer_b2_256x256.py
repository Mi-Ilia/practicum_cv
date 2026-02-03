"""Конфигурация Эксперимент 2: SegFormer-B2 + Усиленные аугментации

**Идея эксперимента**  
Проверка влияния трансформерной архитектуры SegFormer-B2 на качество сегментации
при тех же аугментациях, что и в Эксперименте 1, для сравнения «архитектура vs архитектура».

**Что меняется относительно H1**  
- Архитектура:
  - Backbone: SegFormer-B2 (MixVisionTransformer) вместо ResNet-50
  - Decoder: SegformerHead (MiT-decoder) вместо DepthwiseSeparableASPPHead
- Loss: ComboLoss (0.5 * CE + 0.5 * Dice) без class weights (как в H1)
- Аугментации: Те же усиленные аугментации, что в Эксперименте 1
- Hyperparams: Адаптированы под SegFormer (lr=6e-5, weight_decay=0.01)

**Ожидаемый эффект**  
- Потенциальный рост mDice по мелким и сложным объектам за счёт лучшего захвата контекста
- Более аккуратные контуры и меньше пропусков на сложных сценах с собаками
- Возможно, более высокая стоимость по памяти/времени
"""

_base_ = [
    '../mmsegmentation/configs/_base_/models/segformer_mit-b0.py',
    '_base_/datasets/train_dataset_for_students.py',
    '../mmsegmentation/configs/_base_/default_runtime.py',
]

default_scope = 'mmseg'

crop_size = (256, 256)

metainfo = dict(
    classes=('background', 'cat', 'dog'),
    palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0]]  # Black, Red, Green
)

# Pipelines с усиленными аугментациями (как в Эксперименте 1)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # Геометрические аугментации
    dict(
        type='RandomResize',
        scale=(256, 256),  # Базовый размер
        ratio_range=(0.5, 2.0),  # Масштабирование от 0.5x до 2x от базового размера
        keep_ratio=True
    ),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        cat_max_ratio=0.75  # Защита от полного вырезания объектов
    ),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # Фотометрические аугментации (легкие параметры)
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=18
    ),
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)

model = dict(  # ← ИСПРАВЛЕНО: убрали дубликат model =
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        norm_cfg=dict(type='LN', requires_grad=True),
        act_cfg=dict(type='GELU'),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'
        )
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,  # Без class weights (как в H1)
                loss_weight=0.5,
                loss_name='loss_ce',
                avg_non_ignore=False
            ),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                activate=True,
                reduction='mean',
                naive_dice=False,
                loss_weight=0.5,
                ignore_index=255,
                eps=1e-3,
                loss_name='loss_dice'
            )
        ]
    ),
    train_cfg=dict(),  # ← ДОБАВЛЕНО
    test_cfg=dict(mode='whole')  # ← ДОБАВЛЕНО
)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root='datasets/train_dataset_for_students',
        metainfo=metainfo,
        data_prefix=dict(img_path='img/train', seg_map_path='labels_sam2/train'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root='datasets/train_dataset_for_students',
        metainfo=metainfo,
        data_prefix=dict(img_path='img/val', seg_map_path='labels_sam2/val'),
        pipeline=val_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root='datasets/train_dataset_for_students',
        metainfo=metainfo,
        data_prefix=dict(img_path='img/test', seg_map_path='labels_sam2/test'),
        pipeline=test_pipeline
    )
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'], prefix='val')
test_evaluator = val_evaluator

# Оптимизатор для SegFormer (адаптированные hyperparams)
optim_wrapper = dict(
    type='AmpOptimWrapper',  # Mixed precision для ускорения
    optimizer=dict(
        type='AdamW',
        lr=6e-5,  # SegFormer обычно требует меньший lr
        weight_decay=0.01,
        betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)  # Head обучается быстрее
        }
    ),
    loss_scale='dynamic'
)

# Scheduler для SegFormer (PolyLR с warmup)
# Используем эпохальный scheduler для совместимости с train.py
# При batch_size=8 и ~800 train изображений: ~100 итераций/эпоху
# Для 80 эпох: ~8000 итераций
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=1.0,
        begin=0,
        end=80,
        by_epoch=True  # Эпохальный scheduler для совместимости
    )
]

# Training loop
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,  # Как в H1/H2
    val_interval=1
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=999999,  # Отключаем вывод по итерациям (используем tqdm)
        log_metric_by_epoch=True
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=5,
        save_best='val/mDice',
        rule='greater',
        max_keep_ckpts=3
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='SegVisualizationHook',
        draw=True,
        interval=500
    )
)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

randomness = dict(seed=42, deterministic=False)
log_level = 'WARNING'  # Уменьшаем детальность логов
