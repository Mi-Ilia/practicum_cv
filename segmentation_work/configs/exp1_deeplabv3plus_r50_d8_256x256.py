"""Конфигурация Эксперимент 1: DeepLabV3+ с ResNet-50 + Усиленные аугментации

**Идея эксперимента**  
Усиление аугментаций относительно H1 для улучшения обобщающей способности модели, 
особенно на сложных сценах с собаками (непривычные ракурсы, освещение).

**Что меняется относительно H1**  
- Добавлены геометрические аугментации:
  - `RandomResize` с масштабированием в диапазоне (0.5, 2.0) с сохранением соотношения сторон
  - `RandomCrop` (256×256) с `cat_max_ratio=0.75` для защиты от полного вырезания объектов
- Добавлены фотометрические аугментации:
  - `PhotoMetricDistortion` с легкими параметрами (яркость, контраст, насыщенность, оттенок)
- Сохранена базовая аугментация:
  - `RandomFlip` (prob=0.5, horizontal)

**Архитектура** (как в H1)  
- Backbone: ResNet‑50 (ImageNet pretrained)  
- Head: DeepLabV3+ (DepthwiseSeparableASPPHead)  
- Classes: 3 (фон, кот, собака)  
- Input size: 256×256  

**Loss** (как в H1)  
- ComboLoss: 0.5 * CrossEntropy + 0.5 * DiceLoss
  Реализовано через стандартные loss‑функции mmsegmentation:  
  `loss_decode = [CrossEntropyLoss(loss_weight=0.5), DiceLoss(loss_weight=0.5)]`  

**Ожидаемый эффект**  
- Меньше переобучения на «типичных» сценах
- Улучшение mDice/mIoU на классе **dog** (нестабильные ракурсы и освещение)
- Возможное небольшое снижение на простых примерах, но рост на сложных
"""

_base_ = [
    '../mmsegmentation/configs/_base_/models/deeplabv3plus_r50-d8.py',
    '_base_/datasets/train_dataset_for_students.py',
    '_base_/schedules/schedule_50e.py',
    '../mmsegmentation/configs/_base_/default_runtime.py',
]

# Default scope
default_scope = 'mmseg'

# Размер изображения (определяем до использования в pipeline)
crop_size = (256, 256)

# Метаданные датасета
metainfo = dict(
    classes=('background', 'cat', 'dog'),
    palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0]]  # Black, Red, Green
)

# Pipelines с усиленными аугментациями
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
        brightness_delta=32,  # Изменение яркости ±32 из 255
        contrast_range=(0.8, 1.2),  # Легкое изменение контраста
        saturation_range=(0.8, 1.2),  # Легкое изменение насыщенности
        hue_delta=18  # Изменение оттенка ±18 градусов
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

# Data preprocessor
data_preprocessor = dict(size=crop_size)

# Модель (идентична H1)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True
    ),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=3,  # фон, кот, собака
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
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
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=0.2,
                loss_name='loss_ce_aux',
                avg_non_ignore=False
            ),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                activate=True,
                reduction='mean',
                naive_dice=False,
                loss_weight=0.2,
                ignore_index=255,
                eps=1e-3,
                loss_name='loss_dice_aux'
            )
        ]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# DataLoader
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=True,
        round_up=True
    ),
    dataset=dict(
        type='BaseSegDataset',
        data_root='datasets/train_dataset_for_students',
        metainfo=metainfo,
        data_prefix=dict(
            img_path='img/train',
            seg_map_path='labels_sam2/train'
        ),
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
        data_prefix=dict(
            img_path='img/val',
            seg_map_path='labels_sam2/val'
        ),
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
        data_prefix=dict(
            img_path='img/test',
            seg_map_path='labels_sam2/test'
        ),
        pipeline=test_pipeline
    )
)

# Evaluator
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
    prefix='val'
)

test_evaluator = val_evaluator

# Default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=999999,
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

# Visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Randomness
randomness = dict(seed=42, deterministic=False)

# Логирование
log_level = 'WARNING'

# Оптимизатор и mixed precision
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-4,
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer,
    loss_scale='dynamic'
)
