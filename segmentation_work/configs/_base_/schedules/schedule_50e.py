"""Расписание обучения: 50 эпох

Оптимизатор: AdamW
- lr = 1e-4
- weight_decay = 1e-4
- betas = (0.9, 0.999)

Scheduler: CosineAnnealingLR или ReduceLROnPlateau
"""

# Optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# Optimizer wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=35, norm_type=2)
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=0.9,
        begin=0,
        end=50 * 1000,  # 50 эпох * приблизительно 1000 итераций на эпоху
        by_epoch=False
    )
]

# Training loop
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=5  # Валидация каждые 5 эпох
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
