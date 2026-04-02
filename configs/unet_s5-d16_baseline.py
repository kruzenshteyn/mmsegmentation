# practicum_work/configs/unet_s5-d16_baseline.py
# Полностью рабочий конфиг U-Net (Этап 2, Гипотеза 1)

_base_ = [
    # '_base_/models/unet-s5-d16_deeplabv3_4xb4-40k_hrf-256x256.py',           # ← модель U-Net
    '_base_/datasets/custom_dataset.py',      # ← твой датасет (img/ + labels/)
    '_base_/default_runtime.py',              # логи, чекпоинты, визуализация
    '_base_/schedules/schedule_40k.py'        # расписание обучения 40k итераций
]

# ==================== МОДЕЛЬ ====================
model = dict(
    decode_head=dict(
        num_classes=6,                    # ← количество классов в твоём датасете
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_weight=2.0)   # Dice для целевой метрики mDice
        ]
    )
)

# ==================== ПРЕПРОЦЕССИНГ ====================
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255
)

# ==================== АУГМЕНТАЦИИ (бейзлайн) ====================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(256, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# ==================== ОБУЧЕНИЕ ====================
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
default_hooks = dict(checkpoint=dict(interval=2000))

# Для визуализации в ClearML (очень рекомендуется)
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
)