


dataset_type = 'CustomDataset'
data_root = 'dataset/'                     # ← корень датасета

# ====================== METRICS ======================
val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
test_evaluator = val_evaluator

# ====================== PIPELINES (базовые) ======================
# Они будут переопределяться в каждом конфиге модели (unet, deeplab и т.д.)
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

# ====================== TRAIN DATALOADER ======================
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img/',          # если у тебя внутри img/train/ — поменяй на 'img/train/'
            seg_map_path='labels/'    # если у тебя внутри labels/train/ — поменяй на 'labels/train/'
        ),
        # ann_file='splits/train.txt',   # раскомментируй, если создашь файлы сплитов
        pipeline=train_pipeline,       # будет переопределяться в конфиге модели
        test_mode=False,
    )
)

# ====================== VAL DATALOADER ======================
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img/',          
            seg_map_path='labels/'    
        ),
        # ann_file='splits/val.txt',
        pipeline=test_pipeline,        # будет переопределяться в конфиге модели
        test_mode=True,
    )
)

test_dataloader = val_dataloader

