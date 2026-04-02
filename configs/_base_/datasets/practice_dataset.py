dataset_type = 'PracticeDataset'
data_root = "D:/WORK/Practicum/_NNCV/VS/Sprint_6/mmsegmentation/data/practice_dataset"


# ==== Определяем обуающий пайплайн данных ======
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # Аугментацции 
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomRotFlip', degree=(-45, 45)),
    dict(type='RandomCutOut', prob=0.4, n_holes=(7, 15), cutout_ratio=(0.1, 0.15)),
    # dict(type='Albu', transforms=[dict(type='ElasticTransform', alpha=1.0, sigma=35, p=0.4),
    #                               dict(type="GridDistortion", num_steps=10, p=0.55)]),
    # ===
    dict(type='PackSegInputs')
]


train_dataset=dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='img/train',
        seg_map_path='labels/train'),
    pipeline=train_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset   
)


# ==== Определяем валидационный пайплайн данных ======
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs")
]
test_pipeline = val_pipeline

val_dataset = dataset=dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='img/val',
        seg_map_path='labels/val'),
    pipeline=val_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset
)


# ==== Определяем тестовый пайплайн данных ======
test_dataset = dataset=dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='img/test',
        seg_map_path='labels/test'),
    pipeline=test_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=val_dataset
# )

# В этом случае он у нас аналогичен валидационному
test_dataloader = val_dataloader

# Здесь же в пайплайне данных создаются объекты для подсчета метрик
val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = val_evaluator