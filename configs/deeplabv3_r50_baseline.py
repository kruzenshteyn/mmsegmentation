_base_ = [
    '../configs/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug.py',
    '../configs/_base_/datasets/custom_dataset.py'
]

model = dict(decode_head=dict(num_classes=6))
data_preprocessor = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2),
    dict(type='PackSegInputs')
]

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)