# _base_ = [
#     '../_base_/models/fcn_unet_s5-d16.py', 
#     '../_base_/datasets/practice_dataset.py',
#     '../_base_/default_runtime.py', 
#     '../_base_/schedules/practice_schedule.py'
# ]


# visualizer = dict(
#     type='Visualizer',
#     vis_backends=[
#         dict(type='LocalVisBackend'),      # сохраняем логи локально
#         dict(
#             type='ClearMLVisBackend',      # дублируем всё в ClearML
#             init_kwargs=dict(
#                 project_name='YaPracticum',
#                 task_name='unet_practice',
#                 reuse_last_task_id=False,
#                 continue_last_task=False,
#                 output_uri=None,
#                 auto_connect_arg_parser=True,
#                 auto_connect_frameworks=True,
#                 auto_resource_monitoring=True,
#                 auto_connect_streams=False,
#             )
#         )     
#     ]
# )


# # Определим размер входа 
# input_suze = (256, 256)

# # data_preprocessor = dict(size=input_suze)
# # model = dict(
# #     data_preprocessor=data_preprocessor,
# #     test_cfg=dict(mode="whole")
# # )

# data_preprocessor = dict(size=input_suze)
# model = dict(
#     data_preprocessor=data_preprocessor,
#     test_cfg=dict(mode="whole"),
#     decode_head=dict(
#         num_classes=2,
#         loss_decode=[
#             dict(
#                 type='CrossEntropyLoss',
#                 loss_name='loss_ce',
#                 use_sigmoid=False,
#                 loss_weight=1.0
#             ),
#             dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)
#         ]
#     ),
#     auxiliary_head=dict(
#         num_classes=2,
#         loss_decode=[
#             dict(
#                 type='CrossEntropyLoss',
#                 loss_name='loss_ce',
#                 use_sigmoid=False,
#                 loss_weight=1.0
#             ),
#             dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)
#         ]
#     )
# )

_base_ = [
    # '../_base_/models/deeplabv3_unet_s5-d16', 
    '../_base_/models/fcn_unet_s5-d16.py', 
    '../_base_/datasets/practice_dataset.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/practice_schedule.py'
]


visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),      # сохраняем логи локально
        dict(
            type='ClearMLVisBackend',      # дублируем всё в ClearML
            artifact_suffix=('.py', '.json', '.log'),
            init_kwargs=dict(
                project_name='YaPracticum_SP6',
                task_name='fcn_unet_s5_d16_Theory_1',
                reuse_last_task_id=False,
                continue_last_task=False,
                output_uri=None,
                auto_connect_arg_parser=True,
                auto_connect_frameworks=True,
                auto_resource_monitoring=True,
                auto_connect_streams=True,
            )
        )     
    ]
)


# Определим размер входа 
input_suze = (256, 256)
data_preprocessor = dict(size=input_suze)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode="whole"),
    decode_head=dict(
        num_classes=3,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=1.0
            ),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)
        ]
    ),
    auxiliary_head=dict(
        num_classes=3,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=1.0
            ),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)
        ]
    )
)