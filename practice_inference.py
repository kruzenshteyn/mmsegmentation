from mmseg.apis import inference_model, init_model, show_result_pyplot


#  Загружаем модель 
#  Замените пути на свои 
config_file = 'D:/WORK/Practicum/_NNCV/VS/Sprint_6/SP6_PRJ/UNet/mmsegmentation/work_dirs/unet_256x256/unet_256x256.py'
checkpoint_file = 'D:/WORK/Practicum/_NNCV/VS/Sprint_6/SP6_PRJ/UNet/mmsegmentation/work_dirs/unet_256x256/epoch_280.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')

# Получаем результат 
# Замените путь картинки на актуальный 
# mmsegmentation\data\practice_dataset\img\val\000000000307_5917.jpg
img = 'D:/WORK/Practicum/_NNCV/VS/Sprint_6/SP6_PRJ/UNet/mmsegmentation/data/practice_dataset/img/test/000000306630_7423.jpg'  

result = inference_model(model, img)


# Сохраняем результат 
show_result_pyplot(
    model, 
    img, 
    result,
    opacity=0.5,
    with_labels=True,
    draw_gt=False,
    show=True,
    out_file="simple_inference.jpg",
    save_dir="viz_outputs",
)