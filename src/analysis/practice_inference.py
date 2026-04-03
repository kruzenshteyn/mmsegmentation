from mmseg.apis import inference_model, init_model, show_result_pyplot


# #  Загружаем модель 
# #  Замените пути на свои 
# config_file = 'D:/WORK/Practicum/_NNCV/VS/Sprint_6/SP6_PRJ/UNet/mmsegmentation/work_dirs/unet_256x256/unet_256x256.py'
# # checkpoint_file = 'D:/WORK/Practicum/_NNCV/VS/Sprint_6/SP6_PRJ/UNet/mmsegmentation/work_dirs/unet_256x256/epoch_280.pth'
# checkpoint_file = 'D:/WORK/Practicum/_NNCV/VS/Sprint_6/mmsegmentation/work_dirs/unet_256x256/epoch_500_Exp_1.pth'
# model = init_model(config_file, checkpoint_file, device='cuda:0')

# # Получаем результат 
# # Замените путь картинки на актуальный 
# # mmsegmentation/data/practice_dataset/img/val/000000000307_5917.jpg
# img = 'D:/WORK/Practicum/_NNCV/VS/Sprint_6/SP6_PRJ/UNet/mmsegmentation/data/practice_dataset/img/test/000000306630_7423.jpg'  

# result = inference_model(model, img)


# # Сохраняем результат 
# show_result_pyplot(
#     model, 
#     img, 
#     result,
#     opacity=0.5,
#     with_labels=True,
#     draw_gt=False,
#     show=True,
#     out_file="simple_inference.jpg",
#     save_dir="viz_outputs",
# )

from pathlib import Path
from tqdm import tqdm
import torch

from mmseg.apis import init_model, inference_model, show_result_pyplot

# ====================== НАСТРОЙКИ ======================
config_file = 'D:/WORK/Practicum/_NNCV/VS/Sprint_6/SP6_PRJ/UNet/mmsegmentation/work_dirs/unet_256x256/unet_256x256.py'

# === Пути к чекпоинтам (.pth) ===
checkpoint_files = [
    'D:/WORK/Practicum/_NNCV/VS/Sprint_6/mmsegmentation/work_dirs/unet_256x256/epoch_500_Exp_1.pth',
    'D:/WORK/Practicum/_NNCV/VS/Sprint_6/mmsegmentation/work_dirs/unet_256x256/epoch_300_Exp_1.pth',
    'D:/WORK/Practicum/_NNCV/VS/Sprint_6/mmsegmentation/work_dirs/unet_256x256/epoch_500_Exp_2.pth',
    'D:/WORK/Practicum/_NNCV/VS/Sprint_6/mmsegmentation/work_dirs/unet_256x256/epoch_300_T1.pth',
]

# === Пути к изображениям ===
image_files = [
    'D:/WORK/Practicum/_NNCV/VS/Sprint_6/mmsegmentation/data/practice_dataset/img/test/000000306630_7423.jpg',
    'D:/WORK/Practicum/_NNCV/VS/Sprint_6/mmsegmentation/data/practice_dataset/img/test/000000251623_4307.jpg',
    'D:/WORK/Practicum/_NNCV/VS/Sprint_6/mmsegmentation/data/practice_dataset/img/test/000000258129_3143.jpg',
    'D:/WORK/Practicum/_NNCV/VS/Sprint_6/mmsegmentation/data/practice_dataset/img/test/000000519850_4924.jpg',
    'D:/WORK/Practicum/_NNCV/VS/Sprint_6/mmsegmentation/data/practice_dataset/img/test/000000443499_491.jpg',
    'D:/WORK/Practicum/_NNCV/VS/Sprint_6/mmsegmentation/data/practice_dataset/img/test/000000412681_4015.jpg'
    # Добавляйте сюда все изображения:
    # 'D:/.../другое_изображение.jpg',
]

output_base_dir = Path("D:/WORK/Practicum/_NNCV/VS/Sprint_6/mmsegmentation/viz_outputs")   # Основная папка с результатами
device = 'cuda:0'
opacity = 0.5
# =======================================================

output_base_dir.mkdir(exist_ok=True, parents=True)

print(f"Найдено моделей: {len(checkpoint_files)}")
print(f"Найдено изображений: {len(image_files)}")
print(f"Всего будет обработано: {len(checkpoint_files) * len(image_files)} изображений/n")

for ckpt_path_str in tqdm(checkpoint_files, desc="Обработка моделей"):
    ckpt_path = Path(ckpt_path_str)
    model_name = ckpt_path.stem
    
    print(f"/n{'='*70}")
    print(f"Загружаем модель: {model_name}")
    print(f"{'='*70}")
    
    model = init_model(config_file, str(ckpt_path), device=device)
    
    # Папка для результатов этой модели
    model_out_dir = output_base_dir / model_name
    model_out_dir.mkdir(exist_ok=True)
    
    for img_path_str in tqdm(image_files, desc=f"Изображения → {model_name}", leave=False):
        img_path = Path(img_path_str)
        
        # Инференс
        result = inference_model(model, str(img_path))
        
        # Имя выходного файла
        out_filename = f"{img_path.stem}_{model_name}.jpg"
        
        # Сохранение результата (полностью как в вашем наброске)
        show_result_pyplot(
            model, 
            str(img_path), 
            result,
            opacity=opacity,
            with_labels=True,
            draw_gt=False,
            show=False,                    # важно при пакетной обработке!
            out_file= model_out_dir / out_filename,
            save_dir=str(model_out_dir),
        )
    
    # Очистка памяти GPU после модели
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"/n✅ Пакетная обработка успешно завершена!")
print(f"Результаты сохранены в папку: {output_base_dir.resolve()}")