import os 
# from mmseg.datasets import DummyDataset

from mmseg.datasets import PracticeDataset
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from mmseg.visualization import SegLocalVisualizer
from mmengine.registry import init_default_scope
import random

# Устанавливаем область поиска "mmseg"
# Благодаря этому при инициализации по конфигу
# объекты (модели, датасеты, пайплайны) ищутся именно в mmsegmentation
init_default_scope('mmseg')


def load_dummy_ds() -> PracticeDataset:
    # Инициализируем путь до корневого каталога нашего проекта 
    mmseg_root = os.path.dirname(os.path.abspath(__file__))

    # Эти пути понадобятся для иницализации датасета, 
    # Папка с картинками получится объединением (data_root, data_prefix[img_path])
    # А папка с разметкой — объединением (data_root, data_prefix[seg_map_path])
    data_root = os.path.join(mmseg_root, "data", "practice_dataset")
    data_prefix=dict(img_path=os.path.join("img", "train"), seg_map_path=os.path.join("labels", "train"))


    # Сам датасет отвечает исклчительно за то, чтобы распознать структуру данных и корректно
    # считать метаинформацию 
    # Все остальное передаётся в качестве аргумента pipeline
    # Каждый элемент пайплайна — это конфиг модификатора, реализующего некую операцию
    # Например, LoadImageFromFile обогащает семпл картинкой 
    # А LoadAnnotations обогащает семпл картой сегментации     
    # Каждый модификатор описывается конфигом 
    reading_pipeline = [
        dict(type='LoadImageFromFile'), # Реализован в файле mmsegmentation/mmseg/datasets/transforms/loading.py
        dict(type='LoadAnnotations'), # Реализован в файле mmsegmentation/mmseg/datasets/transforms/loading.py
        dict(type='PhotoMetricDistortion'),
        dict(type='RandomRotFlip'),
        dict(type='RandomCutOut', prob=1, n_holes=(7, 15), cutout_ratio=(0.1, 0.15)),
    ]

    # Создаём наш датасет 
    dataset = PracticeDataset(
        data_root=data_root, 
        data_prefix=data_prefix, 
        pipeline=reading_pipeline,
        img_suffix=".jpg", # Расширение файлов с картинками
        seg_map_suffix=".png" # Расширение файлов с маской сегментации  
    )
    return dataset 


def plot_sample_demo(ds):    
    print(f"Загружен датасет длиной {len(ds)} элементов")

    # считываем метаинформацию  
    ds_meta = ds.metainfo

    # Подготовим визуализатор, результат будет в папке viz_outputs
    seg_local_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir="viz_outputs",
        alpha=0.5,
    )
    # Передаём в визуализатор метаинформацию нашего датасета 
    seg_local_visualizer.dataset_meta = dict(
        classes=ds_meta["classes"],
        palette=ds_meta["palette"]
    )
    random_number = random.randint(0, len(ds) - 1)
    # Оборачиваем семпл в структуру SegDataSample, совместимую с визуализатором 
    orig_sample = ds[random_number]
    print(random_number)
    plot_sample = SegDataSample()
    plot_sample.gt_sem_seg = PixelData(data=orig_sample["gt_seg_map"])

    # Отрисовываем семпл 
    img = orig_sample["img"]
    seg_local_visualizer.add_datasample(
        name="example",
        image=img,
        data_sample=plot_sample,
        show=True,
        draw_pred=True
    )


ds = load_dummy_ds()
plot_sample_demo(ds)