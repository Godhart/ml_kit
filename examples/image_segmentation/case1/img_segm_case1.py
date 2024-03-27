# Этот пример в google_colab можно посмотреть по ссылке https://colab.research.google.com/drive/1vOJJV5Z6seysKCXa5aXplG9EWdrOumCp

# 1. На основе учебного ноутбука проведите финальную подготовку данных. Иизмените количество сегментирующих классов с `16` на `5`.

# 2. Проведите суммарно не менее `10` экспериментов и визуализируйте их результаты (включая точность обучения сетей на одинаковом количестве эпох, например, на `7`):

#   - изменив `filters` в сверточных слоях
#   - изменив `kernel_size` в сверточных слоях
#   - изменив активационную функцию в скрытых слоях с `relu` на `linear` или/и `selu`, `elu`.


# **Важно!**

# Многие эксперименты могут приводить к переполнению ОЗУ в вашем ноутбуке и сброса кода обучения.

# ![image.png](data:image/png;base64,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)



# Для предотвращения переполнения ОЗУ может помочь библиотека `gc`. Вставьте строчку `gc.collect()` в цикл ваших экспериментов для сбора и удаления временных данных (кеш).

# Перед выполнением задания, пожалуйста, запустите ячейку `Подготовка` ниже:

## Подготовка

### Импорт библиотек

 # Импортируем модели keras: Model
from tensorflow.keras.models import Model

 # Импортируем стандартные слои keras
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation
from tensorflow.keras.layers import MaxPooling2D, Conv2D, BatchNormalization, UpSampling2D

# Импортируем оптимизатор Adam
from tensorflow.keras.optimizers import Adam

# Импортируем модуль pyplot библиотеки matplotlib для построения графиков
import matplotlib.pyplot as plt

# Импортируем модуль image для работы с изображениями
from tensorflow.keras.preprocessing import image

# Импортируем библиотеку numpy
import numpy as np

# Импортируем методделения выборки
from sklearn.model_selection import train_test_split

# загрузка файлов по HTML ссылке
import gdown

# Для работы с файлами
import os

# Для генерации случайных чисел
import random

import time

# импортируем модель Image для работы с изображениями
from PIL import Image

# очистка ОЗУ
import gc

### Загрузка датасета

# грузим и распаковываем архив картинок

# Загрузка датасета из облака
from pathlib import Path

dataset_source = 'https://storage.yandexcloud.net/aiueducation/Content/base/l14/construction_256x192.zip'
# 'https://storage.yandexcloud.net/aiueducation/Content/base/l14/construction_512x384.zip'

dataset_file = Path(".") / Path(dataset_source).parts[-1]
if not dataset_file.exists():
    gdown.download(dataset_source, None, quiet=False)

# !unzip -q 'construction_256x192.zip' # распоковываем архив

# Глобальные параметры

IMG_WIDTH = 192               # Ширина картинки
IMG_HEIGHT = 256              # Высота картинки
NUM_CLASSES = 16              # Задаем количество классов на изображении
TRAIN_DIRECTORY = 'train'     # Название папки с файлами обучающей выборки
VAL_DIRECTORY = 'val'         # Название папки с файлами проверочной выборки

# Загрузим оригинальные изображения (код из лекции):

train_images = [] # Создаем пустой список для хранений оригинальных изображений обучающей выборки
val_images = [] # Создаем пустой список для хранений оригинальных изображений проверочной выборки

cur_time = time.time()  # Засекаем текущее время

# Проходим по всем файлам в каталоге по указанному пути
for filename in sorted(os.listdir(TRAIN_DIRECTORY+'/original')):
    # Читаем очередную картинку и добавляем ее в список изображений с указанным target_size
    train_images.append(image.load_img(os.path.join(TRAIN_DIRECTORY+'/original',filename),
                                       target_size=(IMG_WIDTH, IMG_HEIGHT)))

# Отображаем время загрузки картинок обучающей выборки
print ('Обучающая выборка загружена. Время загрузки: ', round(time.time() - cur_time, 2), 'c', sep='')

# Отображаем количество элементов в обучающей выборке
print ('Количество изображений: ', len(train_images))

cur_time = time.time() # Засекаем текущее время

# Проходим по всем файлам в каталоге по указанному пути
for filename in sorted(os.listdir(VAL_DIRECTORY+'/original')):
    # Читаем очередную картинку и добавляем ее в список изображений с указанным target_size
    val_images.append(image.load_img(os.path.join(VAL_DIRECTORY+'/original',filename),
                                     target_size=(IMG_WIDTH, IMG_HEIGHT)))

# Отображаем время загрузки картинок проверочной выборки
print ('Проверочная выборка загружена. Время загрузки: ', round(time.time() - cur_time, 2), 'c', sep='')

# Отображаем количество элементов в проверочной выборке
print ('Количество изображений: ', len(val_images))

# Загрузим сегментированные изображения (код из лекции):

train_segments = [] # Создаем пустой список для хранений оригинальных изображений обучающей выборки
val_segments = [] # Создаем пустой список для хранений оригинальных изображений проверочной выборки

cur_time = time.time() # Засекаем текущее время

for filename in sorted(os.listdir(TRAIN_DIRECTORY+'/segment')): # Проходим по всем файлам в каталоге по указанному пути
    # Читаем очередную картинку и добавляем ее в список изображений с указанным target_size
    train_segments.append(image.load_img(os.path.join(TRAIN_DIRECTORY+'/segment',filename),
                                       target_size=(IMG_WIDTH, IMG_HEIGHT)))

# Отображаем время загрузки картинок обучающей выборки
print ('Обучающая выборка загружена. Время загрузки: ', round(time.time() - cur_time, 2), 'c', sep='')

# Отображаем количество элементов в обучающем наборе сегментированных изображений
print ('Количество изображений: ', len(train_segments))

cur_time = time.time() # Засекаем текущее время

for filename in sorted(os.listdir(VAL_DIRECTORY+'/segment')): # Проходим по всем файлам в каталоге по указанному пути
    # Читаем очередную картинку и добавляем ее в список изображений с указанным target_size
    val_segments.append(image.load_img(os.path.join(VAL_DIRECTORY+'/segment',filename),
                                     target_size=(IMG_WIDTH, IMG_HEIGHT)))

# Отображаем время загрузки картинок проверочной выборки
print ('Проверочная выборка загружена. Время загрузки: ', round(time.time() - cur_time, 2), 'c', sep='')

# Отображаем количество элементов в проверочном наборе сегментированных изображений
print ('Количество изображений: ', len(val_segments))

# ---------------------------------------------------------------------------- #

## Решение

# Ваше решение

# 1. Преобразовать загруженные изображения в матрицы
# 2. Изменить число классов (попробовать сделать через список значений)
# 3. Завести шаблон модели, учитывая варианты параметров
# 4. Обучить и посмотреть на результат

# ---------------------------------------------------------------------------- #

## Персональный вспомогательный код

# В целях упрощения написания, отладки вспомогательного кода и подключения его к ноутбукам - вспомогательный код добавляется через репозиторий github

### Код

import shutil
import sys
from pathlib import Path

ml_kit_path = str((Path(__file__).absolute() / ".." / ".." / ".." / "..").resolve())
if ml_kit_path not in sys.path:
    sys.path.insert(0, ml_kit_path)

try:
    from ml_kit.standalone import STANDALONE
except ImportError:
    STANDALONE = False

if STANDALONE:
    from ml_kit.helpers import *
    from ml_kit.trainer_common import *
    from ml_kit.plots import *
    from ml_kit.pics import *
    import ml_kit.boilerplate as bp
    from ml_kit.reporting import *

# ---------------------------------------------------------------------------- #

## Глобальные параметры

# Списки включения / исключения наборов гиперпараметров

RELOAD_DATA             = None  # Если True зараннее введённые параметры воссаздаются каждый раз с нуля
                                # (классы, статистика по классам и т.п.)
TARGET_CLASSES_COUNT    = 5     # Целевое значение до которого уменьшать кол-во классов (если их больше)
TARGET_CLASSES_BALANCE  = True  # Если True то объединение классов будет сбалансированным

TRAIN_DATA_LIMIT        = None  # Предел используемого числа данных для обучения. None для использования всех данных
TEST_DATA_LIMIT         = None  # Предел используемого числа данных для контроля обучения. None для использования всех данных

SHOW_PLOTS              = True  # True для отображения графики

TRAIN_INCLUDE = None  # Включать всё
TRAIN_EXCLUDE = None  # Ничего не исключать

# TRAIN_INCLUDE = [r"dns3b.*"]
# TRAIN_EXCLUDE = [r"cdns.*"]

def use_model(model_name):
    if TRAIN_EXCLUDE is not None:
        for item in TRAIN_EXCLUDE:
            if re.match(item, model_name) is not None:
                return False
    if TRAIN_INCLUDE is None:
        return True
    else:
        for item in TRAIN_INCLUDE:
            if re.match(item, model_name) is not None:
                return True
        return False


ENV[ENV__DEBUG_PRINT] = True
ENV[ENV__JUPYTER] = False

# Подключение google диска если работа в ноутбуке
# connect_gdrive()

# ---------------------------------------------------------------------------- #

# Перегрузка окружения под текущую задачу
ENV[ENV__TRAIN__DEFAULT_DATA_PATH]       = "lesson_11_lite_1a"
ENV[ENV__TRAIN__DEFAULT_OPTIMIZER]       = [Adam, [], to_dict(learning_rate=1e-3)]
ENV[ENV__TRAIN__DEFAULT_LOSS]            = S_SCCE
ENV[ENV__TRAIN__DEFAULT_METRICS]         = [S_SCA]
ENV[ENV__TRAIN__DEFAULT_BATCH_SIZE]      = 128
ENV[ENV__TRAIN__DEFAULT_EPOCHS]          = 7
ENV[ENV__TRAIN__DEFAULT_TARGET]          = {S_SCA: 1.0} # Критерий немного жёстче целевого, т.к. на тестовой выборке ошибка может оказаться чуть выше
ENV[ENV__TRAIN__DEFAULT_SAVE_STEP]       = 10
ENV[ENV__TRAIN__DEFAULT_FROM_SCRATCH]    = None

SCORE_FACTOR = -1   # 1 если целевая метрика лучше чем ниже значение, -1 если целевая метрика лучше чем выше значение

# ---------------------------------------------------------------------------- #

# Классический набор данных для обучения

x_train = np.array(train_images[:TRAIN_DATA_LIMIT])
y_train = np.array(train_segments[:TRAIN_DATA_LIMIT])

x_test  = np.array(val_images[:TEST_DATA_LIMIT])
y_test  = np.array(val_segments[:TEST_DATA_LIMIT])

# ---------------------------------------------------------------------------- #

# Список классов. Если не известен - будет создан и выведен

# Список задаётся ручками, т.к. его создание на лету занимает довольно много вермени
# (заполняется по результатам первого прогона с неизвестным списком)
original_classes = np.array([
    [  0,   0,   0],
    [  0,   0, 100],
    [  0,   0, 200],
    [  0, 100,   0],
    [  0, 100, 100],
    [  0, 100, 200],
    [  0, 200,   0],
    [  0, 200, 100],
    [  0, 200, 200],
    [100,   0,   0],
    [100,   0, 100],
    [100,   0, 200],
    [100, 100,   0],
    [100, 100, 100],
    [100, 100, 200],
    [100, 200,   0],
    [100, 200, 100],
    [100, 200, 200],
    [200,   0,   0],
    [200,   0, 100],
    [200,   0, 200],
    [200, 100,   0],
    [200, 100, 100],
    [200, 100, 200],
    [200, 200,   0],
    [200, 200, 100],
    [200, 200, 200],
])

y_train_flat = y_train.reshape(mult(y_train.shape[:-1]), y_train.shape[-1])

if RELOAD_DATA is True or original_classes is None:
    original_classes = np.unique(y_train_flat, axis=0)

original_classes


# ---------------------------------------------------------------------------- #

# Количество найденных классов - 27 вместо обещанных в задании 16
# - есть вопросики к заданию или данным...

# ---------------------------------------------------------------------------- #

# Вычисление количества пикселов по всем классам в y_train

pix_count = np.array([1.31066348e+08, 1.09430737e+08, 9.53483310e+07, 1.80173444e+08,
       1.58537833e+08, 1.44455427e+08, 1.16336855e+08, 9.47012440e+07,
       8.06188380e+07, 9.05240520e+07, 6.88884410e+07, 5.48060350e+07,
       1.39631148e+08, 1.17995537e+08, 1.03913131e+08, 7.57945590e+07,
       5.41589480e+07, 4.00765420e+07, 8.15520250e+07, 5.99164140e+07,
       4.58340080e+07, 1.30659121e+08, 1.09023510e+08, 9.49411040e+07,
       6.68225320e+07, 4.51869210e+07, 3.11045150e+07])

# Список задаётся ручками, т.к. его создание на лету занимает довольно много вермени
# (заполняется по результатам первого прогона с неизвестным списком)

if RELOAD_DATA is True or pix_count is None:
    pix_count = np.zeros(len(original_classes))
    for i in range(len(original_classes)):
        pix_count[i] = (y_train_flat == original_classes[i]).sum()


# ---------------------------------------------------------------------------- #

# Вывод гистограммы наполненности классов для контроля последующих сокращений

if SHOW_PLOTS:
    plt.figure(figsize = (10, 5))
    plt.bar(np.arange(len(pix_count)), pix_count)
    plt.show()

# ---------------------------------------------------------------------------- #

# Сортировка и сокращение числа классов, группируя вместе те, кол-во которых меньше всего
sorted_classes = sorted(list(range(len(original_classes))), key=lambda x: pix_count[x], reverse=True)
sorted_classes

# ---------------------------------------------------------------------------- #

classes_map = {i: [original_classes[sorted_classes[i]]] for i in range(TARGET_CLASSES_COUNT-1)}
pix_count_alt = [pix_count[sorted_classes[i]] for i in range(TARGET_CLASSES_COUNT-1)]
classes_map[TARGET_CLASSES_COUNT-1] = []
pix_count_alt.append(0)
for i in range(TARGET_CLASSES_COUNT-1, len(sorted_classes)):
    classes_map[TARGET_CLASSES_COUNT-1].append(original_classes[sorted_classes[i]])
    pix_count_alt[-1] += pix_count[sorted_classes[i]]

classes_map

# ---------------------------------------------------------------------------- #

# Контроль что ничего не потеряли и не прибавили лишнего
assert sum(pix_count) == sum(pix_count_alt), "Something went wrong!"

# ---------------------------------------------------------------------------- #

# Вывод гистограммы наполненности классов для контроля сокращений

if SHOW_PLOTS:
    plt.figure(figsize = (10, 5))
    plt.bar(np.arange(len(pix_count_alt)), pix_count_alt)
    plt.show()

# ---------------------------------------------------------------------------- #

# Результат вышел плохо сбалансированным - сделаем иначе:
# Раскидаем оставшиеся классы примерно поровну среди классов
# т.к. никаких критериев объединения классов в задании не было,
# а такой способ позволит улучшить результат в итоговых метриках
if TARGET_CLASSES_BALANCE:
    classes_map[TARGET_CLASSES_COUNT-1] = []
    pix_count_alt[TARGET_CLASSES_COUNT-1] = 0

    balanced_spread_map = [[]]*(TARGET_CLASSES_COUNT)
    balanced_spread_count = np.array(pix_count_alt)

    for i in range(TARGET_CLASSES_COUNT-1, len(sorted_classes)):
        idx = np.argmin(balanced_spread_count)
        balanced_spread_map[idx].append(original_classes[sorted_classes[i]])
        balanced_spread_count[idx] += pix_count[sorted_classes[i]]

    for i in range(TARGET_CLASSES_COUNT):
        classes_map[i] += balanced_spread_map[i]
        pix_count_alt[i] = balanced_spread_count[i]

classes_map

# ---------------------------------------------------------------------------- #

# Контроль что ничего не потеряли и не прибавили лишнего
assert sum(pix_count) == sum(pix_count_alt), "Something went wrong!"

# ---------------------------------------------------------------------------- #

# Вывод гистограммы наполненности классов для контроля сокращений

if SHOW_PLOTS:
    plt.figure(figsize = (10, 5))
    plt.bar(np.arange(len(pix_count_alt)), pix_count_alt)
    plt.show()

# ---------------------------------------------------------------------------- #

# Пробное преобразование y_train для использования sparsed classification

PREVIEW_COUNT    = 5
preview_x_train_rgb         = train_images[:PREVIEW_COUNT]
preview_y_train_rgb         = train_segments[:PREVIEW_COUNT]
preview_y_tain_peek_full    = rgb_to_sparse_classes(train_segments[:PREVIEW_COUNT], {k:v for k,v in enumerate(original_classes)})
preview_y_tain_peek_5       = rgb_to_sparse_classes(train_segments[:PREVIEW_COUNT], classes_map)

preview_imgs = preview_x_train_rgb + preview_y_train_rgb + preview_y_tain_peek_full + preview_y_tain_peek_5
preview_cmap = [None]*PREVIEW_COUNT + [None]*PREVIEW_COUNT + [S_GRAY]*PREVIEW_COUNT + [S_GRAY]*PREVIEW_COUNT

if SHOW_PLOTS:
    plt.figure(figsize=(14, 7))
    plot_images(preview_imgs, 4, 5, ordering=S_ROWS, cmap=preview_cmap)
    plt.show()

del preview_x_train_rgb
del preview_y_train_rgb
del preview_y_tain_peek_full
del preview_y_tain_peek_5
del preview_imgs
del preview_cmap

# ---------------------------------------------------------------------------- #

## Модели

###
# Используемые модели

def unet_conv_down(idx, conv_count, last=False):

    result = []

    name = {}
    for i in range(conv_count):
        if i == conv_count-1:
            name = to_dict(_name_=f"level{idx}_cdn_out")

        result += [
            layer_template(Conv2D,          f"$level{idx}_layers",      f"$level{idx}_kernel", padding=S_SAME, name=f"level{idx}_cdn{i+1}"),
            layer_template(BatchNormalization),
            layer_template(Activation,      f"$level{idx}_activation", **name),
        ]

    if not last:
        result += [
            layer_template(Conv2D,          f"$level{idx}_layers", (1, 1), padding='same', _name_=f"level{idx}_cdn_out_mask", _spinoff_=True),
            layer_template(MaxPooling2D, ),
        ]
    else:
        result += [
            layer_template(MaxPooling2D,    _name_=f"for_pretrained_weight", _spinoff_=True),
        ]

    return result

def unet_conv_up(idx, conv_count):

    result = [
        layer_template(Conv2DTranspose, f"$level{idx}_layers",      (2, 2), strides=(2, 2), padding=S_SAME),
        layer_template(BatchNormalization),
        layer_template(Activation,      f"$level{idx}_activation",  _name_=f"level{idx}_cup_in"),
        layer_template(concatenate,     [f"$level{idx}_cup_in",     f"$level{idx}_cdn_out", f"$level{idx}_cdn_out_mask"], _parent_=None),
    ]
    for i in range(conv_count):
        result += [
            layer_template(Conv2D,      f"$level{idx}_layers",      f"$level{idx}_kernel", padding=S_SAME, name=f"level{idx}_cup{i+1}"),
            layer_template(BatchNormalization),
            layer_template(Activation,  f"$level{idx}_activation"),
        ]

    return result



input_shape  = x_train.shape[1:]
model_input  = [layer_template(Input,   input_shape),]
model_output = [layer_template(Conv2D, "$classes_count", "$output_kernel", activation=S_SOFTMAX, padding=S_SAME),]

handmade_models = to_dict(
    unet = to_dict(
        model_class = Model,
        vars = to_dict(
            classes_count = len(classes_map),
        ),
        thd_kwargs = to_dict(
        ),
        template = [
                *model_input,
                *unet_conv_down(1,2),
                *unet_conv_down(2,2),
                *unet_conv_down(3,3),
                *unet_conv_down(4,3),
                *unet_conv_down(5,3, last=True),
                *unet_conv_up(4,2),
                *unet_conv_up(3,2),
                *unet_conv_up(2,2),
                *unet_conv_up(1,2),
                *model_output,
        ],
    ),
)

generated_models = {}

models = {}
for models_candidates in (handmade_models, generated_models):
    for k, v in models_candidates.items():
        if not use_model(k):
            continue
        models[k] = v

# ---------------------------------------------------------------------------- #

## Гиперпараметры

###
# Гиперпараметры

hp_defaults = to_dict(
        tabs=['learn', 'XvsY'],
)

data_common_vars=to_dict(
    # shuffle         = False,
)

train_common_vars=to_dict(
    # batch_size      = 128,
)


hp_template = to_dict(
    **hp_defaults,
    model_vars=to_dict(
        level1_layers = 64,
        level1_kernel = (3,3),
        level1_activation = S_RELU,

        level2_layers = 128,
        level2_kernel = (3,3),
        level2_activation = S_RELU,

        level3_layers = 256,
        level3_kernel = (3,3),
        level3_activation = S_RELU,

        level4_layers = 512,
        level4_kernel = (3,3),
        level4_activation = S_RELU,

        level5_layers = 512,
        level5_kernel = (3,3),
        level5_activation = S_RELU,

        output_kernel = (3,3),
    ),
    data_vars={
        **data_common_vars,
    },
    train_vars={
        **train_common_vars,
    },
)


hyper_params_sets = {}

xx = 1000
for hp_layers_mult in (1, 0.5):
    for hp_act in (S_RELU, S_ELU, S_LINEAR, ):
        for hp_kernel in ((3,3), (5, 5) ):
            if xx <= 0:
                continue
            xx -= 1
            hp =copy.deepcopy(hp_template)
            hp_name = f"{hp_act}-{hp_kernel[0]}x{hp_kernel[1]}-{hp_layers_mult}"
            for k,v in hp['model_vars'].items():
                if "activation" in k:
                    v = hp_act
                elif "kernel" in k:
                    v = hp_kernel
                elif "layers" in k:
                    v = int(v * hp_layers_mult)
            hyper_params_sets[hp_name] = hp

# ---------------------------------------------------------------------------- #

# Полное преобразование y_train / y_test для использования sparsed classification

y_train = np.array(rgb_to_sparse_classes(train_segments[:TRAIN_DATA_LIMIT], classes_map))
y_test  = np.array(rgb_to_sparse_classes(val_segments[:TEST_DATA_LIMIT], classes_map))


# ---------------------------------------------------------------------------- #

## Вспомогательный код для вывода данных

###
# Создать вкладки для вывода результатов

from IPython.display import display

# def get_tab_run(tab_id, model_name, model_data, hp_name, hp):
#     run_name = f"{model_name}--{hp_name}"
#     tab_name = run_name
#     return tab_id, tab_name, run_name

tabs_dict, tabs_objs = bp.make_tabs(
    models,
    hyper_params_sets,
    # get_tab_run_call=get_tab_run
)

tabs_objs.keys()

# ---------------------------------------------------------------------------- #

# Результаты обучения

## Базовая информация по моделям и итогам обучения

# display(tabs_objs["learn"])


# ---------------------------------------------------------------------------- #

# Код для обучения

# ---------------------------------------------------------------------------- #

## Подготовка данных, обучение, вывод результатов

###
# Подготовка данных, обучение, вывод результатов

def prepare(    model_name,
    model_data,
    hp_name,
    hp,
):
    model_vars = hp['model_vars']
    data_vars = hp['data_vars']
    train_vars = hp['train_vars']

    # original_tabs = [*hp['tabs']]
    # hp['tabs'] = []
    # for v in original_tabs:
    #     tab_id, _, _ = get_tab_run(v, model_name, model_data, hp_name, hp)
    #     hp['tabs'].append(tab_id)

    data_provider = TrainDataProvider(
        x_train = x_train,
        y_train = y_train,

        x_val   = 0.1,
        y_val   = None,

        x_test  = x_test,
        y_test  = y_test,
    )

    return data_provider


score = {}

def plot_XvsY(
    model_name,
    model_data,
    hp_name,
    hp,
    run_name,
    mhd,
    thd,
    tab_group,
    tab_name,
    last_metrics,
    best_metrics,
):
    y_pred = best_metrics['pred']
    if y_pred is None:
        print("No best metrics, trying to use last metrics")
        y_pred = last_metrics['pred']
    if y_pred is None:
        print("No last metrics, noting more to do...")
        return

    SAMPLES_COUNT = 5
    imgs = [img.squeeze() for img in pick_random_pairs([x_test, y_test, y_pred], SAMPLES_COUNT)]
    cmaps = [None, S_GRAY, S_GRAY] * SAMPLES_COUNT
    plt.figure(figsize=(14, 7))
    plot_images(imgs, 3, SAMPLES_COUNT, ordering=S_COLS)
    plt.show()

def print_to_tab(
    model_name,
    model_data,
    hp_name,
    hp,
    run_name,
    mhd,
    thd,
    tab_group,
    tab_name,
    last_metrics,
    best_metrics,
):
    tab_print_map = {}
    for tab_id in hp['tabs']:
        if 'learn' in tab_id:
            tab_print_map[tab_id] = bp.print_to_tab_learn
        elif 'XvsY' in tab_id:
            tab_print_map[tab_id] = plot_XvsY

    if 'learn' in tab_group:
        train_metric = mhd.context.get_metric_value(S_SCA)

        y_pred = best_metrics['pred']
        if y_pred is None:
            print("No best metrics, trying to use last metrics")
            y_pred = last_metrics['pred']
        if y_pred is None:
            print("No last metrics, can't get test_mse...")
            test_metric = 0
        else:
            test_metric = train_metric  # TODO:

        success = True # TODO: define metric for success if necessary
        score[run_name] = to_dict(
            best_epoch = best_metrics['epoch'],
            test_metric = test_metric,
            train_metric = train_metric,
            score = SCORE_FACTOR * (best_metrics['epoch'] + train_metric),
            success = ["No", "Yes"][success],
        )
    else:
        train_metric = False
        test_metric = False

    print_call = tab_print_map.get(tab_group, None)
    if print_call is None:
        bp.print_to_tab_fallback(
            model_name,
            model_data,
            hp_name,
            hp,
            run_name,
            mhd,
            thd,
            tab_group,
            tab_name,
            last_metrics,
            best_metrics,
        )
        return

    print(f"Модель         : {hp['model']}")
    print(f"Гиперпараметры : {hp_name}")
    print(f"Последняя эпоха: {last_metrics['epoch']}")
    print(f"Лучшая эпоха   : {best_metrics['epoch']}")
    if test_metric is not False:
        print(f"Метрика на тест.выб: {test_metric}")
    print("")
    print_call(
        model_name,
        model_data,
        hp_name,
        hp,
        run_name,
        mhd,
        thd,
        tab_group,
        tab_name,
        last_metrics,
        best_metrics,
    )

score.clear()

bp.train_routine(
    models,
    hyper_params_sets,
    tabs_dict,
    preparation_call=prepare,
    # get_tab_run_call=get_tab_run,
    print_to_tab_call=print_to_tab,
)

score_table = dict_to_table(score, dict_key='model', sort_key=lambda x: [
    SCORE_FACTOR * (score[x]['test_metric'] + ENV[ENV__TRAIN__DEFAULT_EPOCHS]),
    SCORE_FACTOR * (score[x]['test_metric'] + score[x]['best_epoch'])
    ][score[x]['success']=="Yes"])

print_table(*score_table)
