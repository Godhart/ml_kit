# Этот пример в google_colab можно посмотреть по ссылке https://colab.research.google.com/drive/1bwMt0EctCyVZ-QUZAMRyt92Y4XOfmPxK

## Задание

# Добейтесь на автокодировщике с 2-мерным скрытым пространством на 3-х цифрах: 0, 1 и 3 – ошибки MSE**<0.034** на скорости обучения **0.001** на **10-й эпохе**.

## Импорт библиотек


# Работа с операционной системой
import os

# Отрисовка графиков
import matplotlib.pyplot as plt

# Операции с путями
import glob

# Работа с массивами данных
import numpy as np

# Слои
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Conv2DTranspose, concatenate, Activation, MaxPooling2D, Conv2D, BatchNormalization, Concatenate

# Модель
from tensorflow.keras import Model

# Датасет
from tensorflow.keras.datasets import mnist

# Оптимизатор для обучения модели
from tensorflow.keras.optimizers import Adam

# Коллбэки для выдачи информации в процессе обучения
from tensorflow.keras.callbacks import LambdaCallback

# %matplotlib inline

## Утилиты

EPOCH_CALLBACK_DATA = {
    'encoder'   : None,
    'x_data'    : None,
    'y_data'    : None
}

# Функция-коллбэк. Отрисовывает объекты в скрытом пространстве

def ae_on_epoch_end(epoch, logs):
    print('________________________')
    print(f'*** Epoch: {epoch+1}, loss: {logs["loss"]} ***')
    print('________________________')

    # Получение картинки латентного пространства в конце эпохи и запись в файл
    # Задание числа пикселей на дюйм
    plt.figure(dpi=100)

    # Предсказание енкодера на тренировочной выборке
    predict = EPOCH_CALLBACK_DATA['encoder'].predict(EPOCH_CALLBACK_DATA['x_data'])

    # Создание рисунка: множество точек на плоскости 3-х цветов (3-х классов)
    scatter = plt.scatter(predict[:,0,],predict[:,1], c=EPOCH_CALLBACK_DATA['y_data'], alpha=0.6, s=5)

    # Создание легенды
    legend2 = plt.legend(*scatter.legend_elements(), loc='upper right', title='Классы')

    # Сохранение картинки с названием, которого еще нет
    paths = glob.glob('*.jpg')
    plt.savefig(f'image_{str(len(paths))}.jpg')

    # Отображение. Без него рисунок не отрисуется
    plt.show()


ae_callback = LambdaCallback(on_epoch_end=ae_on_epoch_end)

# Удаление изображений. Применять при обучении новой модели, чтобы не было путаницы в картинках.

def clean():
  # Получение названий всех картинок
  paths = glob.glob('*.jpg')

  # Удаление всех картинок по полученным путям
  for p in paths:
    os.remove(p)

# Удаление всех картинок
# clean()

## Загрузка данных

# Загрузка датасета
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Нормировка
X_train = X_train.astype('float32')/255.
X_train = X_train.reshape(-1, 28, 28, 1)

# Выбор визуализируемых классов (цифр) и формирование подвыборок для них по маске
numbers = [0, 1, 3]
mask = np.array([(i in numbers) for i in y_train])
X_train = X_train[mask]
y_train = y_train[mask]


# Нормировка тестовых данных
X_test  = X_test.astype('float32')/255.
X_test  = X_test.reshape(-1, 28, 28, 1)

# Выбор визуализируемых классов (цифр) и формирование подвыборок для них по маске
mask_test = np.array([(i in numbers) for i in y_test])
X_test  = X_test[mask_test]
y_test  = y_test[mask_test]


## Создание модели и обучение

# ---------------------------------------------------------------------------- #

# Ваше решение

# Создать шаблон модели используя модель из практики, заложить вариации параметров.
# Начать с типовых параметров, затем пробовать генерировать варианты параметров рандомно в опредлённом диапазоне
# (начинать с фиксированнго сида, нагенерировать кучу вариантов параметров)
# Обучать на N вариантах или пока цель не будет достигнута
# Если не получится:
#   - проанализировать что могло идти не так, насколько далеки от цели варианты и если достаточно близко - попробовать подобрать параметры вручную опираясь на лучшие итерации
#   - попробовать другой шаблон модели и повторить с ним действия описанные выше

# ---------------------------------------------------------------------------- #

EPOCH_CALLBACK_DATA['x_data'] = X_train
EPOCH_CALLBACK_DATA['y_data'] = y_train

x_train = X_train
x_test = X_test

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

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
    from ml_kit.trainer_series import *

# ---------------------------------------------------------------------------- #

# Отрисовка модели
from tensorflow.keras import utils

# ---------------------------------------------------------------------------- #

## Глобальные параметры

# Списки включения / исключения наборов гиперпараметров

TRAIN_INCLUDE = None  # Включать всё
TRAIN_EXCLUDE = None  # Ничего не исключать

# TRAIN_INCLUDE = []
# TRAIN_EXCLUDE = []

ENV[ENV__DEBUG_PRINT] = True

# Подключение google диска если работа в ноутбуке
# connect_gdrive()

# ---------------------------------------------------------------------------- #

# Перегрузка окружения под текущую задачу
ENV[ENV__TRAIN__DEFAULT_DATA_PATH]       = "lesson_10_lite_1a"
ENV[ENV__TRAIN__DEFAULT_OPTIMIZER]       = [Adam, [], to_dict(learning_rate=1e-3)]
ENV[ENV__TRAIN__DEFAULT_LOSS]            = S_MSE
ENV[ENV__TRAIN__DEFAULT_METRICS]         = [S_MSE]
ENV[ENV__TRAIN__DEFAULT_BATCH_SIZE]      = 128
ENV[ENV__TRAIN__DEFAULT_EPOCHS]          = 10
ENV[ENV__TRAIN__DEFAULT_TARGET]          = {S_MSE: 0.034}
ENV[ENV__TRAIN__DEFAULT_SAVE_STEP]       = 10
ENV[ENV__TRAIN__DEFAULT_FROM_SCRATCH]    = None

# ---------------------------------------------------------------------------- #

# Основные слои
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, Conv2D, LSTM, GlobalMaxPooling1D, MaxPooling1D, RepeatVector

# ---------------------------------------------------------------------------- #

## Модели

###
# Используемые модели

models = to_dict(

    dns1 = to_dict(
        model_class = Model,
        template = to_dict(
            branch_general = to_dict(
                input  = True,
                output = True,
                layers = [
                    layer_template(Input,   "$input_shape"),
                    layer_template(Flatten, ),
                    layer_template(Dense,   2,  activation='relu', _name_='encoder'),
                    layer_template(Dense,   "$input_shape_flat", activation='sigmoid'),
                    layer_template(Reshape, "$input_shape"),
                ],
            ),
        ),
    ),

)

# ---------------------------------------------------------------------------- #

## Гиперпараметры

###
# Гиперпараметры

hp_defaults = to_dict(
        tabs=['learn', 'lanim'],
)

data_common_vars=to_dict(
    shuffle         = False,
    batch_size      = 128,
)

def mult(vector):
    result = 1
    for v in vector:
        result *= v
    return result

hp_template = to_dict(
    **hp_defaults,
    model_vars=to_dict(
        input_shape         = x_train.shape[1:],
        input_shape_flat    = mult(x_train.shape[1:]),
        output_drop_rate    = 0.5
    ),
    data_vars={
        **data_common_vars,
    },
)


hyper_params_sets = {}

if True:
    hp =copy.deepcopy(hp_template)
    hyper_params_sets[f"def"] = hp

# ---------------------------------------------------------------------------- #

## Вспомогательный код для вывода данных

###
# Создать вкладки для вывода результатов

def get_tab(tab_id, model_name, hp_name, hp):
    return tab_id, f"{model_name}--{hp_name}"

from IPython.display import clear_output, display
import ipywidgets as widgets
from functools import lru_cache

model_tabs = {}
tabs_dict = {}
for model_name in models:
    for hp_name, hp in hyper_params_sets.items():
        for tab_id in hp['tabs']:
            tab_group, tab_i = get_tab(tab_id, model_name, hp_name, hp)
            if tab_group not in tabs_dict:
                tabs_dict[tab_group] = {}
            widget = tabs_dict[tab_group][tab_i] = widgets.Output()
            with widget:
                # По умолчанию заполнить текст вкладок информацией о параметрах модели
                clear_output()
                print(f"{model_name}--{hp_name}")

tabs_objs = {k: widgets.Tab() for k in tabs_dict}
for k, v in tabs_dict.items():
    tab_items_keys = list(sorted(v.keys()))
    tabs_objs[k].children = [v[kk] for kk in tab_items_keys]
    for i in range(0, len(tab_items_keys)):
        tabs_objs[k].set_title(i, f"{k}:{tab_items_keys[i]}")

tabs_objs.keys()
#display(tabs_objs["xxx"])

# ---------------------------------------------------------------------------- #

# Результаты обучения

## Базовая информация по моделям и итогам обучения

display(tabs_objs["learn"])


# ---------------------------------------------------------------------------- #

# Код для обучения

# ---------------------------------------------------------------------------- #

## Подготовка данных, обучение, вывод результатов

###
# Подготовка данных, обучение, вывод результатов

dummy_output = widgets.Output()

for model_name in models:
    for hp_name in hyper_params_sets:
        def main_logic(model_name, hp_name):

            hp = copy.deepcopy(hyper_params_sets[hp_name])
            if 'model' not in hp:
                hp['model'] = model_name
            else:
                raise ValueError("'model' should be omitted in hyper params for this very case!")

            model_data = models[model_name]

            run_name = f"{model_name}--{hp_name}"
            print(f"Running {run_name}")

            data_vars = hp['data_vars']

            model_vars = copy.deepcopy(hp['model_vars'])

            data_provider = TrainDataProvider(
                x_train = x_train,
                y_train = y_train,
                x_val   = x_test,
                y_val   = y_test,
                x_test  = x_test,
                y_test  = y_test,
            )

            mhd = ModelHandler(
                name            = run_name,
                model_class     = model_data['model_class'],
                optimizer       = model_data.get('optimizer', ENV[ENV__TRAIN__DEFAULT_OPTIMIZER]),
                loss            = model_data.get('loss',      ENV[ENV__TRAIN__DEFAULT_LOSS]),
                metrics         = model_data.get('metrics',   hp.get('metrics', ENV[ENV__TRAIN__DEFAULT_METRICS])),
                model_template  = model_data['template'],
                model_variables = model_vars,
                batch_size      = model_data.get('batch_size',ENV[ENV__TRAIN__DEFAULT_BATCH_SIZE]),
                data_provider   = data_provider,
                load_weights_only = True,   # NOTE: True since named layers are used
            )

            def on_model_update(thd):
                EPOCH_CALLBACK_DATA['encoder'] = mhd.named_layers['branch_general/encoder']

            thd = TrainHandler(
                data_path       = ENV[ENV__MODEL__DATA_ROOT] / model_data.get("data_path", ENV[ENV__TRAIN__DEFAULT_DATA_PATH]),
                data_name       = run_name,
                mhd             = mhd,
                mhd_class       = ModelHandler,
                on_model_update = on_model_update,
            )

            def display_callback(*args, **kwargs):
                pass

            thd.train(
                from_scratch    = model_data.get("from_scratch", ENV[ENV__TRAIN__DEFAULT_FROM_SCRATCH]),
                epochs          = model_data.get("epochs", ENV[ENV__TRAIN__DEFAULT_EPOCHS]),
                target          = model_data.get("target", ENV[ENV__TRAIN__DEFAULT_TARGET]),
                save_step       = model_data.get("save_step", ENV[ENV__TRAIN__DEFAULT_SAVE_STEP]),
                display_callback= display_callback,
            )

            # Вывод результатов для сравнения
            full_history = copy.deepcopy(mhd.context.history)

            epoch_last = mhd.context.epoch
            mhd.update_data(force=True)
            pred_last = mhd.context.test_pred

            thd.load_best()
            epoch_best = mhd.context.epoch
            mhd.update_data(force=True)
            pred_best = mhd.context.test_pred

            mhd.context.report_history = full_history

            with dummy_output:
                plt.show()
                clear_output()

            for tab_id in hp['tabs']:
                tab_group, tab_i = get_tab(tab_id, model_name, hp_name, hp)
                with tabs_dict[tab_group][tab_i]:
                    clear_output()
                    print(f"Модель         : {hp['model']}")
                    print(f"Гиперпараметры : {hp_name}")
                    print(f"Последняя эпоха: {epoch_last}")
                    print(f"Лучшая эпоха   : {epoch_best}")
                    print("")
                    if "learn" in tab_id:
                        mhd.context.report_to_screen()
                        mhd.model.summary()

                        utils.plot_model(mhd.model, dpi=60)
                        plt.show()
                    elif "lanim" in tab_id:
                        pass

                with dummy_output:
                    plt.show()
                    clear_output()

            mhd.context.report_history = None
            mhd.unload_model()

        main_logic(model_name, hp_name)