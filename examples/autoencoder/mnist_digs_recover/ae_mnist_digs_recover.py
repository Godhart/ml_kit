# Этот пример в google_colab можно посмотреть по ссылке https://colab.research.google.com/drive/1mjrcZ70kU9vq8ojJW4sdfewgk2r_5X7i

## Задание

# Создайте автокодировщик, удаляющий черные квадраты в случайных областях изображений.

# Алгоритм действий:
# 1. Возьмите базу картинок Mnist.
# 2. На картинках в случайных местах сделайте чёрные квадраты размера 8 на 8.
# 3. Создайте и обучите автокодировщик восстанавливать оригинальные изображения из "зашумленных" квадратом изображений.
# 4. Добейтесь MSE < 0.0070 на тестовой выборке

## Импорт библиотек

# Отображение
import matplotlib.pyplot as plt

# Для работы с тензорами
import numpy as np

# Класс создания модели
from tensorflow.keras.models import Model

# Для загрузки данных
from tensorflow.keras.datasets import mnist

# Необходимые слои
from tensorflow.keras.layers import Input, Conv2DTranspose, MaxPooling2D, Conv2D, BatchNormalization

# Оптимизатор
from tensorflow.keras.optimizers import Adam

## Данные

# Загрузка данных
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Нормировка данных
X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.

# Изменение формы под удобную для Keras
X_train = X_train.reshape((-1, 28, 28, 1))
X_test = X_test.reshape((-1, 28, 28, 1))

# Ваше решение

# Создать набор данных с "чёрными квадратами", который будет использоваться в качестве входного для модели
# Создать шаблон модели используя модель из практики, заложить вариации параметров.
# Начать с типовых параметров, затем пробовать генерировать варианты параметров рандомно в опредлённом диапазоне
# (начинать с фиксированнго сида, нагенерировать кучу вариантов параметров)
# Обучать на N вариантах или пока цель не будет достигнута
# Если не получится:
#   - проанализировать что могло идти не так, насколько далеки от цели варианты и если достаточно близко - попробовать подобрать параметры вручную опираясь на лучшие итерации
#   - попробовать другой шаблон модели и повторить с ним действия описанные выше

## Импорт библиотек


# Работа с операционной системой
import os

# Слои
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Conv2DTranspose, concatenate, Activation, MaxPooling2D, Conv2D, BatchNormalization, Concatenate

# Функция среднеквадратической ошибки для расчетов вручную
from sklearn.metrics import mean_squared_error

# %matplotlib inline

## Подготовительные операции

# Приведение имён переменных к стандартным именам
x_train = X_train
x_test = X_test

del X_train
del X_test

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

## Формирование "испорченных" данных
x_train_bbox = np.copy(x_train)
x_test_bbox = np.copy(x_test)
for data in (x_train_bbox, x_test_bbox):
    for i in range(data.shape[0]):
        w = 8
        h = 8
        x = np.random.randint(0, data.shape[1]-w)
        y = np.random.randint(0, data.shape[2]-h)
        data[i, x:x+w, y:y+h, :] = 0.

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
    import ml_kit.boilerplate as bp
    from ml_kit.reporting import *

# ---------------------------------------------------------------------------- #

## Глобальные параметры

# Списки включения / исключения наборов гиперпараметров

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
ENV[ENV__TRAIN__DEFAULT_DATA_PATH]       = "lesson_10_ultrapro_1a"
ENV[ENV__TRAIN__DEFAULT_OPTIMIZER]       = [Adam, [], to_dict(learning_rate=1e-4)]
ENV[ENV__TRAIN__DEFAULT_LOSS]            = S_MSE
ENV[ENV__TRAIN__DEFAULT_METRICS]         = [S_MSE]
ENV[ENV__TRAIN__DEFAULT_BATCH_SIZE]      = 128
ENV[ENV__TRAIN__DEFAULT_EPOCHS]          = 50
ENV[ENV__TRAIN__DEFAULT_TARGET]          = {S_MSE: 0.0068} # Критерий немного жёстче целевого, т.к. на тестовой выборке ошибка может оказаться чуть выше
ENV[ENV__TRAIN__DEFAULT_SAVE_STEP]       = 10
ENV[ENV__TRAIN__DEFAULT_FROM_SCRATCH]    = None

# ---------------------------------------------------------------------------- #

# Основные слои
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, Conv2D, LSTM, GlobalMaxPooling1D, MaxPooling1D, RepeatVector

# ---------------------------------------------------------------------------- #

###
# Ешё немного служебных функций

def prod(*args):
    result = 1
    for v in args:
        result *= v
    return result

# ---------------------------------------------------------------------------- #

## Модели

###
# Используемые модели

input_shape = x_train.shape[1:]

model_items = to_dict(
    cnn_input    = [],
    cnn1_encoder = [
                    layer_template(Conv2D,  32, (3, 3), padding='same', activation='$ae_activation'),
                    layer_template(BatchNormalization, ),
                    layer_template(Conv2D,  32, (3, 3), padding='same', activation='$ae_activation'),
                    layer_template(BatchNormalization, ),
                    layer_template(MaxPooling2D, ),
    ],
    cnn1_decoder = [
                    layer_template(Conv2DTranspose, 32, (2, 2), strides=(2, 2), padding='same', activation='$ae_activation'),
                    layer_template(Conv2D,  32, (3, 3), padding='same', activation='$ae_activation'),
                    layer_template(BatchNormalization, ),
                    layer_template(Conv2D,  32, (3, 3), padding='same', activation='$ae_activation'),
                    layer_template(BatchNormalization, ),
    ],
    cnn2_encoder = [
                    layer_template(Conv2D,  64, (3, 3), padding='same', activation='$ae_activation'),
                    layer_template(BatchNormalization, ),
                    layer_template(Conv2D,  64, (3, 3), padding='same', activation='$ae_activation'),
                    layer_template(BatchNormalization, ),
                    layer_template(MaxPooling2D, ),
    ],
    cnn2_decoder = [
                    layer_template(Conv2DTranspose, 32, (2, 2), strides=(2, 2), padding='same', activation='$ae_activation'),
                    layer_template(Conv2D,  32, (3, 3), padding='same', activation='$ae_activation'),
                    layer_template(BatchNormalization, ),
                    layer_template(Conv2D,  32, (3, 3), padding='same', activation='$ae_activation'),
                    layer_template(BatchNormalization, ),
    ],
    cnn_output = [
                    layer_template(Conv2D,  input_shape[-1], (3, 3), activation='sigmoid', padding='same'),
    ]
)

handmade_models = to_dict(

    cnn2 = to_dict(
        model_class = Model,
        vars = to_dict(
            ae_activation = 'relu'
        ),
        thd_kwargs = to_dict(
        ),
        template = to_dict(
            encoder = to_dict(
                input  = True,
                layers = [
                    layer_template(Input,   input_shape),
                    *model_items['cnn1_encoder'],
                    *model_items['cnn2_encoder'],
                ]
            ),
            decoder = to_dict(
                parents = 'encoder',
                output = True,
                layers = [
                    *model_items['cnn2_decoder'],
                    *model_items['cnn1_decoder'],
                    *model_items['cnn_output'],
                ],
            ),
        ),
    ),

)

models_proto = {}

xx = 0
for dn in range(3):
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                if xx <= 0:
                    continue
                idx = tuple([di+1,dj+1,dk+1][:dn+1])
                if idx in models_proto:
                    continue
                models_proto[idx] = None
                xx -= 1

generated_models = {}

for bn in (True, False):
    for ae_act in ('relu', 'sigmoid', 'tanh'): # TODO: also try , 'elu', 'softsign', 'softplus', 'mish'):
        for lat_act in ('relu', 'sigmoid', 'tanh', 'same'):
            if lat_act == 'same':
                lat_act = ae_act
            for k,v in models_proto.items():

                model_name = 'cnn'+''.join(str(ki) for ki in k)
                if bn:
                    model_name += "b"
                model_name += "-" + ae_act + "-" + lat_act

                if not use_model(model_name):
                    continue

                encoder = []
                for ki in k:
                    encoder += model_items[f'{ki}']
                decoder = [*encoder]
                decoder.reverse() # TODO: replace MaxPolling to opposite
                if bn:
                    for arr in (encoder, decoder):
                        tmp = [*arr]
                        arr.clear()
                        for item in tmp:
                            arr.append(item)
                            arr.append(layer_template(BatchNormalization, ))
                k_model = to_dict(
                    model_class = Model,
                    mhd_kwargs = to_dict(
                        save_model = False,
                        save_weights = False,
                    ),
                    thd_kwargs = to_dict(
                    ),
                    vars = to_dict(
                        ae_activation = ae_act,
                    ),
                    template = to_dict(
                        encoder = to_dict(
                            input  = True,
                            layers = [
                                *model_items['cnn_input'],
                                *encoder,
                            ],
                        ),
                        decoder = to_dict(
                            parents = 'encoder',
                            output  = True,
                            layers = [
                                *decoder,
                                *model_items['cnn_output']
                            ],
                        ),
                    ),
                )

                generated_models[model_name] = k_model

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
        output_drop_rate    = 0.3
    ),
    data_vars={
        **data_common_vars,
    },
    train_vars={
        **train_common_vars,
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

from IPython.display import display

def get_tab_run(tab_id, model_name, model_data, hp_name, hp):
    run_name = f"{model_name}--{hp_name}"
    tab_name = run_name
    return tab_id, tab_name, run_name

tabs_dict, tabs_objs = bp.make_tabs(
    models,
    hyper_params_sets,
    get_tab_run_call=get_tab_run
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

    original_tabs = [*hp['tabs']]
    hp['tabs'] = []
    for v in original_tabs:
        tab_id, _, _ = get_tab_run(v, model_name, model_data, hp_name, hp)
        hp['tabs'].append(tab_id)

    data_provider = TrainDataProvider(
        x_train = x_train_bbox,
        y_train = x_train,  # NOTE: x_train is on purpose since it's autoencoder

        x_val   = 0.1,
        y_val   = None,

        x_test  = x_test_bbox,
        y_test  = x_test,   # NOTE: x_test is on purpose since it's autoencoder
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

    imgs = [img.squeeze() for img in pick_random_pairs([x_test_bbox, y_pred], 5)]
    plt.figure(figsize=(14, 7))
    plot_images(imgs, 2, 5, ordering=S_COLS)
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
        train_mse = mhd.context.get_metric_value(S_MSE)

        y_pred = best_metrics['pred']
        if y_pred is None:
            print("No best metrics, trying to use last metrics")
            y_pred = last_metrics['pred']
        if y_pred is None:
            print("No last metrics, can't get test_mse...")
            test_mse = None
        else:
            image_size = x_test.shape[1] * x_test.shape[2]
            test_mse= mean_squared_error(
                x_test.reshape(-1, image_size).T,
                y_pred.reshape(-1, image_size).T,
                multioutput='raw_values'
            ).mean()

        success = test_mse is not None and test_mse < ENV[ENV__TRAIN__DEFAULT_TARGET][S_MSE]
        score[run_name] = to_dict(
            best_epoch = best_metrics['epoch'],
            test_mse = test_mse,
            train_mse = train_mse,
            score = best_metrics['epoch'] + test_mse,
            success = ["No", "Yes"][success],
        )
    else:
        train_mse = False
        test_mse = False

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
    if test_mse is not False:
        print(f"MSE на тест.выб: {test_mse}")
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
    get_tab_run_call=get_tab_run,
    print_to_tab_call=print_to_tab,
)

score_table = dict_to_table(score, dict_key='model', sort_key=lambda x: [
    score[x]['test_mse'] + ENV[ENV__TRAIN__DEFAULT_EPOCHS],
    score[x]['test_mse'] + score[x]['best_epoch']
    ][score[x]['success']=="Yes"])

print_table(*score_table)
