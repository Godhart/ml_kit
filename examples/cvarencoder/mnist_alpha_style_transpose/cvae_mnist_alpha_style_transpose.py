# Этот пример в google_colab можно посмотреть по ссылке https://colab.research.google.com/drive/1BYpocfurl9kF8I2dR4J001FGxWARahI9

# Возьмите из ноутбука по практическому занятия "Автокодировщики" сверточный вариационный энкодер или напишите свой и обучите его на датасете Emnist letters.

# Датасет содержит изображения рукописных латинских букв.

# Размер обучающей выборки 697932 изображений, тестовой - 116323.

# Данный автокодировщик показывает весьма слабые результаты. При воспроизведении букв много "брака". Повысьте качество его работы. Обратите внимание на следущие гиперпараметры:

# Размерность скрытого пространства
# Количество сверточных слоев
# Число эпох обучения
# Добейтесь существенного улучшения качество воспроизведения букв.

# В выводах укажите минимальные значения подбираемых параметров, при которых удается получить желаемое качество воспроизведения символов (пример ниже).

# Подключим Numpy
import numpy as np

# Подключим библиотеку отображения графиков
import matplotlib.pyplot as plt

# %matplotlib inline


# Импортируем Keras
from tensorflow import keras

import tensorflow as tf

# Подключим все необходимые слои Keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Lambda, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Dropout, concatenate, Conv2D, Conv2DTranspose

# Подключим модуль вычислений на Keras
import keras.backend as K

# Подключим датасет рукописных букв
S_DIGS = "DIGS"
S_ALPHAS = "ALPHAS"
DATA = S_DIGS

if DATA == S_DIGS:
    # Для загрузки данных
    from tensorflow.keras.datasets import mnist
elif DATA == S_ALPHAS:
    from emnist import extract_training_samples, extract_test_samples
else:
    raise ValueError(f"'DATA' should be in {(S_DIGS, S_ALPHAS)}")

# Подключим модуль работы с операционной системой
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # От tensorflow будет получать только ошибки

# ---------------------------------------------------------------------------- #

if DATA == S_DIGS:
    (xs_train, y_train), (xs_test, y_test) = mnist.load_data()
elif DATA == S_ALPHAS:
    # Скачаем обучающую выборку
    xs_train, y_train = extract_training_samples('letters')

    # Скачаем тестовую выборку
    xs_test, y_test = extract_test_samples('letters')

# Добавим 1 размерность numpy-массиву обучающей выборки + приведем к диапазону 0...1
x_train = np.reshape(xs_train, (len(xs_train), 28, 28, 1))/255.

# Добавим 1 размерность numpy-массиву тестовой выборки + приведем к диапазону 0...1
x_test = np.reshape(xs_test, (len(xs_test), 28, 28, 1))/255.

# ---------------------------------------------------------------------------- #

# Определим функцию отображения 100 картинок

def showResult(re): # Получим 100 картинок

    total = 10                               # Считаем полное количество выводимых мультяшек +1
    plt.figure(figsize=(total, total))       # Создаем заготовку для финальной картинки 10x10
    num = 1                                  # Счетчик выводимых мультяшек
    for i in range(100):                     # Цикл по картинкам
        ax = plt.subplot(total, total, num)  # Добавим место для графика
        img = re[num-1:num,:,:,:]            # Сформируем очередную картинку
        num += 1                             # Инкремент номера графика
        plt.imshow(img.squeeze())            # Рисуем картинки
        ax.get_xaxis().set_visible(False)    # Спрячем ось X
        ax.get_yaxis().set_visible(False)    # Спрячем ось Y

# ---------------------------------------------------------------------------- #

print (x_train.shape) # Выведем размерность обучающей выборки изображений
print (x_test.shape)  # Выведем размерность тестовой выборки изображений
print (y_train.shape) # Выведем размерность обучающей выборки меток
print (y_test.shape)  # Выведем размерность тестовой выборки меток

# ---------------------------------------------------------------------------- #

showResult(x_train[:100,:,:,:])

# ---------------------------------------------------------------------------- #

# Взглянем на первые 100 меток обучающей выборки

y_train[:100]

# ---------------------------------------------------------------------------- #

# Ваше решение

# Создать исходную модель, обучить, посмотреть на результат
# Попробовать изменить гиперпараметры
# Попробовать сделать что-то с моделью если изменение гиперпараметров не даёт желаемого результата

# ---------------------------------------------------------------------------- #

## Импорт библиотек

# Слои
from tensorflow.keras.layers import Dense, Flatten, Reshape, concatenate, Activation, MaxPooling2D, Concatenate

# Класс создания модели
from tensorflow.keras.models import Model

# Оптимизатор
from tensorflow.keras.optimizers import Adam

# Функция среднеквадратической ошибки для расчетов вручную
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------- #

## Подготовительные операции

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
ENV[ENV__TRAIN__DEFAULT_DATA_PATH]       = "lesson_14_pro_1a"
ENV[ENV__TRAIN__DEFAULT_OPTIMIZER]       = [Adam, [], to_dict(learning_rate=1e-4)] # TODO: change learning rate ?
ENV[ENV__TRAIN__DEFAULT_LOSS]            = S_MSE            # TODO: None
ENV[ENV__TRAIN__DEFAULT_METRICS]         = [S_MSE]
ENV[ENV__TRAIN__DEFAULT_BATCH_SIZE]      = 128
ENV[ENV__TRAIN__DEFAULT_EPOCHS]          = 2
ENV[ENV__TRAIN__DEFAULT_TARGET]          = {S_MSE: 0.0068}  # TODO: no target
ENV[ENV__TRAIN__DEFAULT_SAVE_STEP]       = 5
ENV[ENV__TRAIN__DEFAULT_FROM_SCRATCH]    = None

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

    # TODO: include labels
    data_provider = TrainDataProvider(
        x_train = x_train,
        y_train = x_train,  # NOTE: x_train is on purpose since it's autoencoder

        x_val   = x_test,
        y_val   = x_test,   # NOTE: x_test is on purpose since it's autoencoder

        x_test  = x_test,
        y_test  = x_test,   # NOTE: x_test is on purpose since it's autoencoder
    )

    # TODO: data provider for hidden layer

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

    imgs = [img.squeeze() for img in pick_random_pairs([x_test, y_pred], 5)]
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
