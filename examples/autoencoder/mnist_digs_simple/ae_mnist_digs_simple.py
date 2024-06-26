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

from pathlib import Path
# %matplotlib inline

## Утилиты

EPOCH_CALLBACK_DATA = {
    'latent'    : None,
    'x_data'    : None,
    'y_data'    : None,
    'save_path' : None,
}

# Функция-коллбэк. Отрисовывает объекты в скрытом пространстве

def ae_on_epoch_end(epoch, logs):
    print('________________________')
    print(f'*** Epoch: {epoch+1}, loss: {logs["loss"]} ***')
    print('________________________')

    if EPOCH_CALLBACK_DATA['latent'] is None:
        return

    if EPOCH_CALLBACK_DATA['save_path'] is None:
        return

    # Получение картинки латентного пространства в конце эпохи и запись в файл
    # Задание числа пикселей на дюйм
    plt.figure(dpi=100)

    # Предсказание енкодера на тренировочной выборке
    predict = EPOCH_CALLBACK_DATA['latent'].predict(EPOCH_CALLBACK_DATA['x_data'])

    # Создание рисунка: множество точек на плоскости 3-х цветов (3-х классов)
    scatter = plt.scatter(predict[:,0,],predict[:,1], c=EPOCH_CALLBACK_DATA['y_data'], alpha=0.6, s=5)

    # Создание легенды
    legend2 = plt.legend(*scatter.legend_elements(), loc='upper right', title='Классы')

    # Сохранение картинки с названием соответствующим названию эпохи
    image_path = Path({EPOCH_CALLBACK_DATA['save_path']})/f"image_{epoch}.jpg"
    if image_path.exists():
        image_path.unlink()
    if not image_path.parent.exists():
        os.makedirs(image_path.parent, exist_ok=True)
    plt.savefig(image_path)

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
    import ml_kit.boilerplate as bp
    from ml_kit.reporting import *

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
    latent_cnn = [
                    layer_template(Flatten, ),
                    layer_template(Dense,   2,  activation='relu', _name_='latent'),
                    layer_template(Dense,   "$latent_expand_size_flat", activation='$latent_out_activation'),
                    layer_template(Reshape, "$latent_expand_size"),
    ],
    latent_dns = [
                    layer_template(Dense,   2,  activation='$latent_out_activation', _name_='latent'),
    ],
    dns1       = [
                    layer_template(Dense,   28*28*2, activation='$ae_activation'),
    ],
    dns2       = [
                    layer_template(Dense,   14*14, activation='$ae_activation'),
    ],
    dns3       = [
                    layer_template(Dense,   7*7, activation='$ae_activation'),
    ],
    dns_input  = [
                    layer_template(Input,   input_shape),
                    layer_template(Flatten, ),
    ],
    dns_output = [
                    layer_template(Dense,   prod(*input_shape), activation='sigmoid'),
                    layer_template(Reshape, input_shape),
    ],
    cnn1_encoder = [
                    layer_template(Conv2D,  32, (3, 3), padding='same', activation='relu'),
                    layer_template(BatchNormalization, ),
                    layer_template(Conv2D,  32, (3, 3), padding='same', activation='relu'),
                    layer_template(BatchNormalization, ),
                    layer_template(MaxPooling2D, ),
    ],
    cnn1_decoder = [
                    layer_template(Conv2DTranspose, 32, (2, 2), strides=(2, 2), padding='same', activation='relu'),
                    layer_template(Conv2D,  32, (3, 3), padding='same', activation='relu'),
                    layer_template(BatchNormalization, ),
                    layer_template(Conv2D,  32, (3, 3), padding='same', activation='relu'),
                    layer_template(BatchNormalization, ),
    ],
    cnn2_encoder = [
                    layer_template(Conv2D,  64, (3, 3), padding='same', activation='relu'),
                    layer_template(BatchNormalization, ),
                    layer_template(Conv2D,  64, (3, 3), padding='same', activation='relu'),
                    layer_template(BatchNormalization, ),
                    layer_template(MaxPooling2D, ),
    ],
    cnn2_decoder = [
                    layer_template(Conv2DTranspose, 32, (2, 2), strides=(2, 2), padding='same', activation='relu'),
                    layer_template(Conv2D,  32, (3, 3), padding='same', activation='relu'),
                    layer_template(BatchNormalization, ),
                    layer_template(Conv2D,  32, (3, 3), padding='same', activation='relu'),
                    layer_template(BatchNormalization, ),
    ],
    cnn_output = [
                    layer_template(Conv2D,  input_shape[-1], (3, 3), activation='sigmoid', padding='same'),
    ]
)

latent_expand_size = {
    "native"    : input_shape,
    "cnn1"      : (14,14,32),
    "cnn2"      : (7,7,64),
}

cnn_models = to_dict(

    latent = to_dict(
        model_class = Model,
        vars = to_dict(
            latent = 'native',
            latent_out_activation = 'sigmoid',
        ),
        thd_kwargs = to_dict(
            fit_callbacks   = [ae_callback],
        ),
        template = to_dict(
            encoder = to_dict(
                input  = True,
                output = True,
                layers = [
                    layer_template(Input,   input_shape),
                    *model_items['latent_cnn']
                ],
            ),
        ),
    ),

    cdns1 = to_dict(
        model_class = Model,
        vars = to_dict(
            latent = 'cnn1',
            latent_out_activation = 'relu',
        ),
        thd_kwargs = to_dict(
            fit_callbacks   = [ae_callback],
        ),
        template = to_dict(
            encoder = to_dict(
                input  = True,
                layers = [
                    layer_template(Input,   input_shape),
                    *model_items['cnn1_encoder'],
                    *model_items['latent_cnn'],
                ]
            ),
            decoder = to_dict(
                parents = 'encoder',
                output = True,
                layers = [
                    *model_items['cnn1_decoder'],
                    *model_items['cnn_output'],
                ],
            ),
        ),
    ),

    cdns2 = to_dict(
        model_class = Model,
        vars = to_dict(
            latent = 'cnn2',
            latent_out_activation = 'relu',
        ),
        thd_kwargs = to_dict(
            fit_callbacks   = [ae_callback],
        ),
        template = to_dict(
            encoder = to_dict(
                input  = True,
                layers = [
                    layer_template(Input,   input_shape),
                    *model_items['cnn1_encoder'],
                    *model_items['cnn2_encoder'],
                    *model_items['latent_cnn'],
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

    cnn2 = to_dict(
        model_class = Model,
        vars = to_dict(
        ),
        thd_kwargs = to_dict(
            fit_callbacks   = [ae_callback],
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

dns_models_proto = {}

xx = 3
for dn in range(3):
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                if xx <= 0:
                    continue
                idx = tuple([di+1,dj+1,dk+1][:dn+1])
                if len([v for v in idx if v==1]) > 1:
                    continue
                if idx in dns_models_proto:
                    continue
                dns_models_proto[idx] = None
                xx -= 1

dns_models = {}

for bn in (True, False):
    for ae_act in ('relu', 'sigmoid', 'tanh'): # TODO: also try , 'elu', 'softsign', 'softplus', 'mish'):
        for lat_act in ('relu', 'sigmoid', 'tanh', 'same'):
            if lat_act == 'same':
                lat_act = ae_act
            for k,v in dns_models_proto.items():

                model_name = 'dns'+''.join(str(ki) for ki in k)
                if bn:
                    model_name += "b"
                model_name += "-" + ae_act + "-" + lat_act

                if not use_model(model_name):
                    continue

                dns_encoder = []
                for ki in k:
                    dns_encoder += model_items[f'dns{ki}']
                dns_decoder = [*dns_encoder]
                dns_decoder.reverse()
                if bn:
                    for arr in (dns_encoder, dns_decoder):
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
                        fit_callbacks   = [ae_callback],
                    ),
                    vars = to_dict(
                        ae_activation = ae_act,
                        latent_out_activation = lat_act,
                    ),
                    template = to_dict(
                        encoder = to_dict(
                            input  = True,
                            layers = [
                                *model_items['dns_input'],
                                *dns_encoder,
                                *model_items['latent_dns'],
                            ],
                        ),
                        decoder = to_dict(
                            parents = 'encoder',
                            output  = True,
                            layers = [
                                *dns_decoder,
                                *model_items['dns_output']
                            ],
                        ),
                    ),
                )

                dns_models[model_name] = k_model

models = {}
for models_candidates in (cnn_models, dns_models):
    for k, v in models_candidates.items():
        if not use_model(k):
            continue
        models[k] = v

# ---------------------------------------------------------------------------- #

## Гиперпараметры

###
# Гиперпараметры

hp_defaults = to_dict(
        tabs=['learn', 'animo', 'XvsY'],
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
        output_drop_rate    = 0.5
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
    if run_name[:3] == "dns":
        tab_suffix, tab_name = run_name.split("-", 1)
        if tab_id is not None:
            if "-" not in tab_id:
                tab_id += "-" + tab_suffix
    else:
        if tab_id is not None:
            if "-" not in tab_id:
                tab_id += "-cnn"
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

    latent_ref = model_vars.get('latent', None)
    if latent_ref is not None:
        if isinstance(latent_ref, str):
            model_vars['latent_expand_size'] = latent_expand_size[model_vars.get('latent', 'native')]
        else:
            model_vars['latent_expand_size'] = latent_ref
        model_vars['latent_expand_size_flat'] = prod(*model_vars['latent_expand_size'])

    data_provider = TrainDataProvider(
        x_train = x_train,
        y_train = x_train,  # NOTE: x_train is on purpose since it's autoencoder

        x_val   = 0.1,
        y_val   = None,

        x_test  = x_test,
        y_test  = x_test,   # NOTE: x_test is on purpose since it's autoencoder
    )

    return data_provider


def on_model_update(thd):
    # TODO: make two models instead of single branched so encoder can be used to visualize latent space, named layers wont work for this
    return
    EPOCH_CALLBACK_DATA['latent'] = thd.mhd.models['encoder_model']
    EPOCH_CALLBACK_DATA['save_path'] = Path(thd.data_path) / "latent"

score = {}

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

    mse = mhd.context.get_metric_value(S_MSE)
    success = mse < ENV[ENV__TRAIN__DEFAULT_TARGET][S_MSE]
    score[run_name] = to_dict(
        best_epoch = best_metrics['epoch'],
        mse = mse,
        score = best_metrics['epoch'] + mse,
        success = ["No", "Yes"][success],
    )

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
    on_model_update_call=on_model_update,
    get_tab_run_call=get_tab_run,
    print_to_tab_call=print_to_tab,
)

score_table = dict_to_table(score, dict_key='model', sort_key=lambda x: [
    score[x]['mse'] + ENV[ENV__TRAIN__DEFAULT_EPOCHS],
    score[x]['mse'] + score[x]['best_epoch']
    ][score[x]['success']=="Yes"])

print_table(*score_table)
