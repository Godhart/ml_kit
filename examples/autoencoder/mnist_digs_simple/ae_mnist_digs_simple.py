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

###
# Ешё немного служебных функций

def mult(*args):
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
                    layer_template(Dense,   2,  activation='relu', _name_='latent'),
    ],
    dns1       = [
                    layer_template(Dense,   28*28*10,),
    ],
    dns2       = [
                    layer_template(Dense,   14*14,),
    ],
    dns3       = [
                    layer_template(Dense,   7*7,),
    ],
    dns_input  = [
                    layer_template(Input,   input_shape),
                    layer_template(Flatten, ),
    ],
    dns_output = [
                    layer_template(Dense,   mult(*input_shape), activation='$latent_out_activation'),
                    layer_template(Reshape, *input_shape),
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

models = to_dict(

    latent = to_dict(
        model_class = Model,
        vars = to_dict(
            latent = 'native',
            latent_out_activation = 'sigmoid',
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

dns_models = {}

for dn in range(3):
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                idx = tuple([di+1,dj+1,dk+1][:dn+1])
                dns_models[idx] = None

for k,v in dns_models.items():
    if v is not None:
        continue
    dns_encoder = [model_items[f'dns{ki}'] for ki in k]
    dns_decoder = [*dns_encoder]
    dns_decoder.reverse()
    dns_models[k] = to_dict(
        model_class = Model,
        vars = to_dict(
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
    models['dns'+''.join(str(ki) for ki in k)] = dns_models[k]

# ---------------------------------------------------------------------------- #

## Гиперпараметры

###
# Гиперпараметры

hp_defaults = to_dict(
        tabs=['learn', 'lanim'],
)

data_common_vars=to_dict(
    # shuffle         = False,
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

            model_vars = copy.deepcopy(model_data['vars'])
            model_vars = {**model_vars, **copy.deepcopy(hp['model_vars'])}

            latent_ref = model_vars.get('latent', None)
            if latent_ref is not None:
                if isinstance(latent_ref, str):
                    model_vars['latent_expand_size'] = latent_expand_size[model_vars.get('latent', 'native')]
                else:
                    model_vars['latent_expand_size'] = latent_ref
                model_vars['latent_expand_size_flat'] = mult(*model_vars['latent_expand_size'])

            data_provider = TrainDataProvider(
                x_train = x_train,
                y_train = x_train,  # NOTE: x_train is on purpose since it's autoencoder

                x_val   = 0.1,
                y_val   = None,

                x_test  = x_test,
                y_test  = x_test,   # NOTE: x_test is on purpose since it's autoencoder
            )

            mhd = ModelHandler(
                name            = run_name,
                model_class     = model_data['model_class'],
                optimizer       = model_data.get('optimizer', ENV[ENV__TRAIN__DEFAULT_OPTIMIZER]),
                loss            = model_data.get('loss',      ENV[ENV__TRAIN__DEFAULT_LOSS]),
                metrics         = model_data.get('metrics',   hp.get('metrics', ENV[ENV__TRAIN__DEFAULT_METRICS])),
                model_template  = model_data['template'],
                model_variables = model_vars,
                batch_size      = data_vars.get('batch_size',ENV[ENV__TRAIN__DEFAULT_BATCH_SIZE]),
                data_provider   = data_provider,
                load_weights_only = True,   # NOTE: True since named layers are used
            )

            def on_model_update(thd):
                # TODO: make two models instead of branched so encoder can be used to visualize latent space
                return
                EPOCH_CALLBACK_DATA['latent'] = encoder_model
                EPOCH_CALLBACK_DATA['save_path'] = Path(thd.data_path) / "latent"

            thd = TrainHandler(
                data_path       = ENV[ENV__MODEL__DATA_ROOT] / model_data.get("data_path", ENV[ENV__TRAIN__DEFAULT_DATA_PATH]),
                data_name       = run_name,
                mhd             = mhd,
                mhd_class       = ModelHandler,
                on_model_update = on_model_update,
                fit_callbacks   = [ae_callback],
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
