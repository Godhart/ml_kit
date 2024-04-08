# Этот пример в google_colab можно посмотреть по ссылке
# для букв - https://colab.research.google.com/drive/1S5G9ENkY3JbzJomr8fV16EUfMdAEvz7a
# для цифр - https://colab.research.google.com/drive/1Tbneu8h_0UjB0SiGn4ab3XEUE9o638lA

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

# Получим входные данные
if DATA == S_DIGS:
    (xs_train, y_train), (xs_test, y_test) = mnist.load_data()
elif DATA == S_ALPHAS:
    # Скачаем обучающую выборку
    # a = emnist.list_datasets()
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

# Создание собственных слоёв
from tensorflow.keras import layers

# Функция среднеквадратической ошибки для расчетов вручную
from sklearn.metrics import mean_squared_error

from functools import partial

import math

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
TRAIN_EXCLUDE = [r"hm(3|4)_(.*?)--ld.*"]

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


def use_model_hp(model_name, model_data, hp_name, hp, run_name):
    if TRAIN_EXCLUDE is not None:
        for item in TRAIN_EXCLUDE:
            if re.match(item, run_name) is not None:
                return False
    if TRAIN_INCLUDE is None:
        return True
    else:
        for item in TRAIN_INCLUDE:
            if re.match(item, run_name) is not None:
                return True
        return False


ENV[ENV__DEBUG_PRINT] = True
ENV[ENV__JUPYTER] = False

# Подключение google диска если работа в ноутбуке
# connect_gdrive()

# ---------------------------------------------------------------------------- #

# Перегрузка окружения под текущую задачу
ENV[ENV__TRAIN__DEFAULT_DATA_PATH]       = "lesson_14_pro_1"+['b','a'][DATA==S_DIGS]
ENV[ENV__TRAIN__DEFAULT_OPTIMIZER]       = [Adam, [], to_dict(learning_rate=1e-4)] # TODO: change learning rate ?
ENV[ENV__TRAIN__DEFAULT_LOSS]            = None
ENV[ENV__TRAIN__DEFAULT_METRICS]         = [S_MSE]
ENV[ENV__TRAIN__DEFAULT_BATCH_SIZE]      = 128
ENV[ENV__TRAIN__DEFAULT_EPOCHS]          = 2
ENV[ENV__TRAIN__DEFAULT_TARGET]          = {S_MSE: 0.0}
ENV[ENV__TRAIN__DEFAULT_SAVE_STEP]       = 5
ENV[ENV__TRAIN__DEFAULT_FROM_SCRATCH]    = None


# ---------------------------------------------------------------------------- #

# Доподготовка данных

NUM_CLASSES = int(np.max(y_train)) + 1

y_train_ohe = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_ohe  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

# ---------------------------------------------------------------------------- #

###
# Вспомогательные функции

# Создадим функцию - генератор случайных чисел с заданными параметрами
# (используется в модели hm1)
def noise_gen(args, latent_dim):
    z_mean, z_log_var = args

    # Генерируем тензор из нормальных случайных  чисел с параметрами (0,1)
    N = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)

    # Вернем тензор случайных чисел с заданной дисперсией и мат.ожиданием
    return K.exp(z_log_var / 2) * N + z_mean


# Функция потерь с учётом данных скрытого пространства
def cvae_loss(mhd):
    input_img   = mhd.named_layers['encoder/encoder_main/input']
    z_mean      = mhd.named_layers['encoder/latent/z_log_var']
    z_log_var   = mhd.named_layers['encoder/latent/z_mean']
    outputs     = mhd.data["ae"][S_MODEL]

    reconstruction_loss = keras.losses.MSE(input_img, outputs)      # Рассчитаем ошибку восстановления изображения - лоссы MSE
    reconstruction_loss *= mult(x_train.shape[1:])                  # Уберем нормировку MSE
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)   # Рассчитаем лоссы KL
    kl_loss = -0.5* K.sum(kl_loss, axis=-1)                         #
    result = K.mean(reconstruction_loss) +  K.mean(kl_loss)         # Суммируем лоссы - здесь можно вводить веса
    return result


# Пустышка функции потерь
# (использовать там где формально параметр требуется, но он влияния не оказывает)
def dummy_loss(mhd):
    return 0


# Генератор выходных изображений из точек скрытого пространства
def lat2img(
    model,
    class_n,
    total_samples,
    classes_count = None,   # TODO: get from model
    latent_dim    = None,   # TODO: get from model
):
    # Общий размер картинки
    # (сторона квадрата способного вместить в себя total_samples)
    axis_size = int(math.ceil(math.sqrt(total_samples)))
    # Метка требуемого класса в кодировке OHE
    input_lbl = np.zeros((1, classes_count))
    input_lbl[0, class_n] = 1
    # Заготовим общую картинку
    plt.figure(figsize=(axis_size, axis_size))
    # Переменная для перебора точек в скрытом пространстве
    h = np.zeros((1, latent_dim))
    # Исходное положение точки - "отрицательный угол" в скрытом пространстве
    h.fill(-1)

    # Шаг точки в скрытом пространстве
    # (такой чтобы за нужное число сэмплов перебрать пространство полностью)
    latent_step = 2. / math.pow(total_samples, 1/latent_dim)
    # NOTE: 2 т.к. диапазон перебора по каждому измерению [-1:1]

    for i in range(total_samples):
        # Заготовим отдельную картинку для цифры
        ax = plt.subplot(axis_size, axis_size, i+1)
        # Генерируем изображение, соответствующее точке скрытого пространства и требуемому классу
        img = model.predict([h, input_lbl])
        # Вывод изображения
        plt.imshow(img.squeeze(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Рассчёт следующей точки в скрытом пространстве
        for j in range(latent_dim):
            h[0][j] += latent_step
            if h[0][j] <= 1:
                break
            else:
                h[0][j] -= 2
    plt.show()


# Функция для оценки переноса стиля
def img2img(
    model,
    imgs,
    labels,
    direction     = S_COLS,
    classes_count = None,  # TODO: get from model
):
    # Общий размер картинки
    # (сторона квадрата способного вместить в себя total_samples)
    if direction == S_COLS:
        h_size = len(imgs)
        v_size = classes_count + 1
        step = h_size
    else:
        v_size = len(imgs)
        h_size = classes_count + 1
        step = 1
    # Заготовим общую картинку
    plt.figure(figsize=(v_size, h_size))

    for i in range(len(imgs)):
        if direction == S_COLS:
            img_idx = 1+i
        else:
            img_idx = 1+i*(classes_count+1)

        # Показать исходное изображение
        ax = plt.subplot(v_size, h_size, img_idx)
        plt.imshow(imgs[i].squeeze(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


        # Показать производные от него для других классов
        for j in range(classes_count):

            # Метка требуемого класса в кодировке OHE
            output_lbl = np.zeros((1, classes_count))
            output_lbl[0, j] = 1

            # Генерируем изображение, соответствующее по стилю входному изображению и требуемому классу
            img = model.predict([imgs[i:i+1,], labels[i:i+1,], output_lbl])

            # Заготовим отдельную картинку для изображения
            img_idx += step
            ax = plt.subplot(v_size, h_size, img_idx)
            # Вывод изображения
            plt.imshow(img.squeeze(), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()


# Создадим класс для генерации случайных чисел Sampling
# (используется в модели hm2)
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):               # На входе мат ожидание и дисперсия
        z_mean, z_log_var = inputs        # Разделим вход на 2 параметра
        batch = tf.shape(z_mean)[0]       # Найдем размер батча
        dim = tf.shape(z_mean)[1]         # Найдем размер элемента

        # Создадим тензор из нормальных случайных чисел параметрами (0,1)
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        #Создадим и вернем тензор в нужныммат.ожиданием и дисперсией
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ---------------------------------------------------------------------------- #

## Модели

###
# Используемые модели


input_shape = x_train.shape[1:]


def enc_cnn_layer(prefix, idx, neurons, kernel_size, strides):
    return [
        layer_template(Conv2D,  neurons, kernel_size=kernel_size, strides=strides, padding='same', name=f"{prefix}_cnn{idx}",),
        layer_template(BatchNormalization, name=f"{prefix}_ban{idx}",),
        layer_template("$cnn_activation", name=f"{prefix}_act{idx}",),
    ]


def dec_cnn_layer(prefix, idx, neurons, kernel_size, strides):
    return [
        layer_template(Conv2DTranspose,  neurons, kernel_size=kernel_size, strides=strides, padding='same', name=f"{prefix}_cnn{idx}",),
        layer_template(BatchNormalization, name=f"{prefix}_ban{idx}",),
        layer_template("$cnn_activation", name=f"{prefix}_act{idx}",),
    ]

########################################
# Elementary items to construct models #
# (those that are common to models)    #
########################################
model_items = to_dict(
    encoder_classes = to_dict(
        input = True,
        layers = [
            layer_template(Input,   NUM_CLASSES, name='input_classes'),
        ],
    ),
    encoder_concat = to_dict(
        parents = ["encoder_main", "encoder_classes"],
        layers = [
            layer_template(concatenate, ["$encoder_main", "$encoder_classes",], _parent_=None, name=None),
        ],
    ),
    latent_lambda = to_dict(
        parents = ["encoder_concat"],
        output = True,
        layers = [
            layer_template(Dense, "$ldense_dim", activation='linear',   name='latent_input', ),
            layer_template(Dense, "$latent_dim", name="z_mean"    ,     _output_=0),        # TODO: Activation?
            layer_template(Dense, "$latent_dim", name="z_log_var" ,     _output_=1),        # TODO: Activation?
            layer_template(
                Lambda, "$noise_gen", output_shape=("$latent_dim",) ,   name='noise_gen',
                _parent_=["$z_mean", "$z_log_var"],                     _output_=2),
        ],
    ),
    latent_sampling = to_dict(
        parents = ["encoder_concat"],
        output = True,
        layers = [
            layer_template(Dense, "$ldense_dim", activation='linear',   name='latent_input', ),
            layer_template(Dense, "$latent_dim", name="z_mean"    ,     _output_=0),        # TODO: Activation?
            layer_template(Dense, "$latent_dim", name="z_log_var" ,     _output_=1),        # TODO: Activation?
            layer_template(Sampling, name="noise_gen",
                _parent_=["$z_mean", "$z_log_var"],                     _output_=2),
        ],
    ),
    decoder_input = to_dict(
        input = True,
        layers = [
            layer_template(Input,  shape=("$latent_dim", ), name="input_latent",    _input_ = 0),
            layer_template(Input,  shape=(NUM_CLASSES, ),   name="input_classes",   _input_ = 1),
            layer_template(concatenate, ["$input_latent",   "$input_classes",],     _parent_=None, name=None),
        ],
    ),
)

#####################################
# Large parts / models              #
# (those that are common to models) #
#####################################
handmade_models_parts = to_dict(

    ##############
    # Encoder #1 #
    ##############
    hm1_enc = to_dict(
        model_class = Model,
        vars = to_dict(
            # NOTE: may overridden via hyper params
            ldense_dim  = 2,
            latent_dim  = 2,
            kernel_size = 3,
            noise_gen   = partial(noise_gen, latent_dim=2),
            cnn_activation = LeakyReLU,
        ),
        template = to_dict(
            encoder_main = to_dict(
                input  = True,
                layers = [
                    layer_template(Input,   input_shape, name="input"),
                    *enc_cnn_layer("enc_l", 1, 32, "$kernel_size", 1),
                    *enc_cnn_layer("enc_l", 2, 64, "$kernel_size", 2),
                    *enc_cnn_layer("enc_l", 3, 64, "$kernel_size", 2),
                    *enc_cnn_layer("enc_l", 4, 64, "$kernel_size", 1),
                    layer_template(Flatten, name="enc_cnn_flat"),
                ],
            ),
            encoder_classes = {**model_items['encoder_classes']},
            encoder_concat  = {**model_items['encoder_concat']},
            latent          = {**model_items['latent_lambda']},
        ),
    ),

    ##############
    # Decoder #1 #
    ##############
    hm1_dec = to_dict(
        model_class = Model,
        vars = to_dict(
            # NOTE: may overridden via hyper params
            ldense_dim  = 2,
            latent_dim  = 2,
            kernel_size = 3,
            noise_gen   = partial(noise_gen, latent_dim=2),
            cnn_activation = LeakyReLU,
        ),
        template = to_dict(
            decoder_input = {**model_items['decoder_input']},
            decoder_cnn = to_dict(
                output = True,
                parents = ["decoder_input"],
                layers = [
                    layer_template(Dense,   mult(7, 7, 64), name="dec_input_expand"),   # TODO: Activation?
                    layer_template(Reshape,     (7, 7, 64), name="dec_input_reshape"),
                    *dec_cnn_layer("dec_l", 3, 64,      "$kernel_size", 1),
                    *dec_cnn_layer("dec_l", 2, 64,      "$kernel_size", 2),
                    *dec_cnn_layer("dec_l", 1, 32,      "$kernel_size", 2),
                    layer_template(Conv2DTranspose, 1,  "$kernel_size", padding="same", activation="sigmoid", name="output"),
                ],
            ),
        ),
    ),

    ##############
    # Encoder #2 #
    ##############

    hm2_enc = to_dict(
        model_class = Model,
        vars = to_dict(
            # NOTE: may overridden via hyper params
            ldense_dim  = 16,
            latent_dim  = 2,
            kernel_size = 3,
            cnn_act     = "relu",
        ),
        template = to_dict(
            encoder_main = to_dict(
                input  = True,
                layers = [
                    layer_template(Input,   input_shape, name="input"),
                    layer_template(Conv2D,  32, "$kernel_size", strides=2, activation="$cnn_act", padding="same", name="enc_l1_cnn"),
                    layer_template(Conv2D,  64, "$kernel_size", strides=2, activation="$cnn_act", padding="same", name="enc_l2_cnn"),
                    layer_template(Flatten, name="enc_cnn_flat"),
                ],
            ),
            encoder_classes = {**model_items['encoder_classes']},
            encoder_concat  = {**model_items['encoder_concat']},
            latent          = {**model_items['latent_sampling']},
        ),
    ),

    ##############
    # Decoder #2 #
    ##############
    hm2_dec = to_dict(
        model_class = Model,
        vars = to_dict(
            # NOTE: may overridden via hyper params
            ldense_dim  = 16,
            latent_dim  = 2,
            kernel_size = 3,
            cnn_act     = "relu",
        ),
        template = to_dict(
            decoder_input = {**model_items['decoder_input']},
            decoder_cnn = to_dict(
                output = True,
                parents = ["decoder_input"],
                layers = [
                    layer_template(Dense,   mult(7, 7, 64), name="dec_input_expand"),   # TODO: Activation?
                    layer_template(Reshape,     (7, 7, 64), name="dec_input_reshape"),
                    layer_template(Conv2DTranspose,64,  "$kernel_size", strides=2, padding="same", activation="$cnn_act",   name="dec_l2_cnn"),
                    layer_template(Conv2DTranspose,32,  "$kernel_size", strides=2, padding="same", activation="$cnn_act",   name="dec_l1_cnn"),
                    layer_template(Conv2DTranspose, 1,  "$kernel_size",            padding="same", activation="sigmoid",    name="output"),
                ],
            ),
        ),

        ##############
        # Encoder #3 #
        ##############
        hm3_enc = to_dict(
            model_class = Model,
            vars = to_dict(
                # NOTE: may overridden via hyper params
                ldense_dim  = 2,
                latent_dim  = 2,
                kernel_size = 3,
                noise_gen   = partial(noise_gen, latent_dim=2),
                cnn_activation = LeakyReLU,
            ),
            template = to_dict(
                encoder_main = to_dict(
                    input  = True,
                    layers = [
                        layer_template(Input,   input_shape, name="input"),
                        *enc_cnn_layer("enc_la", 1, 32, "$kernel_size", 1),
                        *enc_cnn_layer("enc_lb", 1, 32, "$kernel_size", 1),
                        *enc_cnn_layer("enc_la", 2, 64, "$kernel_size", 1),
                        *enc_cnn_layer("enc_lb", 2, 64, "$kernel_size", 2),
                        *enc_cnn_layer("enc_la", 3, 64, "$kernel_size", 1),
                        *enc_cnn_layer("enc_lb", 3, 64, "$kernel_size", 2),
                        *enc_cnn_layer("enc_la", 4, 64, "$kernel_size", 1),
                        *enc_cnn_layer("enc_lb", 4, 64, "$kernel_size", 1),
                        layer_template(Flatten, name="enc_cnn_flat"),
                    ],
                ),
                encoder_classes = {**model_items['encoder_classes']},
                encoder_concat  = {**model_items['encoder_concat']},
                latent          = {**model_items['latent_lambda']},
            ),
        ),

        ##############
        # Decoder #3 #
        ##############
        hm3_dec = to_dict(
            model_class = Model,
            vars = to_dict(
                # NOTE: may overridden via hyper params
                ldense_dim  = 2,
                latent_dim  = 2,
                kernel_size = 3,
                noise_gen   = partial(noise_gen, latent_dim=2),
                cnn_activation = LeakyReLU,
            ),
            template = to_dict(
                decoder_input = {**model_items['decoder_input']},
                decoder_cnn = to_dict(
                    output = True,
                    parents = ["decoder_input"],
                    layers = [
                        layer_template(Dense,   mult(7, 7, 64), name="dec_input_expand"),   # TODO: Activation?
                        layer_template(Reshape,     (7, 7, 64), name="dec_input_reshape"),
                        *dec_cnn_layer("dec_la", 3, 64,     "$kernel_size", 1),
                        *dec_cnn_layer("dec_lb", 3, 64,     "$kernel_size", 1),
                        *dec_cnn_layer("dec_la", 2, 64,     "$kernel_size", 2),
                        *dec_cnn_layer("dec_lb", 2, 64,     "$kernel_size", 1),
                        *dec_cnn_layer("dec_la", 1, 32,     "$kernel_size", 2),
                        *dec_cnn_layer("dec_lb", 1, 32,     "$kernel_size", 1),
                        layer_template(Conv2DTranspose, 1,  "$kernel_size", padding="same", activation="sigmoid", name="output"),
                    ],
                ),
            ),
        ),

        ##############
        # Encoder #4 #
        ##############
        hm4_enc = to_dict(
            model_class = Model,
            vars = to_dict(
                # NOTE: may overridden via hyper params
                ldense_dim  = 2,
                latent_dim  = 2,
                kernel_size = 3,
                noise_gen   = partial(noise_gen, latent_dim=2),
                cnn_activation = LeakyReLU,
            ),
            template = to_dict(
                encoder_main = to_dict(
                    input  = True,
                    layers = [
                        layer_template(Input,   input_shape, name="input"),
                        *enc_cnn_layer("enc_la", 1, 64,  "$kernel_size", 1),
                        *enc_cnn_layer("enc_lb", 1, 64,  "$kernel_size", 1),
                        *enc_cnn_layer("enc_la", 2, 128, "$kernel_size", 1),
                        *enc_cnn_layer("enc_lb", 2, 128, "$kernel_size", 2),
                        *enc_cnn_layer("enc_la", 3, 128, "$kernel_size", 1),
                        *enc_cnn_layer("enc_lb", 3, 128, "$kernel_size", 2),
                        *enc_cnn_layer("enc_la", 4, 128, "$kernel_size", 1),
                        *enc_cnn_layer("enc_lb", 4, 128, "$kernel_size", 1),
                        layer_template(Flatten, name="enc_cnn_flat"),
                    ],
                ),
                encoder_classes = {**model_items['encoder_classes']},
                encoder_concat  = {**model_items['encoder_concat']},
                latent          = {**model_items['latent_lambda']},
            ),
        ),

        ##############
        # Decoder #4 #
        ##############
        hm4_dec = to_dict(
            model_class = Model,
            vars = to_dict(
                # NOTE: may overridden via hyper params
                ldense_dim  = 2,
                latent_dim  = 2,
                kernel_size = 3,
                noise_gen   = partial(noise_gen, latent_dim=2),
                cnn_activation = LeakyReLU,
            ),
            template = to_dict(
                decoder_input = {**model_items['decoder_input']},
                decoder_cnn = to_dict(
                    output = True,
                    parents = ["decoder_input"],
                    layers = [
                        layer_template(Dense,   mult(7, 7, 64), name="dec_input_expand"),   # TODO: Activation?
                        layer_template(Reshape,     (7, 7, 64), name="dec_input_reshape"),
                        *dec_cnn_layer("dec_la", 3, 128,    "$kernel_size", 1),
                        *dec_cnn_layer("dec_lb", 3, 128,    "$kernel_size", 1),
                        *dec_cnn_layer("dec_la", 2, 128,    "$kernel_size", 2),
                        *dec_cnn_layer("dec_lb", 2, 128,    "$kernel_size", 1),
                        *dec_cnn_layer("dec_la", 1, 64,     "$kernel_size", 2),
                        *dec_cnn_layer("dec_lb", 1, 64,     "$kernel_size", 1),
                        layer_template(Conv2DTranspose, 1,  "$kernel_size", padding="same", activation="sigmoid", name="output"),
                    ],
                ),
            ),
        ),
    ),

    ######################################
    # Common part of CVAE                #
    # (excluding encoder/decoder models) #
    ######################################
    cvae_common_part = to_dict(
        ae = to_dict(
            model_class = "decoder",
            kwargs = to_dict(name = None),
            inputs = [
                ["encoder_i_",     S_MODEL, 2, ],
                ["decoder", S_NAMED_LAYERS, "decoder_input/input_classes", ],
            ],
        ),
        cvae = to_dict(
            model_class = Model,
            inputs = [
                ["encoder", S_NAMED_LAYERS, "encoder_main/input", ],
                ["encoder", S_NAMED_LAYERS, "encoder_classes/input_classes", ],
                ["decoder", S_NAMED_LAYERS, "decoder_input/input_classes", ],
            ],
            outputs = [
                ["ae", S_MODEL, ]
            ],
        ),
        z_meaner = to_dict(
            model_class = Model,
            make_instance = True,
            inputs = [
                ["encoder", S_NAMED_LAYERS, "encoder_main/input", ],
                ["encoder", S_NAMED_LAYERS, "encoder_classes/input_classes", ],
            ],
            outputs = [
                ["encoder", S_NAMED_LAYERS, "latent/z_mean"],
            ],
        ),
        decoder_tr = to_dict(
            model_class = "decoder",
            kwargs = to_dict(name = None),
            inputs = [
                ["z_meaner_i_",    S_MODEL, ],
                ["decoder", S_NAMED_LAYERS, "decoder_input/input_classes", ],
            ]
        ),
        tr_style = to_dict(
            model_class = Model,
            inputs = [
                ["encoder", S_NAMED_LAYERS, "encoder_main/input", ],
                ["encoder", S_NAMED_LAYERS, "encoder_classes/input_classes", ],
                ["decoder", S_NAMED_LAYERS, "decoder_input/input_classes", ],
            ],
            outputs = [
                ["decoder_tr",    S_MODEL, ],
            ],
        ),
        _output_ = "cvae",
    ),
)


handmade_models = {}

for prefix in ("hm1", "hm2", "hm3", "hm4"):
    handmade_models[f"{prefix}_full"] = to_dict(
        loss = cvae_loss,
        mhd_kwargs = to_dict(
            load_weights_only = True
        ),
        model_class = None,
        vars = to_dict(
            # Synchronize submodel's common vars
            latent_dim  = 2,
            kernel_size = 3,
            noise_gen   = partial(noise_gen, latent_dim=2),
        ),
        submodels = to_dict(
            _kind_ = S_COMPLEX,
            encoder = to_dict(
                model_template = copy.deepcopy(handmade_models_parts[f"{prefix}_enc"]),
                make_instance = True,
            ),
            decoder = to_dict(
                model_template = copy.deepcopy(handmade_models_parts[f"{prefix}_dec"]),
            ),
            **copy.deepcopy(handmade_models_parts["cvae_common_part"]),
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

# NOTE: kept as a sample how to autogenerate models
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

if False:
    from ml_kit.trainer_common import ModelHandler

    mhd = ModelHandler(
        "test",
        Model,
        Adam,
        S_MSE,
        [S_MSE],
        models['hm1_full'][S_SUBMODELS],
        models["hm1_full"][S_VARS],
        load_weights_only=True,
    )

    mhd.create()

    mhd.data['encoder'][S_MODEL].summary()
    mhd.data['decoder'][S_MODEL].summary()
    mhd.model.summary()

    a = 1 / 0


# ---------------------------------------------------------------------------- #

## Гиперпараметры

###
# Гиперпараметры

hp_defaults = to_dict(
        tabs=['learn', 'XvsY', 'lat2img', 'img2img'],
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
    ),
    data_vars={
        **data_common_vars,
    },
    train_vars={
        **train_common_vars,
    },
)


hyper_params_sets = {}

for name, custom_vars in (
    ("ld3", to_dict(latent_dim = 3, noise_gen = partial(noise_gen, latent_dim=3))),
    ("ld4", to_dict(latent_dim = 4, noise_gen = partial(noise_gen, latent_dim=4))),
    ("def", {}),
):
    hp =copy.deepcopy(hp_template)
    hp['model_vars'] = {**hp['model_vars'], **custom_vars}
    hyper_params_sets[name] = hp
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

# Подготовка информации для контролируемого разбиения данных
# В данном случае данные не будут разделяться на выборки
# Будет только использоваться перемешивание
data_split_provider = TrainDataProvider(
    x_train = [i for i in range(int(x_train.shape[0]))], # индексы элементов исходных данных,
                            # которые будут перемешаны и разделены
    y_train = [0]*int(x_train.shape[0]), # актуальные данные для y_train в нашем случае не требуются

    x_val   = 0.1,          # число данных в проверочной выборке
    y_val   = None,

    x_test  = [],           # тестовая выборка не будет использоваться
    y_test  = [],
)

# В словаре хранятся индексы элементов в исходных данных для каждой из подвыборок
data_split = to_dict(
    train =  data_split_provider.x_train + data_split_provider.x_val,
    # восполним обучающую выборку, в качестве проверочной далее в коде будем использовать тестовую
)

# Проинициализировать словарь входных данных (на основании ветвей модели, помеченных как вход)
train_data = {}

y_train_dummy = [None]*int(x_train.shape[0])
y_val_dummy   = [None]*int(x_test.shape[0])
y_test_dummy  = [None]*int(x_test.shape[0])

train_data["images_input"] = TrainDataProvider(
    x_train = x_train,
    y_train = y_train_dummy,
    x_val   = x_test,
    y_val   = y_val_dummy,
    x_test  = x_test,
    y_test  = y_test_dummy,
    split   = data_split,
    split_y = True,
)

train_data["classes_enc"] = TrainDataProvider(
    x_train = y_train_ohe,
    y_train = y_train_dummy,
    x_val   = y_test_ohe,
    y_val   = y_test_dummy,
    x_test  = y_test_ohe,
    y_test  = y_test_dummy,
    split   = data_split,
    split_y = True,
)

train_data["classes_dec"] = TrainDataProvider(
    x_train = y_train_ohe,
    y_train = y_train_dummy,
    x_val   = y_test_ohe,
    y_val   = y_test_dummy,
    x_test  = y_test_ohe,
    y_test  = y_test_dummy,
    split   = data_split,
    split_y = True,
)

train_data["images_output"] = TrainDataProvider(
    x_train = x_train,
    y_train = x_train,
    x_val   = x_test,
    y_val   = x_test,
    x_test  = x_test,
    y_test  = x_test,
    split   = data_split,
    split_y = True,
)


def prepare(
    model_name,
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

    dp_kwargs = {}
    for attr in ("x_train", "x_val", "x_test"):
        dp_kwargs[attr] = {
            "encoder_main/input"            : getattr(train_data["images_input"], attr),
            "encoder_classes/input_classes" : getattr(train_data["classes_enc"],  attr),
            "decoder_input/input_classes"   : getattr(train_data["classes_dec"],  attr),
        }
    for attr in ("y_train", "y_val", "y_test"):
        dp_kwargs[attr] = getattr(train_data["images_output"], attr)

    data_provider = TrainDataProvider(**dp_kwargs)

    return data_provider


def on_model_update(thd: TrainHandler):
    thd.mhd.data_provider.x_order = thd.mhd.inputs_order


score = {}


def plot_learn(
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
    mhd.context.report_to_screen()
    mhd.model.summary()
    mhd.data["encoder"][S_MODEL].summary()
    mhd.data["decoder"][S_MODEL].summary()
    a = 1


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


def plot_lat2img(
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
    for i in range(NUM_CLASSES):
        lat2img(
            mhd.data["decoder"][S_MODEL],
            i,
            100,
            NUM_CLASSES,
            hp['model_vars']['latent_dim'],
        )


def plot_img2img(
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
    img2img(
        mhd.data["tr_style"][S_MODEL],
        x_train[:10],
        y_train_ohe[:10],
        S_COLS,
        NUM_CLASSES
    )


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
            tab_print_map[tab_id] = plot_learn
        elif 'XvsY' in tab_id:
            tab_print_map[tab_id] = plot_XvsY
        elif 'img2img' in tab_id:
            tab_print_map[tab_id] = plot_img2img
        elif 'lat2img' in tab_id:
            tab_print_map[tab_id] = plot_lat2img

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
    on_model_update_call=on_model_update,
    use_model_hp=use_model_hp,
)

score_table = dict_to_table(score, dict_key='model', sort_key=lambda x: [
    score[x]['test_mse'] + ENV[ENV__TRAIN__DEFAULT_EPOCHS],
    score[x]['test_mse'] + score[x]['best_epoch']
    ][score[x]['success']=="Yes"])

print_table(*score_table)
