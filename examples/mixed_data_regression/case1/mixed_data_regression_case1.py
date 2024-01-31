# Работа с массивами данных
import numpy as np

# Работа с табличными данными
import pandas as pd

# Функции-утилиты для работы с категориальными данными
from tensorflow.keras import utils

# Класс для конструирования последовательной модели нейронной сети
from tensorflow.keras.models import Sequential, Model

# Основные слои
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation, Input, concatenate
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D

# Оптимизаторы
from tensorflow.keras.optimizers import Adam, Adadelta, SGD, Adagrad, RMSprop

# Токенизатор для преобразование текстов в последовательности
from tensorflow.keras.preprocessing.text import Tokenizer

# Масштабирование данных
from sklearn.preprocessing import StandardScaler

# Загрузка датасетов из облака google
import gdown

# Регулярные выражения
import re

# Отрисовка графиков
import matplotlib.pyplot as plt

# Метрики для расчета ошибок
from sklearn.metrics import mean_squared_error, mean_absolute_error

from pathlib import Path

# -------------------------------------------------------------------------------------------------------------------- #

# скачиваем базу

d_path = Path('hh_fixed.csv').absolute().resolve()
if not d_path.exists():
    gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l10/hh_fixed.csv', None, quiet=True)

# Чтение файла базы данных
df = pd.read_csv('hh_fixed.csv', index_col=0)
df = df[:125]   # TODO: disable this line

# Вывод количества резюме и числа признаков
print(df.shape)

df.head(3)

# -------------------------------------------------------------------------------------------------------------------- #
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
    from ml_kit.trainer_texts import *
    from ml_kit.texts import *
    from ml_kit.classes import *

# -------------------------------------------------------------------------------------------------------------------- #

# Настройка номеров столбцов

COL_SEX_AGE     = df.columns.get_loc('Пол, возраст')
COL_SALARY      = df.columns.get_loc('ЗП')
COL_POS_SEEK    = df.columns.get_loc('Ищет работу на должность:')
COL_POS_PREV    = df.columns.get_loc('Последеняя/нынешняя должность')
COL_CITY        = df.columns.get_loc('Город')
COL_EMPL        = df.columns.get_loc('Занятость')
COL_SCHED       = df.columns.get_loc('График')
COL_EXP         = df.columns.get_loc('Опыт (двойное нажатие для полной версии)')
COL_EDU         = df.columns.get_loc('Образование и ВУЗ')
COL_UPDATED     = df.columns.get_loc('Обновление резюме')


### Параметрические данные для функций разбора ###

# Курсы валют для зарплат
currency_rate = {
    'usd'    : 65.,
    'kzt'    : 0.17,
    'грн'    : 2.6,
    'белруб' : 30.5,
    'eur'    : 70.,
    'kgs'    : 0.9,
    'сум'    : 0.007,
    'azn'    : 37.5
}

# -------------------------------------------------------------------------------------------------------------------- #

# Списки и словари для разбиения на классы
# Для ускорения работы добавлен счетчик классов, который будет вычислен ниже

# Список порогов возраста
age_class = distinct_classes([0, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63], fuzzy=True)
age_checkup = {}
for age in age_class.classes_labels:
    for i in range(-1,2):
        age_checkup[age+i] = age_class.ohe(age+i)

# Список порогов опыта работы в месяцах
experience_class = distinct_classes([0, 7, 13, 25, 37, 61, 97, 121, 157, 193, 241], fuzzy=True)
exp_checkup = {}
for exp in experience_class.classes_labels:
    for i in range(-1,2):
        exp_checkup[exp+i] = experience_class.ohe(exp+i)

# Классы городов
city_class = multivalue_classes(
    classes={
        'столица'   : ['москва',],
        'мегаполис' : ['санкт-петербург',],
        'миллионник': [
               'новосибирск'     ,
               'екатеринбург'    ,
               'нижний новгород' ,
               'казань'          ,
               'челябинск'       ,
               'омск'            ,
               'самара'          ,
               'ростов-на-дону'  ,
               'уфа'             ,
               'красноярск'      ,
               'пермь'           ,
               'воронеж'         ,
               'волгоград'       ,
            ],
        'прочие города': ['китеж',],
    },
    rest = 'китеж',
    fuzzy = True,
)
city_checkup = {}
for city in city_class.values:
    city_checkup[city] = city_class.ohe(f"Славный Город {city.capitalize()}")

# Классы занятости
employment_class = multivalue_classes(
    values_map= {
        'стажировка'          : 0,
        'частичная занятость' : 1,
        'проектная работа'    : 2,
        'полная занятость'    : 3,
    },
    fuzzy=True,
)
emp_checkup = {}
for i in range(1, 2**len(employment_class)):
    value = ""
    for j in range(len(employment_class)):
        if ((i >> j) & 0x1) == 0x1:
            value += " " + employment_class.classes[j][0]
    emp_checkup[value] = employment_class.mpe(value)

# Классы графика работы
schedule_class = multivalue_classes(
    values_map = {
        'гибкий график'         : 0,
        'полный день'           : 1,
        'сменный график'        : 2,
        'удаленная работа'      : 3
    },
    fuzzy=True,
)

# Классы образования
education_class = multivalue_classes(
    values_map = {
        'высшее образование'   : 0,
        'higher education'     : 0,
        'среднее специальное'  : 1,
        'неоконченное высшее'  : 2,
        'среднее образование'  : 3
    },
    fuzzy=True,
)

# -------------------------------------------------------------------------------------------------------------------- #

# Нетипичный парсинг

# Разбор значений пола, возраста

base_update_year = 2019

def extract_sex_age_years(arg):
    # Ожидается, что значение содержит "мужчина" или "женщина"
    # Если "мужчина" - результат 1., иначе 0.
    arg = purify(arg)
    sex = 1. if 'муж' in arg else 0.

    try:
        # Выделение года и вычисление возраста
        years = base_update_year - int(re.search(r'\d{4}', arg)[0])

    except (IndexError, TypeError, ValueError):
        # В случае ошибки год равен 0
        years = 0

    return sex, years

# Разбор значения зарплаты

def extract_salary(arg):
    arg = purify(arg)
    try:
        # Выделение числа и преобразование к float
        value = float(re.search(r'\d+', arg)[0])

        # Поиск символа валюты в строке, и, если найдено,
        # приведение к рублю по курсу валюты
        for currency, rate in currency_rate.items():
            if currency in arg:
                value *= rate
                break

    except TypeError:
        # Если не получилось выделить число - вернуть 0
        value = 0.

    return value / 1000.                  # В тысячах рублей

# Разбор данных о городe и преобразование в one hot encoding

def extract_education_to_multi(arg):
    arg = purify(arg)
    result = education_class.mpe(arg)

    # Поправка: неоконченное высшее не может быть одновременно с высшим
    if result[2] > 0.:
        result[0] = 0.

    return result

# Разбор данных об опыте работы - результат в месяцах

def extract_experience_months(arg):
    arg = purify(arg)
    try:
        # Выделение количества лет, преобразование в int
        years = int(re.search(r'(\d+)\s+(год.?|лет)', arg)[1])

    except (IndexError, TypeError, ValueError):
        # Неудача - количество лет равно 0
        years = 0

    try:
        # Выделение количества месяцев, преобразование в int
        months = int(re.search(r'(\d+)\s+месяц', arg)[1])

    except (IndexError, TypeError, ValueError):
        # Неудача - количество месяцев равно 0
        months = 0

    # Возврат результата в месяцах
    return years * 12 + months

# Функция извлечения данных о профессии

def extract_prof_text(row_list):
    result = []

    # Для всех строк таблицы: собрать значения
    # столбцов желаемой и прошлой должности
    # если есть информация о зарплате

    for row in row_list:
        if extract_salary(row[COL_SALARY]) > 0:
            result.append(str(row[COL_POS_SEEK]) + ' ' + str(row[COL_POS_PREV]))

    # Возврат в виде массива
    return result

# Функция извлечения данных описания опыта работы

def extract_exp_text(row_list):
    result = []

    # Для всех строк таблицы: собрать значения опыта работы,
    # если есть информация о зарплате
    for row in row_list:
        if extract_salary(row[COL_SALARY]) > 0:
            result.append(str(row[COL_EXP]))

    # Возврат в виде массива
    return result

# -------------------------------------------------------------------------------------------------------------------- #

def extract_row_data(row):

    # Извлечение и преобразование данных
    sex, age = extract_sex_age_years(row[COL_SEX_AGE])      # Пол, возраст
    sex_vec = np.array([sex])                               # Пол в виде вектора
    age_ohe = age_class.ohe(age)                            # Возраст в one hot encoding
    city_ohe = city_class.ohe(row[COL_CITY])                # Город
    empl_multi = employment_class.mpe(row[COL_EMPL])        # Тип занятости
    sсhed_multi = schedule_class.mpe(row[COL_SCHED])        # График работы
    edu_multi = extract_education_to_multi(row[COL_EDU])    # Образование
    exp_months = extract_experience_months(row[COL_EXP])    # Опыт работы в месяцах
    exp_ohe = experience_class.ohe(exp_months)              # Опыт работы в one hot encoding
    salary = extract_salary(row[COL_SALARY])                # Зарплата в тысячах рублей
    salary_vec = np.array([salary])                         # Зарплата в виде вектора

    # Объединение всех входных данных в один общий вектор
    x_data = np.hstack(
        [
            sex_vec,
            age_ohe,
            city_ohe,
            empl_multi,
            sсhed_multi,
            edu_multi,
            exp_ohe
        ]
    )

    # Возврат входных данных и выходных (зарплаты)
    return x_data, salary_vec


# Создание общей выборки
def construct_train_data(row_list):
    x_data = []
    y_data = []

    for row in row_list:
        x, y = extract_row_data(row)
        if y[0] > 0:                      # Данные добавляются, только если есть зарплата
            x_data.append(x)
            y_data.append(y)

    return np.array(x_data), np.array(y_data)

# -------------------------------------------------------------------------------------------------------------------- #

# Формирование выборки из загруженного набора данных
x_data, y_data = construct_train_data(df.values)

# Извлечение текстов об опыте работы для выборки
prof_texts = extract_prof_text(df.values)

# Извлечение текстов об опыте работы для выборки
exp_texts = extract_exp_text(df.values)

# -------------------------------------------------------------------------------------------------------------------- #

# Форма наборов параметров и зарплат
print(f"x_data shape: {x_data.shape}")
print(f"y_data shape: {y_data.shape}")

# Пример обработанных данных
n = 0
print(f"x_data[{n}] : {x_data[n]}")
print(f"y_data[{n}] : {y_data[n]}")

# -------------------------------------------------------------------------------------------------------------------- #


# Исходные данные для примеров
print(df.values[120])

# Пример текста об опыте работы из резюме
print(prof_texts[120])

# Пример текста об опыте работы из резюме
print(exp_texts[120])


# -------------------------------------------------------------------------------------------------------------------- #

###
# Решение задачи

ENV[ENV__DEBUG_PRINT] = True

if not STANDALONE:
    # Подключение google диска если работа в ноутбуке
    connect_gdrive()

    # А так же директива для отрисовки matplot в вывод ноутбука
    # %matplotlib inline

# -------------------------------------------------------------------------------------------------------------------- #
# Перегрузка констант под текущую задачу

ENV[ENV__TEXTS__BOW_USED] = False

ENV[ENV__TRAIN__DEFAULT_DATA_PATH]       = "lesson_7_lite_1a"
ENV[ENV__TRAIN__DEFAULT_OPTIMIZER]       = [Adam, [], to_dict(learning_rate=1e-3)]
ENV[ENV__TRAIN__DEFAULT_LOSS]            = "mse"
ENV[ENV__TRAIN__DEFAULT_METRICS]         = ["mae"]
ENV[ENV__TRAIN__DEFAULT_BATCH_SIZE]      = 256
ENV[ENV__TRAIN__DEFAULT_EPOCHS]          = 2 # TODO: 50
ENV[ENV__TRAIN__DEFAULT_TARGET_ACCURACY] = 1.0   # максимум чтобы каждая сеть прошла все эпохи обучения
ENV[ENV__TRAIN__DEFAULT_SAVE_STEP]       = 10
ENV[ENV__TRAIN__DEFAULT_FROM_SCRATCH]    = None

# -------------------------------------------------------------------------------------------------------------------- #

# Для нормализации данных используется готовый инструмент
y_scaler = StandardScaler()

# Нормализация выходных данных по стандартному нормальному распределению
y_data_scaled = y_scaler.fit_transform(y_data)

# Проверка нормализации
print(y_data_scaled.shape)
print(f'Оригинальное значение зарплаты:  {y_data[1, 0]}')
print(f'Нормированное значение зарплаты: {y_data_scaled[1, 0]}')

# Вывод границ ненормализованных и нормализованных данных
print(y_data.mean(), y_data.std())
print(y_data_scaled.mean(), y_data_scaled.std())

# Поменять значение y_data на нормализованное, чтобы в коде ниже оставить имя переменной y_data
y_data_unscaled = y_data
y_data = y_data_scaled

# Поставщик данных для ветки "базового" типа
general_data = TrainDataProvider(x_data, y_data, None, None, None, None)

# Шаблоны моделей для обучения

models = to_dict(
    initial = to_dict(
        name = "initial",
        model_class = Model,
        template = to_dict(

            # Branch 1 (processes x_train)
            branch_general = to_dict(
                input = True,
                layers = [
                    layer_template(Input,   "$branch_general_input_shape"),
                    layer_template(Dense,   128, activation="relu"),
                    layer_template(Dense,   1000, activation="tanh"),
                    layer_template(Dense,   100, activation="relu"),
                ],
            ),

            # Branch 2 (processes x_train_prof)
            branch_prof = to_dict(
                input = True,
                layers = [
                    layer_template(Input,   "$branch_prof_input_shape"),
                    layer_template(Dense,   20, activation="relu"),
                    layer_template(Dense,   500, activation="relu"),
                    layer_template(Dropout, "$globals_drop_rate"),
                ],
            ),

            # Branch 3 (processes x_train_exp)
            branch_exp = to_dict(
                input = True,
                layers = [
                    layer_template(Input,   "$branch_exp_input_shape"),
                    layer_template(Dense,   30, activation="relu"),
                    layer_template(Dense,   800, activation="relu"),
                    layer_template(Dropout, "$globals_drop_rate"),
                ],
            ),

            # Output branch
            branch_o = to_dict(
                output = True,
                layers = [
                    layer_template(concatenate, ["$branch_general", "$branch_prof", "$branch_exp"]),
                    layer_template(Dense,   15, activation='relu'),
                    layer_template(Dropout, "$output_drop_rate"),
                    layer_template(Dense,   1, activation='linear'),
                ]
            )
        )
    )
)

hyper_params_sets = (
    to_dict(
        tabs=['learn:initial', 'regress:initial'],
        model='initial',
        model_vars=to_dict(
            globals_drop_rate = 0.3,
            output_drop_rate = 0.5
        ),
        prof_data_vars = to_dict(vocab_size=3000, chunk_size=None, chunk_step=None),
        exp_data_vars = to_dict(vocab_size=3000, chunk_size=None, chunk_step=None),
    ),
)

# -------------------------------------------------------------------------------------------------------------------- #

# Создать вкладки для вывода результатов
from IPython.display import clear_output, display
import ipywidgets as widgets
from functools import lru_cache

model_tabs = {}
tabs_dict = {}
for k in models:
    for hyper_params in hyper_params_sets:
        for tab_id in hyper_params['tabs']:
            tab_group, tab_i = tab_id.split(":")
            if tab_group not in tabs_dict:
                tabs_dict[tab_group] = {}
            widget = tabs_dict[tab_group][tab_i] = widgets.Output()
            with widget:
                # По умолчанию заполнить текст вкладок информацией о параметрах модели
                clear_output()
                print(f"{k}--{hyper_params}")

tabs_objs = {k: widgets.Tab() for k in tabs_dict}
for k, v in tabs_dict.items():
    tab_items_keys = list(sorted(v.keys()))
    tabs_objs[k].children = [v[kk] for kk in tab_items_keys]
    for i in range(0, len(tab_items_keys)):
        tabs_objs[k].set_title(i, f"{k}:{tab_items_keys[i]}")


# Вывести вкладки с результатами
# -------------------------------------------------------------------------------------------------------------------- #
tabs_objs.keys()
# -------------------------------------------------------------------------------------------------------------------- #
display(tabs_objs["learn"])
# -------------------------------------------------------------------------------------------------------------------- #
display(tabs_objs["regress"])
# -------------------------------------------------------------------------------------------------------------------- #

dummy_output = widgets.Output()

# Обучение сети с нужными параметрами и сохранение результатов на диск

n_data = [i for i in range(len(y_data))]

data_split_provider = TrainDataProvider(
    x_train = n_data,
    y_train = y_data,
    x_val   = 0.1,
    y_val   = None,
    x_test  = 0.1,
    y_test  = None,
)

data_split = to_dict(
    train =  data_split_provider.x_train,
    val   =  data_split_provider.x_val,
    test  =  data_split_provider.x_test,
)

for hyper_params in hyper_params_sets:
    hyper_params = copy.deepcopy(hyper_params)  # NOTE: make a copy since hyper_params may be changed, so original would would left at place and won't be polluted with temporary classes

    with dummy_output:
        clear_output()

    # Функция, которая донастраивает модели из шаблона под текущие параметры
    def tune_model(model, hyper_params):
        tuned_model = copy.deepcopy(model)
        # TODO: change name according to hyper params like this: tuned_model['name'] += f"--vocab_{hyper_params['v']['vocab_size']}--chunk_{hyper_params['v']['chunk_size']}_{hyper_params['v']['chunk_step']}"
        # TODO: better use subst_vars for this (after regex support)
        return tuned_model

    def display_callback(message):
        for tab_id in hyper_params['tabs']:
            tab_group, tab_i = tab_id.split(":")
            with tabs_dict[tab_group][tab_i]:
                # Вывести во вкладку информацию об обучении
                print(message)

    model_data  = models[hyper_params['model']]
    model_vars  = hyper_params['model_vars']

    tune_model(model_data, hyper_params)

    model_x_data = {
        k: None for k, v in model_data['template'].items() if v.get("input") is True
    }

    for k in model_x_data:

        if k == "branch_general":
            model_x_data[k] = general_data
            continue

        elif k == "branch_prof":
            data = TextTrainDataProvider(
                texts=TrainDataProvider(prof_texts, y_data, None, None, None, None),
                bow_default=True,
                text_prepare_function=prepare_texts_dummy,
                debug=ENV[ENV__DEBUG_PRINT],
                **hyper_params.get(
                    "prof_data_vars",
                    to_dict(
                        vocab_size = 3000,
                        chunk_size = None,
                        chunk_step = None,
                    ),
                ),
            )
            model_x_data[k] = data
            continue

        elif k == "branch_exp":
            data = TextTrainDataProvider(
                texts=TrainDataProvider(exp_texts, y_data, None, None, None, None),
                bow_default=True,
                text_prepare_function=prepare_texts_dummy,
                debug=ENV[ENV__DEBUG_PRINT],
                **hyper_params.get(
                    "exp_data_vars",
                    to_dict(
                        vocab_size = 3000,
                        chunk_size = None,
                        chunk_step = None,
                    ),
                ),
            )
            model_x_data[k] = data
            continue
        
        else:
            raise ValueError(f"Don't know how to provide data to branch '{k}'!")
        
    x_data_dict = {
        k:v.x_train for k,v in model_x_data.items()
    }

    data_provider = TrainDataProvider(
        x_train = x_data_dict,
        y_train = data_split_provider.y_train,
        x_val   = None,
        y_val   = data_split_provider.y_val,
        x_test  = None,
        y_test  = data_split_provider.y_test,
        split   = data_split,
        split_y = False,
    )

    for k,v in model_x_data.items():
        model_vars[k+"_input_shape"] = v.x_train.shape[1]

    mhd = ModelHandler(
        name            = model_data['name'],
        model_class     = model_data['model_class'],
        optimizer       = model_data.get('optimizer', ENV[ENV__TRAIN__DEFAULT_OPTIMIZER]),
        loss            = model_data.get('loss',      ENV[ENV__TRAIN__DEFAULT_LOSS]),
        metrics         = model_data.get('metrics',   ENV[ENV__TRAIN__DEFAULT_METRICS]),
        model_template  = model_data['template'],
        model_variables = model_vars,
        batch_size      = model_data.get('batch_size',ENV[ENV__TRAIN__DEFAULT_BATCH_SIZE]),
        data_provider   = data_provider,
    )

    def on_model_update(thd):
        data_provider.x_order = thd.mhd.inputs_order

    thd = TrainHandler(
        data_path       = ENV[ENV__MODEL__DATA_ROOT] / model_data.get("data_path", ENV[ENV__TRAIN__DEFAULT_DATA_PATH]),
        data_name       = model_data['name'],
        mhd             = mhd,
        mhd_class       = ModelHandler,
        on_model_update = on_model_update
    )
    
    thd.train(
        from_scratch    = model_data.get("from_scratch", ENV[ENV__TRAIN__DEFAULT_FROM_SCRATCH]),
        epochs          = model_data.get("epochs", ENV[ENV__TRAIN__DEFAULT_EPOCHS]),
        target_accuracy = model_data.get("target_accuracy", ENV[ENV__TRAIN__DEFAULT_TARGET_ACCURACY]),
        save_step       = model_data.get("save_step", ENV[ENV__TRAIN__DEFAULT_SAVE_STEP]),
        display_callback= display_callback
    )

    # Вывод результатов для сравнения
    full_history = copy.deepcopy(mhd.context.history)
    thd.load_best()
    mhd.context.report_history = full_history

    for tab_id in hyper_params['tabs']:
        tab_group, tab_i = tab_id.split(":")
        # TODO: is it required to call `mhd.update_data()` ?
        with tabs_dict[tab_group][tab_i]:
            clear_output()
            if tab_group == "learn":
                mhd.context.report_to_screen()
                mhd.model.summary()
                utils.plot_model(mhd.model, dpi=60)
            else:
                limit = 1000.   # TODO: get from input data / classes specification
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(data_provider.y_test, mhd.context.test_pred)
                ax.set_xlim(0, limit)                     # Пределы по x, y
                ax.set_ylim(0, limit)
                ax.plot(plt.xlim(), plt.ylim(), 'r')      # Отрисовка диагональной линии
                plt.xlabel('Правильные значения')
                plt.ylabel('Предсказания')
                plt.grid()
                plt.show()

        with dummy_output:
            clear_output()

    mhd.context.report_history = None
    mhd.unload_model()
