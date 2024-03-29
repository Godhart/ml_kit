# PLT_CUSTOM = [
#     plt.xlim(0, length),
#     plt.tight_layout(),
#     plt.show(),
# ]

# Этот пример в google_colab можно посмотреть по ссылке https://colab.research.google.com/drive/1QLOn1dScOUWrnLCiV9NrO0wgbz4R_XgD

# Работа с массивами
import numpy as np

# Работа с таблицами
import pandas as pd

# Классы-конструкторы моделей нейронных сетей
from tensorflow.keras.models import Sequential, Model

# Основные слои
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, Conv2D, LSTM, GlobalMaxPooling1D, MaxPooling1D, RepeatVector

# Оптимизаторы
from tensorflow.keras.optimizers import Adam

# Генератор выборки временных рядов
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Отрисовка модели
from tensorflow.keras import utils

# Нормировщики
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Загрузка датасетов из облака google
import gdown

# Отрисовка графиков
import matplotlib.pyplot as plt

# Отрисовка графики в ячейке colab
# %matplotlib inline

# Отключение предупреждений
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

# -------------------------------------------------------------------------------------------------------------------- #

# Назначение размера и стиля графиков по умолчанию
from pylab import rcParams
plt.style.use('ggplot')
rcParams['figure.figsize'] = (14, 7)

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
    from ml_kit.plots import *
    from ml_kit.trainer_series import *

# -------------------------------------------------------------------------------------------------------------------- #

# скачиваем данные

data_sources = [
    'https://storage.yandexcloud.net/aiueducation/Content/base/l11/16_17.csv',
    'https://storage.yandexcloud.net/aiueducation/Content/base/l11/18_19.csv',
]

for ds in data_sources:
    d_path = Path(ds.split('/')[-1]).absolute().resolve()
    if not d_path.exists():
        gdown.download(ds, None, quiet=True)

# -------------------------------------------------------------------------------------------------------------------- #

# Загрузка датасетов с удалением ненужных столбцов по дате и времени

data16_17 = pd.read_csv('16_17.csv', sep=';').drop(columns=['DATE', 'TIME'])
data18_19 = pd.read_csv('18_19.csv', sep=';').drop(columns=['DATE', 'TIME'])

# Создание общего набора данных из двух датасетов

data = pd.concat([data16_17, data18_19]).to_numpy()

# Задание текстовых меток каналов данных (столбцов)

channel_names = ['Open', 'Max', 'Min', 'Close', 'Volume']

if True:
    # Проверка на тестовых данных
    j=11
    n=0
    test_data = []
    for i in range(10000):
        if True:
            test_data.append([n,n,n,n,n])
        else:
            if n < j-1:
                test_data.append([n,n,n,n,1.])
            else:
                test_data.append([n,n,n,n,0.])
        n += 1
        if n >= j:
            n = 0
            j += 7
        if j > 256:
            j = 1

    data = pd.DataFrame(test_data, columns=['OPEN', 'MAX', 'MIN', 'CLOSE', 'VOLUME']).to_numpy()


# -------------------------------------------------------------------------------------------------------------------- #

ENV[ENV__DEBUG_PRINT] = True

IDX_LABEL = {k:v for k,v in enumerate(channel_names)}
LABEL_IDX = {v:k for k,v in IDX_LABEL.items()}

# Подключение google диска если работа в ноутбуке
if not STANDALONE:
    connect_gdrive()

# -------------------------------------------------------------------------------------------------------------------- #

###
# Задача в целом:
# - прогнозирование 'Close' из предыдущих 'Open', 'Max', 'Min', 'Close', 'Volume'
#   - просто очередное значение = массив из 1 очередного значения
#   - условие не понятно -
#       - или предсказать массив из 10 очередных значений
#       - или сделать 10 вариантов с предсказанием одного значения но на N шагов вперёд?
# - построить графики сравнения / корреляции
# - обучить на разных архитектурах и сравнить

# Перегрузка окружения под текущую задачу
ENV[ENV__TRAIN__DEFAULT_DATA_PATH]       = "lesson_8_lite_1b"
ENV[ENV__TRAIN__DEFAULT_OPTIMIZER]       = [Adam, [], to_dict(learning_rate=1e-4)]
ENV[ENV__TRAIN__DEFAULT_LOSS]            = S_MSE
ENV[ENV__TRAIN__DEFAULT_METRICS]         = [S_CORRELATION, S_CORRELATION_PEAK, S_MAE]
ENV[ENV__TRAIN__DEFAULT_BATCH_SIZE]      = 128
ENV[ENV__TRAIN__DEFAULT_EPOCHS]          = 20 # TODO: 50
ENV[ENV__TRAIN__DEFAULT_TARGET]          = {S_CORRELATION: 1, S_CORRELATION_PEAK: 0, S_MAE: 0}
ENV[ENV__TRAIN__DEFAULT_SAVE_STEP]       = 5
ENV[ENV__TRAIN__DEFAULT_FROM_SCRATCH]    = None

# -------------------------------------------------------------------------------------------------------------------- #

###
# Используемые модели

models = to_dict(

    dense = to_dict(
        model_class = Model,
        template = to_dict(
            branch_general = to_dict(
                input  = True,
                output = True,
                layers = [
                    layer_template(Input,   "$branch_general_input_shape"),
                    layer_template(Dense,   150,                  activation='relu'),
                    layer_template(Flatten, ),
                    layer_template(Dense,   "$output_neurons",    activation='$auto_output_activation'),
                ],
            ),
        ),
    ),

    conv1d = to_dict(
        model_class = Model,
        template = to_dict(
            branch_general = to_dict(
                input = True,
                output = True,
                layers = [
                    layer_template(Input,   "$branch_general_input_shape"),
                    layer_template(Conv1D, 64, 5, activation='relu'),
                    layer_template(Conv1D, 64, 5, activation='relu'),
                    layer_template(MaxPooling1D, ),
                    layer_template(Flatten, ),
                    layer_template(Dense,   "$output_neurons",    activation='$auto_output_activation'),
                ],
            ),
        ),
    ),

    rnn = to_dict(
        model_class = Model,
        template = to_dict(
            branch_general = to_dict(
                input = True,
                output = True,
                layers = [
                    layer_template(Input,   "$branch_general_input_shape"),
                    layer_template(LSTM,    5,),
                    layer_template(Dense,   10, activation='relu'),
                    layer_template(Dense,   "$output_neurons",    activation='$auto_output_activation'),
                ],
            ),
        ),
    ),

)

# -------------------------------------------------------------------------------------------------------------------- #

###
# Гиперпараметры

hp_defaults = to_dict(
        tabs=['learn', 'corel', ],
        # 'corel-scaled', # NOTE: use 'corel-scaled' to compare scaled/unscaled data if something goes wrong
)

data_common_vars=to_dict(
    source          = data,
    val_size        = None,
    test_size       = 0.1,
    seq_len         = 100,
    predict_col     = ['Close'],
    x_scaler        = MinMaxScaler,
    y_scaler        = MinMaxScaler,
    stride          = 1,
    sampling_rate   = 1,
    shuffle         = False,
    batch_size      = 128,
)

hp_template = to_dict(
    **hp_defaults,
    model_vars=to_dict(
        globals_drop_rate = 0.3,
        output_drop_rate = 0.5
    ),
    data_vars={
        **data_common_vars,
        **to_dict(
            predict_range = (1,2),
        )
    },
)


hyper_params_sets = {}

if True:
    # Варианты с предсказанием одного шага
    for i in range(1,11,9):
        hp =copy.deepcopy(hp_template)
        hp['data_vars']['predict_range'] = (i, i+1)
        hyper_params_sets[f"n{i}"] = hp

if True:
    # Варианты с предсказанием серии шагов
    hp =copy.deepcopy(hp_template)
    hp['data_vars']['predict_range'] = (1, 11)
    hp['tabs'] = ['learn', 'c-range', ] # TODO: try 'corel-series'
    hp['metrics'] = [S_MAE]
    hyper_params_sets[f"n-1:10"] = hp

# -------------------------------------------------------------------------------------------------------------------- #

###
# Создать вкладки для вывода результатов

def get_tab(tab_id, model_name, hp_name, hp):
    if hp['data_vars']['predict_range'][1]-hp['data_vars']['predict_range'][0] == 1:
        return tab_id, f"{model_name[:3]}-{hp['data_vars']['predict_range'][0]:02}"
    else:
        return tab_id, f"{model_name[:3]}-{hp['data_vars']['predict_range'][0]:02}:{hp['data_vars']['predict_range'][1]-1:02}"

def get_tab_r(tab_id, model_name, hp_name, hp):
    return   f"{tab_id[:3]}--{model_name[:3]}-{hp['data_vars']['predict_range'][0]:02}:{hp['data_vars']['predict_range'][1]-1:02}", None

from IPython.display import clear_output, display
import ipywidgets as widgets
from functools import lru_cache

model_tabs = {}
tabs_dict = {}
for model_name in models:
    for hp_name, hp in hyper_params_sets.items():
        for tab_id in hp['tabs']:
            if "-range" not in tab_id:
                tab_group, tab_i = get_tab(tab_id, model_name, hp_name, hp)
                if tab_group not in tabs_dict:
                    tabs_dict[tab_group] = {}
                widget = tabs_dict[tab_group][tab_i] = widgets.Output()
                with widget:
                    # По умолчанию заполнить текст вкладок информацией о параметрах модели
                    clear_output()
                    print(f"{model_name}--{hp_name}")
            else:
                tab_group, _ = get_tab_r(tab_id, model_name, hp_name, hp)
                if tab_group not in tabs_dict:
                    tabs_dict[tab_group] = {}
                for i in range(*hp['data_vars']['predict_range']):
                    tab_i = f"{i:02}"
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

# -------------------------------------------------------------------------------------------------------------------- #

tabs_objs.keys()
#display(tabs_objs["xxx"])

# -------------------------------------------------------------------------------------------------------------------- #

###
# Функция подготовки данных для обучения

def prepare_data(
    data,
    split,
    x_cols,
    y_cols,
    data_vars,
):
    """
    Подготовка данных исходя из условий задачи и гиперпараметров
    """
    y_predict_range     = data_vars['predict_range']
    seq_len             = data_vars['seq_len']
    x_scaler_class      = data_vars['x_scaler']
    y_scaler_class      = data_vars['y_scaler']

    # Определение границ подвыборок перед обучением
    train_se, val_se, test_se = train_val_test_boundaries(
        split,
        len(data),
    )
    data_crop = split.y_end_offset  # Размер подрезаемых данных

    # Подготовка scalera на тренировочных данных
    x_rows = [LABEL_IDX[k] for k in x_cols]
    y_rows = [LABEL_IDX[k] for k in y_cols]

    x_data = data[
        :-data_crop,    # x_data будет использоваться совместно с y_samples_data, который будет подрезан
        x_rows]
    y_data = data[:, y_rows]

    x_data_train = x_data[train_se[0]:train_se[1], :]
    y_data_train = y_data[train_se[0]+split.y_start_offset : train_se[1]+split.y_end_offset, :]

    # Нормализация данных
    x_scaler = x_scaler_class()
    x_scaler.fit(x_data_train)
    x_data_scaled = x_scaler.transform(x_data)

    y_scaler = y_scaler_class()
    y_scaler.fit(y_data_train)
    y_data_scaled = y_scaler.transform(y_data)

    # Создание массива выходных данных
    y_idxs = list(range(*y_predict_range))
    if len(y_idxs) < 1:
        raise ValueError(f"Predict range is empty!")
    elif len(y_idxs) == 1:
        y_samples_scaled = [
            y_data_scaled[i + split.y_start_offset, : ]
            for i in range(y_data_scaled.shape[0]-data_crop-1)
        ]
    else:
        y_samples_scaled = []
        for i in range(y_data_scaled.shape[0]-data_crop-1):
            x = y_data_scaled[i + split.y_start_offset : i + split.y_end_offset+1, : ]
            # TODO: take into account y_idxs
            y_samples_scaled.append(x)
    # Сдвиг на 1 элемент чтобы значение с индексом N следовало
    # сразу после последнего предиката из соответствующей входной последовательности (с индексом N-1)
    y_samples_scaled.insert(0, y_samples_scaled[0])
    # Перевод результата в np.array
    y_samples_scaled = np.array(y_samples_scaled)
    y_samples_scaled = np.reshape(y_samples_scaled, y_samples_scaled.shape[:2])

    # Формирование наборов данных
    data_provider = TrainSequenceProvider(
        x_train     = x_data_scaled,
        y_train     = y_samples_scaled,
        x_val       = None,
        y_val       = None,
        x_test      = None,
        y_test      = None,
        split       = split,
        seq_len     = seq_len,
        stride      = data_vars.get('stride', 1),
        sampling_rate = data_vars.get('sampling_rate', 1),
        shuffle     = data_vars.get('shuffle', False),
        batch_size  = data_vars.get('batch_size', ENV[ENV__TRAIN__DEFAULT_BATCH_SIZE]),
    )

    input_shape = (seq_len, *x_data_scaled.shape[1:])
    output_data = {
        "x_data_scaled"     : x_data_scaled,
        "y_data_scaled"     : y_data_scaled,
        "y_samples_scaled"  : y_samples_scaled,
        "train_se"          : train_se,
        "val_se"            : val_se,
        "test_se"           : test_se,
    }
    return data_provider, x_scaler, y_scaler, input_shape, output_data

# -------------------------------------------------------------------------------------------------------------------- #

###
# Функция вывода графиков и характеристик работы модели на тестовой выборке

def print_results(epoch_best, epoch_last, y_ref, y_best, y_last, cg_range, display_range, data_vars, custom_header=None):
    if custom_header is not None:
        for line in custom_header:
            print(line)
    print(f"Корреляция на 'лучшей' эпохе  ({epoch_best}): {correlation(y_ref[:, 0], y_best[:, 0])}")
    print(f"Корреляция на последней эпохе ({epoch_last}): {correlation(y_ref[:, 0], y_last[:, 0])}")
    print("")

    # Подготовка данных для анализа корреляции
    cg_auto = correlation_graph(
        y_ref[:, 0],
        y_ref[:, 0],
        0,
        y_ref.shape[0],
        cg_range
    )

    cg_best = correlation_graph(
        y_ref [:, 0],
        y_best[:, 0],
        0,
        y_ref.shape[0],
        cg_range
    )
    cg_best_peak = np.argmax(cg_best)+cg_range[0]

    cg_last = correlation_graph(
        y_ref [:, 0],
        y_last[:, 0],
        0,
        y_ref.shape[0],
        cg_range
    )
    cg_last_peak = np.argmax(cg_last)+cg_range[0]


    print(f"Пик корреляции на 'лучшей' эпохе  ({epoch_best}): {cg_best_peak}")
    print(f"Пик корреляции на последней эпохе ({epoch_last}): {cg_last_peak}")
    print("")

    # Графики значений исходных / предсказанных
    graph_data = [
            y_ref [:, 0],
            y_best[:, 0],
            y_last[:, 0],
    ]
    graph_def = GraphDef(
        idx_label = {
            0 : "Исходные данные",
            1 : f"Предсказание на 'лучшей' эпохе (({epoch_best}))'",
            2 : f"Предсказание на последней эпохе (({epoch_last}))"
        },
        title   = 'Сравнение предсказаний с реальными данными',
        y_label = data_vars['predict_col'],
        x_label = None,
        plot_f  = plot_sequences,
    )
    if epoch_last == epoch_best:
        del graph_def.idx_label[2]
    fig, subplots = plt.subplots(1, 1, figsize=(15,5))
    plot_graph(
        subplots,
        graph_data,
        graph_def,
        *display_range,
    )
    # Регулировка пределов оси x
    plt.xlim(0, display_range[1]-display_range[0])
    # Фиксация графика
    plt.show()


    # Графики корреляции (отрисовка)
    cg_x = list(range(cg_range[0], cg_range[1]+1))
    graph_data = [
            [cg_x, cg_auto],
            [cg_x, cg_best],
            [cg_x, cg_last],
    ]
    graph_def = GraphDef(
        idx_label = {
            0 : "Автокорреляция исходных данных",
            1 : f"Корреляция на 'лучшей' эпохе ({epoch_best})",
            2 : f"Корреляция на последней эпохе ({epoch_last})"
        },
        title   = 'Корреляция',
        y_label = data_vars['predict_col'],
        x_label = None,
        plot_f  = plot_xy,
    )
    if epoch_last == epoch_best:
        del graph_def.idx_label[2]
    fig, subplots = plt.subplots(1, 1, figsize=(15,3))
    plot_graph(
        subplots,
        graph_data,
        graph_def,
        *display_range,
    )
    # # Регулировка пределов оси x
    plt.xlim(cg_range[0], cg_range[1]+1)
    # Фиксация графика
    plt.show()

# -------------------------------------------------------------------------------------------------------------------- #

###
# Функция вывода корреляции для серии предсказаний

def print_series_result(epoch_best, epoch_last, y_ref, y_best, y_last, cg_range, display_range, data_vars):

    # Подготовка данных для анализа корреляции
    cg_auto = correlation_graph(
        y_ref,
        y_ref,
        0,
        y_ref.shape[0],
        cg_range,
        method=S_COREL_METHOD__SCIPY_CORRELATE,
    )

    cg_best = correlation_graph(
        y_ref,
        y_best,
        0,
        y_ref.shape[0],
        cg_range,
        method=S_COREL_METHOD__SCIPY_CORRELATE,
    )
    cg_best_peak = np.argmax(cg_best)+cg_range[0]

    cg_last = correlation_graph(
        y_ref,
        y_last,
        0,
        y_ref.shape[0],
        cg_range,
        method=S_COREL_METHOD__SCIPY_CORRELATE,
    )
    cg_last_peak = np.argmax(cg_last)+cg_range[0]

    print(f"Пик корреляции на 'лучшей' эпохе  ({epoch_best}): {cg_best_peak}")
    print(f"Пик корреляции на последней эпохе ({epoch_last}): {cg_last_peak}")
    print("")

    # Графики корреляции (отрисовка)
    cg_x = list(range(cg_range[0], cg_range[1]+1))
    # TODO: split cg_auto/cg_best/cg_last per dimension
    graph_data = [
            [cg_x, cg_auto],
            [cg_x, cg_best],
            [cg_x, cg_last],
    ]
    graph_def = GraphDef(
        idx_label = {
            0 : "Автокорреляция исходных данных",
            1 : f"Корреляция на 'лучшей' эпохе ({epoch_best})",
            2 : f"Корреляция на последней эпохе ({epoch_last})"
        },
        title   = 'Корреляция',
        y_label = data_vars['predict_col'],
        x_label = None,
        plot_f  = plot_xy,
    )
    if epoch_last == epoch_best:
        del graph_def.idx_label[2]
    fig, subplots = plt.subplots(1, 1, figsize=(10,6))
    plot_graph(
        subplots,
        graph_data,
        graph_def,
        *display_range,
    )
    # # Регулировка пределов оси x
    plt.xlim(cg_range[0], cg_range[1]+1)
    # Фиксация графика
    plt.show()

# -------------------------------------------------------------------------------------------------------------------- #

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
            data = data_vars['source']

            split = SplitSequenceDef(
                val_size    = data_vars['val_size'],
                test_size   = data_vars['test_size'],
                margin      = data_vars['seq_len']*2,
                y_start_offset= data_vars['predict_range'][0],
                y_end_offset  = data_vars['predict_range'][1]-1
            )

            data_provider, x_scaler, y_scaler, input_shape, prepare_aux = prepare_data(
                data    = data,
                split   = split,
                x_cols  = list(LABEL_IDX.keys()),
                y_cols  = data_vars['predict_col'],
                data_vars = data_vars,
            )

            model_vars = copy.deepcopy(hp['model_vars'])
            model_vars['branch_general_input_shape'] = input_shape
            model_vars['output_neurons'] = data_vars['predict_range'][1] - data_vars['predict_range'][0]    # TODO: there could be 3rd arg in range, take it into account
            if isinstance(y_scaler, MinMaxScaler):
                model_vars['auto_output_activation'] = 'relu'
                # TODO: also it could be sigmoid
                # TODO: also it could be tanh in case if MinMax output range is [-1:1]
            elif isinstance(y_scaler, StandardScaler):
                model_vars['auto_output_activation'] = 'linear'
            else:
                raise ValueError("Can't determine 'auto_output_activation' for used Y scaler")

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
            )

            def on_model_update(thd):
                pass

            thd = TrainHandler(
                data_path       = ENV[ENV__MODEL__DATA_ROOT] / model_data.get("data_path", ENV[ENV__TRAIN__DEFAULT_DATA_PATH]),
                data_name       = run_name,
                mhd             = mhd,
                mhd_class       = ModelHandler,
                on_model_update = on_model_update
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
            pred_last_scaled = mhd.context.test_pred
            pred_last = y_scaler.inverse_transform(pred_last_scaled)
            epoch_best = mhd.context.epoch

            thd.load_best()
            mhd.update_data(force=True)
            pred_best_scaled = mhd.context.test_pred
            pred_best = y_scaler.inverse_transform(pred_best_scaled)

            mhd.context.report_history = full_history

            # Воссоздать исходные данные, с которыми сравнивать результаты предсказаний
            y_test_samples = y_scaler.inverse_transform(data_provider.y_test)

            # Вывод результатов во вкладки
            display_range = (0, 501)    # Диапазон по X для графиков значений
            cg_range = (-10, 110)       # Диапазон для графиков корреляции
            assert y_test_samples.shape == pred_best.shape, "Something went wrong!"
            assert y_test_samples.shape == pred_last.shape, "Something went wrong!"

            with dummy_output:
                plt.show()
                clear_output()

            for tab_id in hp['tabs']:
                if "-range" not in tab_id:
                    # Вывод основной информации
                    # Вывод графиков предсказаний и корреляции если предсказания были только на один шаг
                    tab_group, tab_i = get_tab(tab_id, model_name, hp_name, hp)
                    with tabs_dict[tab_group][tab_i]:
                        clear_output()
                        print(f"Модель         : {hp['model']}")
                        print(f"Гиперпараметры : {hp_name}")
                        print("")
                        if "learn" in tab_id:
                            mhd.context.report_to_screen()
                            mhd.model.summary()

                            utils.plot_model(mhd.model, dpi=60)
                            plt.show()
                        elif "corel" in tab_id:
                            if "-scaled" in tab_id:
                                y_ref  = data_provider.y_test
                                y_best = pred_best_scaled
                                y_last = pred_last_scaled
                            else:
                                y_ref  = y_test_samples
                                y_best = pred_best
                                y_last = pred_last
                            if "-series" in tab_id:
                                print_series_result(epoch_best, epoch_last, y_ref, y_best, y_last, cg_range, display_range, data_vars)
                            else:
                                print_results(epoch_best, epoch_last, y_ref, y_best, y_last, cg_range, display_range, data_vars)
                else:
                    # Вывод графиков предсказаний и корреляции если предсказания были на несколько шагов
                    tab_group, _ = get_tab_r(tab_id, model_name, hp_name, hp)
                    j = -1
                    for i in range(*hp['data_vars']['predict_range']):
                        tab_i = f"{i:02}"
                        j += 1
                        custom_header=[
                            f"Предсказание на {j} шаг вперёд",
                            ""
                        ]
                        with tabs_dict[tab_group][tab_i]:
                            y_ref  = y_test_samples [:,j:j+1]
                            y_best = pred_best      [:,j:j+1]
                            y_last = pred_last      [:,j:j+1]
                            print_results(epoch_best, epoch_last, y_ref, y_best, y_last, cg_range, display_range, data_vars, custom_header)

                with dummy_output:
                    plt.show()
                    clear_output()

            mhd.context.report_history = None
            mhd.unload_model()

        main_logic(model_name, hp_name)
