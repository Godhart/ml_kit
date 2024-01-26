###
# Решение задачи

try:
    from ml_kit.src.standalone import STANDALONE
except ImportError:
    STANDALONE = False

if STANDALONE:
    from ml_kit.src.helpers import *
    from ml_kit.src.trainer_common import *
    from ml_kit.src.trainer_texts import *
    
# Класс для конструирования последовательной модели нейронной сети
from tensorflow.keras.models import Model, Sequential

# Основные слои
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D

###

ENV[ENV__DEBUG_PRINT] = True

###
if not STANDALONE:
    # Подключение google диска если работа в ноутбуке
    connect_gdrive()

    # А так же директива для отрисовки matplot в вывод ноутбука
    # %matplotlib inline

###

# Перегрузка констант под текущую задачу

ENV[ENV__TEXTS__BOW_USED] = False

ENV[ENV__TRAIN__DEFAULT_DATA_PATH]       = "lesson_6_lite_1"
ENV[ENV__TRAIN__DEFAULT_LOSS]            = "categorical_crossentropy"
ENV[ENV__TRAIN__DEFAULT_BATCH_SIZE]      = 512
ENV[ENV__TRAIN__DEFAULT_EPOCHS]          = 100
ENV[ENV__TRAIN__DEFAULT_TARGET_ACCURACY] = 1.0   # максимум чтобы каждая сеть прошла все эпохи обучения
ENV[ENV__TRAIN__DEFAULT_SAVE_STEP]       = 10
ENV[ENV__TRAIN__DEFAULT_FROM_SCRATCH]    = None

ENV[ENV__TEXTS__TRAIN_TEXTS_PATH]        = "writers"
ENV[ENV__TEXTS__TRAIN_TEXTS_NAME_REGEX]  = ('\((.+)\) (\S+)_', 0, 1)
# regex определяет имена файлов, которые будут загружены для обучения
#   индексы после regex задают номер capture group для:
#   - первый индекс - для имени класса
#   - второй индекс - для названия набора выборки

ENV[ENV__TEXTS__TRAIN_TEXTS_SUBSETS]     = ('обучающая', 'тестовая')
# определяет как поименованы файлы для (какое содержат слово)
# - первый элемент - для обучающей выборки
# - второй элемент - для тестовой выборки
###

# -------------------------------------------------------------------------------------------------------------------- #

# Загрузка текстов
if True or STANDALONE:
    texts = load_texts_from_dir(ENV[ENV__TEXTS__TRAIN_TEXTS_PATH], ENV[ENV__TEXTS__TRAIN_TEXTS_NAME_REGEX], ENV[ENV__TEXTS__TRAIN_TEXTS_SUBSETS])
# т.к. тексты уже загружены - привести к нужному формату
else:
    texts = TrainTexts(CLASS_LIST, text_train, text_test)

# Шаблоны моделей для обучения
models = to_dict(
    conv1 = to_dict(
        name            = "conv1",
        model_class     = Sequential,
        template        = [
            # Список слоев и их параметров
            layer_template(Embedding,           '$vocab_size', 10, input_length='$chunk_size'),
            layer_template(SpatialDropout1D,    0.2),
            layer_template(BatchNormalization),
            # Два слоя одномерной свертки Conv1D
            layer_template(Conv1D,              20, 5, activation='relu', padding='same'),
            layer_template(Conv1D,              20, 5, activation='relu'),
            # Слой подвыборки/пулинга с функцией максимума
            layer_template(MaxPooling1D,        2),
            layer_template(Dropout,             0.2),
            layer_template(BatchNormalization),
            layer_template(Flatten,),
            layer_template(Dense,               '$classes_count', activation='softmax'),
        ]
    )
)

hyper_params_sets = (
    to_dict(v=to_dict(vocab_size = 20_000, chunk_size = 1_000, chunk_step = 100,), tabs=["vs:20000","cs:1000"]),
    to_dict(v=to_dict(vocab_size =  5_000, chunk_size = 1_000, chunk_step = 100,), tabs=["vs:5000"]),
    to_dict(v=to_dict(vocab_size = 10_000, chunk_size = 1_000, chunk_step = 100,), tabs=["vs:10000"]),
    to_dict(v=to_dict(vocab_size = 40_000, chunk_size = 1_000, chunk_step = 100,), tabs=["vs:40000"]),
    to_dict(v=to_dict(vocab_size = 20_000, chunk_size =   500, chunk_step =  50,), tabs=["cs:500"]),
    to_dict(v=to_dict(vocab_size = 20_000, chunk_size = 2_000, chunk_step = 200,), tabs=["cs:2000"]),
)

# -------------------------------------------------------------------------------------------------------------------- #

# Создать вкладки для вывода результатов
from IPython.display import clear_output, display
import ipywidgets as widgets
from functools import lru_cache

tabs_dict = {}
for k in models:
    for hyper_params in hyper_params_sets:
        for tab_id in hyper_params['tabs']:
            tab_group, tab_i = tab_id.split(":")
            tab_group = f"{k}-"+tab_group
            if tab_group not in tabs_dict:
                tabs_dict[tab_group] = {}
            widget = tabs_dict[tab_group][tab_i] = widgets.Output()
            with widget:
                # По умолчанию заполнить текст вкладок информацией о параметрах модели
                clear_output()
                print(f"{k}--{hyper_params}")

tabs_objs = {k: widgets.Tab() for k in tabs_dict}
for k, v in tabs_dict.items():
    sorted_tab_items_keys = list(sorted(v.keys(), key=lambda x: int(x)))
    tabs_objs[k].children = [v[kk] for kk in sorted_tab_items_keys]
    for i in range(0, len(sorted_tab_items_keys)):
        tabs_objs[k].set_title(i, f"{k}:{sorted_tab_items_keys[i]}")


# Вывести вкладки с результатами
# -------------------------------------------------------------------------------------------------------------------- #
tabs_objs.keys()
# -------------------------------------------------------------------------------------------------------------------- #
display(tabs_objs["conv1-vs"])
# -------------------------------------------------------------------------------------------------------------------- #
display(tabs_objs["conv1-cs"])
# -------------------------------------------------------------------------------------------------------------------- #

dummy_output = widgets.Output()

# Обучение сети с нужными параметрами и сохранение результатов на диск
for hyper_params in hyper_params_sets:
    with dummy_output:
        clear_output()

    # Подготовить данные в соответствии с параметрами
    text_train_data = TextTrainDataProvider(
        texts,
        bow_used=ENV[ENV__TEXTS__BOW_USED],
        debug=ENV[ENV__DEBUG_PRINT],
        **hyper_params['v'],
    )

    # Функция, которая донастраивает модели из шаблона под текущие параметры
    def tune_model(model, hyper_params):
        tuned_model = copy.deepcopy(model)
        tuned_model['name'] += f"--vocab_{hyper_params['v']['vocab_size']}--chunk_{hyper_params['v']['chunk_size']}_{hyper_params['v']['chunk_step']}"
        return tuned_model

    # Обучить модели под текущие параметры
    # (или загрузить сохраненные результаты если таковые будут)
    for k, v in models.items():
        # Обучаться будут по одной модели за раз
        train_models = [
            tune_model(v, hyper_params),
        ]

        def display_callback(message):
            for tab_id in hyper_params['tabs']:
                tab_group, tab_i = tab_id.split(":")
                tab_group = f"{k}-"+tab_group
                with tabs_dict[tab_group][tab_i]:
                    # Вывести во вкладку информацию об обучении
                    print(message)

        variables = to_dict(
            classes_count   = len(text_train_data.classes),
            vocab_size      = text_train_data.vocab_size,

            chunk_size      = hyper_params['v']['chunk_size'],
        )

        results = text_train__all_together(
            train_data=text_train_data,
            models=train_models,
            variables=variables,
            display_callback=display_callback,
        )

        # Вывод результатов для сравнения
        for tab_id in hyper_params['tabs']:
            tab_group, tab_i = tab_id.split(":")
            tab_group = f"{k}-"+tab_group
            thd = results[0]
            full_history = copy.deepcopy(thd.mhd.context.history)
            thd.load_best()
            thd.mhd.context.report_history = full_history
            with tabs_dict[tab_group][tab_i]:
                clear_output()
                thd.mhd.context.report_to_screen()
                thd.mhd.model.summary()
            thd.mhd.context.report_history = None
            with dummy_output:
                clear_output()
