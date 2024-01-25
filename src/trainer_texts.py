###
# Text Trainer functions

try:
    from standalone import STANDALONE
except ImportError:
    STANDALONE = False

if STANDALONE:
    from env import *
    from helpers import *
    from trainer_common import *

###
# TODO: redefine in main section as necessary
BOW_USED = False

DEFAULT_DATA_PATH       = "models_storage"
DEFAULT_OPTIMIZER       = "rmsprop" # rmsprop is better for ?texts/rnn? than adam
DEFAULT_LOSS            = "categorical_crossentropy"
DEFAULT_BATCH_SIZE      = 10
DEFAULT_EPOCHS          = 50
DEFAULT_TARGET_ACCURACY = 1.0
DEFAULT_SAVE_STEP       = 10
DEFAULT_FROM_SCRATCH    = None

TOKENIZER_FILTERS       = '!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff'
TRAIN_TEXTS_PATH        = "train_texts"
TRAIN_TEXTS_NAME_REGEX  = ('\((.+)\) (\S+)_', 0, 1)
# regex определяет имена файлов, которые будут загружены для обучения
#   индексы после regex задают номер capture group для:
#   - первый индекс - для имени класса
#   - второй индекс - для названия набора выборки

TRAIN_TEXTS_SUBSETS     = ('обучающая', 'тестовая')
# определяет как поименованы файлы для (какое содержат слово)
# - первый элемент - для обучающей выборки
# - второй элемент - для тестовой выборки
###

# Работа с массивами данных
import numpy as np

# Функции операционной системы
import os

# Регулярные выражения
import re

# Работа с путями в файловой системе
from pathlib import Path

# Функции-утилиты для работы с категориальными данными
from tensorflow.keras import utils

# Токенизатор для преобразование текстов в последовательности
from tensorflow.keras.preprocessing.text import Tokenizer


################################################################################
#

class TrainTexts:
    """
    Содержит тексты для обучения и информацию о них
        classes_list    : список классов,
        text_train      : список списков обучающей выборки.
                            1-е измерение - индекс класса,
                            2-е измерение - в каждом элементе данные одного из файлов
        text_test       : список списков тестовой выборки.
                            1-е измерение - индекс класса,
                            2-е измерение - в каждом элементе данные одного из файлов
    """

    def __init__(self, classes, train, test):
        self.classes    = classes
        self.train      = train
        self.test       = test

    @property
    def classes_labels(self):
        return [item[1] for item in self.classes]

    @property
    def classes_ids(self):
        return [item[0] for item in self.classes]


def load_texts_from_dir(path, regex=('\((.+)\) (\S+)_', 0, 1), subsets=('обучающая', 'тестовая')):
    """
    Загружает текстовые данные из папки, имена которых соответствуют рег.выражению regex[0]

    индексы в regex задают номер capture group для:
    - первый индекс - для имени класса
    - второй индекс - для названия набора выборки

    subsets задаёт названия наборов для обучающей и тестовой выборки

    Возвращает TrainTexts
    """
    path = Path(path)
    classes_list = []
    text_train = []
    text_test = []

    for file_name in os.listdir(path):
        if not (path / file_name):
            continue
        # Выделение имени класса и типа выборки из имени файла
        m = re.match(regex[0], file_name)
        # Если выделение получилось, то файл обрабатывается
        if m:
            class_name = m.groups()[regex[1]].lower()
            subset_name = m.groups()[regex[2]].lower()

            # Проверка типа выборки в имени файла
            is_train = subsets[0] in subset_name
            is_test = subsets[1] in subset_name

            if not is_train and not is_test:
                continue

            # Добавление нового класса, если его еще нет в списке
            if class_name not in classes_list:
                print(f'Добавление класса "{class_name}"')
                classes_list.append(class_name)
                # Инициализация соответствующих классу строк текста
                text_train.append([])
                text_test.append([])

            # Поиск индекса класса для добавления содержимого файла в выборку
            cls = classes_list.index(class_name)
            print(f'Добавление файла "{file_name}" в класс "{classes_list[cls]}", {subset_name} выборка.')
            with open(path / file_name, 'r') as f:
                # Загрузка содержимого файла в строку
                text = f.read()
            # Определение выборки, куда будет добавлено содержимое
            subset = text_train if is_train else text_test
            # Добавление текста к соответствующей выборке класса. Концы строк заменяются на пробел
            subset[cls].append(re.subn(r"\s+", " ", text.strip())[0])

    classes = []
    for i in range(len(classes_list)):
        classes.append((i, classes_list[i]))

    result = TrainTexts(
        classes = classes,
        train   = [flat_text(item) for item in text_train],
        test    = [flat_text(item) for item in text_test],
    )
    return result


def flat_text(text):
    if isinstance(text, (tuple, list)):
        return " ".join([flat_text(item) for item in text])
    return text


def prepare_tokenizer(text, vocab_size, oov_token='_oov_'):
    """
    Создаёт токены на базе текстов
    """
    tokenizer = Tokenizer(num_words=vocab_size, filters=TOKENIZER_FILTERS,
                        lower=True, split=' ', oov_token=oov_token, char_level=False)
    tokenizer.fit_on_texts(text)
    return tokenizer


def words_to_tokens(texts, tokenizer):
    return tokenizer.texts_to_sequences(texts)


def tokens_to_bow(sequences, tokenizer):
    return tokenizer.sequences_to_matrix(sequences)


def texts_stats(texts, classes, tokens=None):
    """
    Вывод статистики по списку текстов

    texts   - список текстов
    classes - (опционально) словарь идентификаторов классов для текстов. ключ - индекс класса, значение - метка класса
              Если None то формируется автоматом из предположения один текст - один класс
    tokens  - (опционально) список векторов токенов, представляющих соответствующий текст
    """
    if classes is None:
        classes = [(i, str(i)) for i in range(0, len(texts))]
    chars_total = 0
    words_total = 0
    tokens_total = 0
    result = {}
    for i in range(0, len(texts)):
        class_idx, class_label = classes[i]
        chars = sum(len(word) for word in texts[i])
        words = len(texts[i])
        chars_total += chars
        words_total += words
        result[class_label] = {
            'class_idx' : class_idx,
            'chars'     : chars,
            'words'     : words
        }
        if tokens is not None:
            tokens = len(tokens[i])
            tokens_total += tokens
            result[class_label]['tokens'] = tokens
    result['Суммарно'] = {
        'class_idx'     : None,
        'chars'         : chars_total,
        'words'         : words_total,
    }
    if tokens is not None:
        result['Суммарно']['tokens'] = tokens_total
    return result


def prepare_long_texts(texts, classes, chunk_size, chunks_step):
    """
    Функция подготовки длинных текстов в данные для обучения
    формирует выборку отрезков из текстов и соответствующих им меток классов в виде one hot encoding

    texts   - список текстов
    classes - список идентификаторов классов для текстов в виде пар: индекс класса, метка класса
              Если None то формируется автоматом из предположения один текст - один класс
    """
    if classes is None:
        classes = [(i, str(i)) for i in range(0, len(texts))]

    class_count = max(class_info[0] for class_info in classes)+1
    print(f"class_count={class_count}")

    # Списки для исходных векторов и категориальных меток класса
    x, y = [], []

    # Для каждого класса:
    print(f"texts={len(texts)}")
    for i in range(0, len(texts)):
        # Разбиение длинного текста на куски (с перекрытием если chunk_step < chunk_size)
        chunks = chop_list_by_sliding_window(texts[i], chunk_size, chunks_step)
        # Добавление отрезков в выборку
        x += chunks
        # Для всех отрезков класса cls добавление меток класса в виде OHE
        y += [utils.to_categorical(classes[i][0], class_count)] * len(chunks)

    # Возврат результатов как numpy-массивов
    return np.array(x), np.array(y)


class TextTrainDataProvider(TrainDataProvider):
    """
    Помогатор обучения на длинных текстах
    Подготавливает предварительно загруженные тексты для обучения
    - токенизирует
    - разбивает большие тексты на куски заданного размер используя скользящее окно с заданным шагом
    - опционально создает BOW
    """

    def __init__(
        self, texts:TrainTexts, vocab_size:int, chunk_size:int, chunk_step:int, bow_used:bool=False, debug:bool=False,
        **ignored_args
    ):
        self._texts = texts
        self._vocab_size = vocab_size
        self._chunk_size = chunk_size
        self._chunk_step = chunk_step
        self._bow_used = bow_used
        self._debug = debug
        self._prepare()

    @property
    def classes(self):
        return self._texts.classes

    @property
    def classes_labels(self):
        return self._texts.classes_labels

    @property
    def classes_ids(self):
        return self._texts.classes_ids

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def x_val(self):
        # TODO: now test and validation data are the same
        return self.x_test

    @property
    def x_val_bow(self):
        # TODO: now test and validation data are the same
        return self.x_test_bow

    @property
    def y_val(self):
        # TODO: now test and validation data are the same
        return self.y_test

    def _prepare(self):
        self._tokenizer = prepare_tokenizer(self._texts.train, self._vocab_size)

        if self._debug:
            # Построение словаря в виде пар слово - индекс
            items = list(self._tokenizer.word_index.items())

            # Вывод нескольких наиболее часто встречающихся слов
            print(f"Первые 120 элементов словаря: {items[:120]}")

            # Размер словаря может быть больше, чем num_words, но при преобразовании в последовательности
            # и векторы bag of words будут учтены только первые num_words слов
            print("Размер словаря", len(items))

        self._seq_train = words_to_tokens(self._texts.train, self._tokenizer)
        self._seq_test  = words_to_tokens(self._texts.test,  self._tokenizer)

        if self._debug:
            print("Фрагмент обучающего текста:")
            print("В виде оригинального текста:              ", self._texts.train[1][:101])
            print("Он же в виде последовательности индексов: ", self._seq_train[1][:20])

        with timex("Преобразование текста в обучающие последовательности"):
            self.x_train, self.y_train = prepare_long_texts(
                self._seq_train, self._texts.classes, self._chunk_size, self._chunk_step)

            self.x_test, self.y_test = prepare_long_texts(
                self._seq_test, self._texts.classes, self._chunk_size, self._chunk_step)

        if self._debug:
            # Проверка формы сформированных данных
            print("x_train.shape=", self.x_train.shape, " y_train.shape=", self.y_train.shape)
            print("x_test.shape =", self.x_test.shape,  " y_test.shape =", self.y_test.shape)
            # Вывод отрезка индексов тренировочной выборки
            print("Отрезок индексов тренировочной выборки", self.x_train[0])

        if self._bow_used:
            with timex("Преобразование последовательностей в bow"):
                # На входе .sequences_to_matrix() ожидает список, .tolist() выполняет преобразование типа
                self.x_train_bow = tokens_to_bow(self.x_train.tolist(), self._tokenizer)
                self.x_test_bow  = tokens_to_bow(self.x_test.tolist(), self._tokenizer)

            if self._debug:
                # Вывод формы обучающей выборки в виде разреженной матрицы Bag of Words
                print("x_train_bow.shape=", self.x_train_bow.shape)
                # Вывод фрагмента отрезка обучающего текста в виде Bag of Words
                print("Фрагмент отрезка обучающего текста в виде Bag of Words", self.x_train_bow[0][0:100])
            else:
                self.x_train_bow = None
                self.x_test_bow = None

    def bow_all_as_tuple(self):
        if not self._bow_used:
            raise ValueError("No BOW data!")
        return self.x_train_bow, self.y_train, self.x_val_bow, self.y_val, self.x_test_bow, self.y_test

    def bow_train_val_as_tuple(self):
        if not self._bow_used:
            raise ValueError("No BOW data!")
        return self.x_train_bow, self.y_train, self.x_val_bow, self.y_val

    def bow_train_as_tuple(self):
        if not self._bow_used:
            raise ValueError("No BOW data!")
        return self.x_train_bow, self.y_train

    def bow_val_as_tuple(self):
        if not self._bow_used:
            raise ValueError("No BOW data!")
        return self.x_val_bow, self.y_val

    def bow_test_as_tuple(self):
        if not self._bow_used:
            raise ValueError("No BOW data!")
        return self.x_test_bow, self.y_test

    def bow_all_as_dict(self):
        if not self._bow_used:
            raise ValueError("No BOW data!")
        return {
            "x_train"   : self.x_train_bow,
            "y_train"   : self.y_train,
            "x_val"     : self.x_val_bow,
            "y_val"     : self.y_val,
            "x_test"    : self.x_test_bow,
            "y_test"    : self.y_test,
        }

    def bow_all_as_dict(self):
        if not self._bow_used:
            raise ValueError("No BOW data!")
        return {
            "x_train"   : self.x_train_bow,
            "y_train"   : self.y_train,
            "x_val"     : self.x_val_bow,
            "y_val"     : self.y_val,
            "x_test"    : self.x_test_bow,
            "y_test"    : self.y_test,
        }

    def bow_tain_val_as_dict(self):
        if not self._bow_used:
            raise ValueError("No BOW data!")
        return {
            "x_train"   : self.x_train_bow,
            "y_train"   : self.y_train,
            "x_val"     : self.x_val_bow,
            "y_val"     : self.y_val,
        }

    def bow_train_as_dict(self):
        if not self._bow_used:
            raise ValueError("No BOW data!")
        return {
            "x_train"   : self.x_train_bow,
            "y_train"   : self.y_train,
        }

    def bow_val_as_dict(self):
        if not self._bow_used:
            raise ValueError("No BOW data!")
        return {
            "x_val"     : self.x_val_bow,
            "y_val"     : self.y_val,
        }

    def bow_test_as_dict(self):
        if not self._bow_used:
            raise ValueError("No BOW data!")
        return {
            "x_test"    : self.x_test_bow,
            "y_test"    : self.y_test,
        }


def text_train__all_together(
    train_data : TextTrainDataProvider,
    models,
    variables=None,
    display_callback=print,
    unload_models=True,
):

    if variables is None:
        variables = {}

    result = []
    for model_data in models:
        mhd = ClassClassifierHandler(
            name            = model_data['name'],
            model_class     = model_data['class'],
            optimizer       = model_data.get('optimizer', DEFAULT_OPTIMIZER),
            loss            = model_data.get('loss',      DEFAULT_LOSS),
            model_template  = model_data['template'],
            model_variables = variables,
            class_labels    = train_data.classes_labels,
            batch_size      = model_data.get('batch_size',DEFAULT_BATCH_SIZE),
            **train_data.all_as_dict()
        )
        thd = TrainHandler(
            data_path       = ENV[S_MODELS_ROOT] / model_data.get("data_path", DEFAULT_DATA_PATH),
            data_name       = model_data['name'],
            mhd             = mhd,
            mhd_class       = ClassClassifierHandler,
        )
        thd.train(
            from_scratch    = model_data.get("from_scratch", DEFAULT_FROM_SCRATCH),
            epochs          = model_data.get("epochs", DEFAULT_EPOCHS),
            target_accuracy = model_data.get("target_accuracy", DEFAULT_TARGET_ACCURACY),
            save_step       = model_data.get("save_step", DEFAULT_SAVE_STEP),
            display_callback= display_callback
        )
        if unload_models:
            mhd.unload_model()
        result.append(thd)
    return result

###
# Usage example
if STANDALONE:
    if False and __name__ == "__main__":
        texts = load_texts_from_dir(TRAIN_TEXTS_PATH, TRAIN_TEXTS_NAME_REGEX, TRAIN_TEXTS_SUBSETS)
        text_train_data = TextTrainDataProvider(
            texts,
            vocab_size,
            chunk_size,
            chunk_step,
            bow_used=BOW_USED,
            debug=ENV[ENV__DEBUG_PRINT],
        )
