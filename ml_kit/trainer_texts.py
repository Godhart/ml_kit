###
# Text Trainer functions
import sys
from pathlib import Path

ml_kit_path = str((Path(__file__).absolute() / ".." / "..").resolve())
if ml_kit_path not in sys.path:
    sys.path.insert(0, ml_kit_path)

try:
    from ml_kit.standalone import STANDALONE
except ImportError:
    STANDALONE = False

if STANDALONE:
    from ml_kit.env import *
    from ml_kit.helpers import *
    from ml_kit.trainer_common import *

###
ENV__TEXTS__BOW_USED                = "ENV__TEXTS__BOW_USED"
ENV__TEXTS__TOKENIZER_FILTERS       = "ENV__TEXTS__TOKENIZER_FILTERS"
ENV__TEXTS__TRAIN_TEXTS_PATH        = "ENV__TEXTS__TRAIN_TEXTS_PATH"
ENV__TEXTS__TRAIN_TEXTS_NAME_REGEX  = "ENV__TEXTS__TRAIN_TEXTS_NAME_REGEX"
ENV__TEXTS__TRAIN_TEXTS_SUBSETS     = "ENV__TEXTS__TRAIN_TEXTS_SUBSETS"


ENV[ENV__TRAIN__DEFAULT_DATA_PATH]       = "models_storage"
ENV[ENV__TRAIN__DEFAULT_OPTIMIZER]       = "rmsprop" # rmsprop is better for ?texts/rnn? than adam
ENV[ENV__TRAIN__DEFAULT_LOSS]            = "categorical_crossentropy"
ENV[ENV__TRAIN__DEFAULT_METRICS]         = [S_ACCURACY]
ENV[ENV__TRAIN__DEFAULT_BATCH_SIZE]      = 10
ENV[ENV__TRAIN__DEFAULT_EPOCHS]          = 50
ENV[ENV__TRAIN__DEFAULT_TARGET]          = {S_ACCURACY:1.0}
ENV[ENV__TRAIN__DEFAULT_SAVE_STEP]       = 10
ENV[ENV__TRAIN__DEFAULT_FROM_SCRATCH]    = None

ENV[ENV__TEXTS__BOW_USED] = False
ENV[ENV__TEXTS__TOKENIZER_FILTERS]       = '!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff'
ENV[ENV__TEXTS__TRAIN_TEXTS_PATH]        = "train_texts"
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

class TrainTextsClassified:
    """
    Содержит тексты для обучения и информацию о них
        classes_list    : список классов состоящий из пар значений,
                            1-е значение - индекс класса
                            2-е значение - метка класса
        text_train      : список текстов обучающей выборки.
                            в каждом элементе данные для соответствующего класса
        text_test       : список текстов тестовой выборки.
                            в каждом элементе данные для соответствующего класса
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

    Возвращает TrainTextsClassified
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

    result = TrainTextsClassified(
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
    tokenizer = Tokenizer(num_words=vocab_size, filters=ENV[ENV__TEXTS__TOKENIZER_FILTERS],
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


def prepare_texts_dummy(x_sequences, y_data):
    # TODO: x_sequences to np.array()
    return x_sequences, y_data


class TextTrainDataProvider(TrainDataProvider):
    """
    Помогатор обучения на длинных текстах
    Подготавливает предварительно загруженные тексты для обучения
    - токенизирует
    - разбивает большие тексты на куски заданного размер используя скользящее окно с заданным шагом
    - опционально создает BOW
    """

    def __init__(
        self,
        texts: TrainDataProvider | TrainTextsClassified,
        vocab_size:int,
        chunk_size:int,     # NOTE: used when text_prepare_function=prepare_long_texts
        chunk_step:int,     # NOTE: used when text_prepare_function=prepare_long_texts
        seq_used:bool=True,
        bow_used:bool=False,
        bow_default:bool=False,
        debug:bool=False,
        text_prepare_function=None,
    ):
        super(TextTrainDataProvider, self).__init__(
            x_train = None,
            y_train = None,
            x_val   = None,
            y_val   = None,
            x_test  = None,
            y_test  = None,
        )
        if not isinstance(texts, (TrainDataProvider, TrainTextsClassified)):
            raise ValueError("'texts' should be an instance of TrainDataProvider, TrainTextsClassified!")
        self._texts = texts
        self._vocab_size = vocab_size
        self._chunk_size = chunk_size
        self._chunk_step = chunk_step
        self._seq_used = seq_used
        self._bow_used = bow_used or bow_default or not seq_used
        self._bow_default = bow_default or not seq_used
        self._debug = debug
        if text_prepare_function is not None:
            self._text_prepare_function = text_prepare_function
        else:
            if isinstance(texts, TrainTextsClassified):
                self._text_prepare_function = prepare_long_texts
            else:
                raise ValueError("Define 'text_prepare_function' as it can't determined automatically q")

        self._prepare()

    @property
    def x_order(self):
        return None

    @x_order.setter
    def x_order(self, value):
        if value is not None:
            raise ValueError("'x_order' other than None is not supported by TextTrainDataProvider!")

    @property
    def y_order(self):
        return None

    @x_order.setter
    def y_order(self, value):
        if value is not None:
            raise ValueError("'y_order' other than None is not supported by TextTrainDataProvider!")

    @property
    def classes(self):
        if isinstance(self._texts, TrainTextsClassified):
            return self._texts.classes
        else:
            return None

    @property
    def classes_labels(self):
        if isinstance(self._texts, TrainTextsClassified):
            return self._texts.classes_labels
        else:
            return None

    @property
    def classes_ids(self):
        if isinstance(self._texts, TrainTextsClassified):
            return self._texts.classes_ids
        else:
            return None

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def tokenizer(self):
        return self._tokenizer

    def _prepare(self):
        if isinstance(self._texts, TrainTextsClassified):
            self._tokenizer = prepare_tokenizer(self._texts.train, self._vocab_size)
        elif isinstance(self._texts, TrainDataProvider):
            self._tokenizer = prepare_tokenizer(self._texts.x_train, self._vocab_size)
        else:
            assert False, "Shouldn't be here!"

        if self._debug:
            # Построение словаря в виде пар слово - индекс
            items = list(self._tokenizer.word_index.items())

            # Вывод нескольких наиболее часто встречающихся слов
            print(f"Первые 120 элементов словаря: {items[:120]}")

            # Размер словаря может быть больше, чем num_words, но при преобразовании в последовательности
            # и векторы bag of words будут учтены только первые num_words слов
            print("Размер словаря", len(items))

        if isinstance(self._texts, TrainTextsClassified):
            self._seq_train = words_to_tokens(self._texts.train,    self._tokenizer)
            self._seq_test  = words_to_tokens(self._texts.test,     self._tokenizer)
            self._seq_val = None
        elif isinstance(self._texts, TrainDataProvider):
            self._seq_train = words_to_tokens(self._texts.x_train,  self._tokenizer)
            if self._texts._x_test is not None:
                self._seq_test = words_to_tokens(self._texts.x_test,   self._tokenizer)
            else:
                self._seq_test = None
            if self._texts._x_val is not None:
                self._seq_val = words_to_tokens(self._texts.x_val,    self._tokenizer)
            else:
                self._seq_val = None
        else:
            assert False, "Shouldn't be here!"

        if self._debug:
            print("Фрагмент обучающего текста:")
            if isinstance(self._texts, TrainTextsClassified):
                print("В виде оригинального текста:              ", self._texts.train[0][:101])
            elif isinstance(self._texts, TrainDataProvider):
                print("В виде оригинального текста:              ", self._texts.x_train[0][:101])
            print("Он же в виде последовательности индексов: ", self._seq_train[0][:20])

        with timex("Преобразование текста в обучающие последовательности"):
            if self._text_prepare_function == prepare_long_texts:
                if isinstance(self._texts, TrainTextsClassified):
                    self._x_train, self._y_train = prepare_long_texts(
                        self._seq_train,
                        self._texts.classes,
                        self._chunk_size,
                        self._chunk_step
                    )
                else:
                    raise NotImplementedError("don't know (yet) how to match prepare_long_texts with anything but TrainTextsClassified")
            else:
                if isinstance(self._texts, TrainDataProvider):
                    self._x_train, self._y_train = self._text_prepare_function(
                        x_sequences = self._seq_train,
                        y_data = self._texts.y_train,
                    )
                elif isinstance(self._texts, TrainTextsClassified):
                    self._x_train, self._y_train = self._text_prepare_function(
                        x_sequences = self._seq_train,
                        y_data = self._texts.classes
                    )
                else:
                    assert False, "Shouldn't be here!"

            if self._seq_test is None:
                self._x_test = None
                self._y_test = None
            if self._text_prepare_function == prepare_long_texts:
                if isinstance(self._texts, TrainTextsClassified):
                    self._x_test, self._y_test = prepare_long_texts(
                        self._seq_test,
                        self._texts.classes,
                        self._chunk_size,
                        self._chunk_step
                    )
                else:
                    raise NotImplementedError("don't know (yet) how to match prepare_long_texts with anything but TrainTextsClassified")
            else:
                if isinstance(self._texts, TrainDataProvider):
                    self._x_test, self._y_test = self._text_prepare_function(
                        x_sequences = self._seq_test,
                        y_data = self._texts.y_test,
                    )
                elif isinstance(self._texts, TrainTextsClassified):
                    self._x_test, self._y_test = self._text_prepare_function(
                        x_sequences = self._seq_test,
                        y_data = self._texts.classes
                    )
                else:
                    assert False, "Shouldn't be here!"

            if self._seq_val is None:
                self._x_val = self._x_test
                self._y_val = self._y_test
            elif self._text_prepare_function == prepare_long_texts:
                if isinstance(self._texts, TrainTextsClassified):
                    self._x_val, self._y_val = prepare_long_texts(
                        self._seq_val,
                        self._texts.classes,
                        self._chunk_size,
                        self._chunk_step
                    )
                else:
                    raise NotImplementedError("don't know (yet) how to match prepare_long_texts with anything but TrainTextsClassified")
            else:
                if isinstance(self._texts, TrainDataProvider):
                    self._x_val, self._y_val = self._text_prepare_function(
                        x_sequences = self._seq_val,
                        y_data = self._texts.y_val,
                    )
                elif isinstance(self._texts, TrainTextsClassified):
                    self._x_val, self._y_val = self._text_prepare_function(
                        x_sequences = self._seq_val,
                        y_data = self._texts.classes
                    )
                else:
                    assert False, "Shouldn't be here!"

        if False and self._debug:
            # Проверка формы сформированных данных
            print("x_train.shape=", self._x_train.shape, " y_train.shape=", self._y_train.shape)
            if self._x_test is not None:
                print("x_test.shape =", self._x_test.shape,  " y_test.shape =", self._y_test.shape)
            # Вывод отрезка индексов тренировочной выборки
            print("Отрезок индексов тренировочной выборки", self.x_train[0])

        if self._bow_used:
            with timex("Преобразование последовательностей в bow"):
                # На входе .sequences_to_matrix() ожидает список, .tolist() выполняет преобразование типа
                if not isinstance(self._x_train, list):
                    tokens = self._x_train.tolist()
                else:
                    tokens = self._x_train
                self._x_train_bow       = tokens_to_bow(tokens, self._tokenizer)

                if self._x_test is not None:
                    if not isinstance(self._x_test, list):
                        tokens = self._x_test.tolist()
                    else:
                        tokens = self._x_test
                    self._x_test_bow    = tokens_to_bow(tokens, self._tokenizer)
                else:
                    self._x_test_bow = None

                if self._x_val is not None:
                    if not isinstance(self._x_val, list):
                        tokens = self._x_val.tolist()
                    else:
                        tokens = self._x_val
                    self._x_val_bow     = tokens_to_bow(self._x_val.tolist(), self._tokenizer)
                else:
                    self._x_val_bow = None

            if self._debug:
                # Вывод формы обучающей выборки в виде разреженной матрицы Bag of Words
                print("x_train_bow.shape=", self._x_train_bow.shape)
                # Вывод фрагмента отрезка обучающего текста в виде Bag of Words
                print("Фрагмент отрезка обучающего текста в виде Bag of Words", self._x_train_bow[0][0:100])
            else:
                self._x_train_bow = None
                self._x_test_bow = None
                self._x_val_bow = None

        if not self._seq_train:
            # Save some RAM if sequences aren't required
            self._seq_train = None
            self._seq_val = None
            self._seq_test = None

        # Save a bit more RAM since tokenizer isn't required anymore
        self._tokenizer = None

    @property
    def x_train(self):
        if self._bow_default:
            return self.x_train_bow
        else:
            return self.x_train_seq

    @property
    def x_val(self):
        if self._bow_default:
            return self.x_val_bow
        else:
            return self.x_val_seq

    @property
    def x_test(self):
        if self._bow_default:
            return self.x_test_bow
        else:
            return self.x_test_seq

    @property
    def x_train_seq(self):
        return self._x_train

    @property
    def x_val_seq(self):
        return self._x_val

    @property
    def x_test_seq(self):
        return self._x_test

    @property
    def x_train_bow(self):
        return self._x_train_bow

    @property
    def x_val_bow(self):
        return self._x_val_bow

    @property
    def x_test_bow(self):
        return self._x_test_bow


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
            model_class     = model_data['model_class'],
            optimizer       = model_data.get('optimizer', ENV[ENV__TRAIN__DEFAULT_OPTIMIZER]),
            loss            = model_data.get('loss',      ENV[ENV__TRAIN__DEFAULT_LOSS]),
            metrics         = model_data.get('metrics',   ENV[ENV__TRAIN__DEFAULT_METRICS]),
            model_template  = model_data['template'],
            model_variables = variables,
            class_labels    = train_data.classes_labels,
            batch_size      = model_data.get('batch_size',ENV[ENV__TRAIN__DEFAULT_BATCH_SIZE]),
            data_provider   = train_data,
        )
        thd = TrainHandler(
            data_path       = ENV[ENV__MODEL__DATA_ROOT] / model_data.get("data_path", ENV[ENV__TRAIN__DEFAULT_DATA_PATH]),
            data_name       = model_data['name'],
            mhd             = mhd,
            mhd_class       = ClassClassifierHandler,
        )
        thd.train(
            from_scratch    = model_data.get("from_scratch", ENV[ENV__TRAIN__DEFAULT_FROM_SCRATCH]),
            epochs          = model_data.get("epochs", ENV[ENV__TRAIN__DEFAULT_EPOCHS]),
            target          = model_data.get("target", ENV[ENV__TRAIN__DEFAULT_TARGET]),
            save_step       = model_data.get("save_step", ENV[ENV__TRAIN__DEFAULT_SAVE_STEP]),
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
        texts = load_texts_from_dir(ENV[ENV__TEXTS__TRAIN_TEXTS_PATH], ENV[ENV__TEXTS__TRAIN_TEXTS_NAME_REGEX], ENV[ENV__TEXTS__TRAIN_TEXTS_SUBSETS])
        text_train_data = TextTrainDataProvider(
            texts,
            vocab_size,
            chunk_size,
            chunk_step,
            bow_used=ENV[ENV__TEXTS__BOW_USED],
            debug=ENV[ENV__DEBUG_PRINT],
        )
