###
# Models Trainer common classes
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

from pathlib import Path
import copy
import os
import shutil
import pickle
import yaml
import numpy as np
import math
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


###
ENV__MODEL__DATA_ROOT = "ENV__MODEL__DATA_ROOT"
ENV[ENV__MODEL__DATA_ROOT] = Path("~/ai-learn")

ENV__TRAIN__DEFAULT_DATA_PATH       = "ENV__TRAIN__DEFAULT_DATA_PATH"
ENV__TRAIN__DEFAULT_OPTIMIZER       = "ENV__TRAIN__DEFAULT_OPTIMIZER"
ENV__TRAIN__DEFAULT_LOSS            = "ENV__TRAIN__DEFAULT_LOSS"
ENV__TRAIN__DEFAULT_METRICS         = "ENV__TRAIN__DEFAULT_METRICS"
ENV__TRAIN__DEFAULT_BATCH_SIZE      = "ENV__TRAIN__DEFAULT_BATCH_SIZE"
ENV__TRAIN__DEFAULT_EPOCHS          = "ENV__TRAIN__DEFAULT_EPOCHS"
ENV__TRAIN__DEFAULT_TARGET          = "ENV__TRAIN__DEFAULT_TARGET"
ENV__TRAIN__DEFAULT_SAVE_STEP       = "ENV__TRAIN__DEFAULT_SAVE_STEP"
ENV__TRAIN__DEFAULT_FROM_SCRATCH    = "ENV__TRAIN__DEFAULT_FROM_SCRATCH"

ENV__TRAIN__SPLIT_SHUFFLE_DEFAULT = "ENV__TRAIN__SPLIT_SHUFFLE_DEFAULT"
ENV__TRAIN__RANDOM_SEED_DEFAULT = "ENV__TRAIN__RANDOM_SEED_DEFAULT"
ENV[ENV__TRAIN__SPLIT_SHUFFLE_DEFAULT] = True
ENV[ENV__TRAIN__RANDOM_SEED_DEFAULT] = 1

def connect_gdrive():
    from google.colab import drive
    drive.mount("/content/drive/")
    ENV[ENV__MODEL__DATA_ROOT] = Path("/content/drive/MyDrive/ai-learn")
###

S_EXTERNAL = "_external_"
S_REGULAR = "regular"
S_BACKUP = "backup"
S_BEST = "best"
S_COMPLETE = "complete"

S_ACCURACY = "accuracy"
S_MAE = "mae"
S_MSE = "mse"
S_LOSS = "loss"

S_CM = "cm"
S_REGRESSION = "regression"

MAE_MAX = float(2**31-1)

S_COMPARE = "compare"
S_FUNCTION = "function"
S_FALLBACK = "fallback"
S_GE = "ge"
S_LE = "le"

METRICS = {
    S_ACCURACY  : {S_COMPARE: S_GE},
    S_MAE       : {S_COMPARE: S_LE},
    S_MSE       : {S_COMPARE: S_LE},
}

# Metrics translation / full sentence
METRICS_T = {
    S_ACCURACY  : "доля верных ответов",
    S_MAE       : "средняя абсолютная ошибка",
    S_LOSS      : "ошибка",
}

# Default fallback based on compare function
METRICS_F = {
    S_GE        : 0.0,
    S_LE        : 2**31,
}


def split_by_idx(source, split, list_conv=None):
    if not isinstance(split, dict):
        raise ValueError("'split' should be a dict with mandatory field 'train' and additional fields 'val' and 'test'!")
    result = {"train": None, "val": None, "test": None}
    if list_conv is None:
        list_conv = np.array
    for field in ("val", "test", "train"):
        if field not in split:
            continue
        if isinstance(source, dict):
            split_data = {}
            for k, v in source.items():
                split_data[k] = list_conv([v[idx] for idx in split[field]])
            result[field] = split_data
        else:
            result[field] = list_conv([source[idx] for idx in split[field]])
    return result


class SplitSequenceDef:

    def __init__(
        self,
        val_size    : int | float | None,
        test_size   : int | float | None,
        margin      : int | None,
        y_start_offset: int | None,
        y_end_offset  : int | None,
    ):
        self.val_size = val_size
        self.test_size = test_size
        self.margin = margin
        self.y_start_offset = y_start_offset
        self.y_end_offset = y_end_offset


def split_to_abs(source_len, split:SplitSequenceDef):
    """
    convert float/none to ints based on source_len
    """
    result = []
    for value in (split.val_size, split.test_size, ):
        if value is not None:
            if isinstance(value, float):
                if value > 0. :
                    value = math.ceil(value*source_len)
                else:
                    value = 0
        else:
            value = 0
        if value < 0:
            value = 0
        result.append(value)
    return SplitSequenceDef(
        *result,
        split.margin,
        split.y_start_offset,
        split.y_end_offset
    )


def train_val_test_boundaries(split:SplitSequenceDef, source_len: int):
    """
    provide boundaries for splitting sequenced data into train/val/test subsets
    returned as tuple of three pairs start/end
    """
    split = split_to_abs(source_len, split)
    train_start = 0
    train_end = source_len - split.val_size - split.test_size - max(split.y_start_offset or 0, split.y_end_offset or 0)
    if split.val_size != 0:
        train_end -= split.margin
    if split.test_size != 0:
        train_end -= split.margin
    if split.val_size > 0:
        val_start = train_end + split.margin
        val_end = val_start + split.val_size
    else:
        val_start = train_end
        val_end = train_end
    if split.test_size > 0:
        test_start = val_end + split.margin
        test_end = test_start + split.test_size
    else:
        test_start = val_end
        test_end = val_end
    if split.val_size == 0:
        val_start = test_start
        val_end = test_end

    return (
        (train_start, train_end),
        (val_start, val_end),
        (test_start, test_end),
    )


class TrainDataProvider:
    """
    Класс для абстракции источников данных для обучения
    Упрощает написание кода, позволяет избегать ошибок из-за путаницы имен
    """

    def __init__(self,
        x_train : list|dict,
        y_train : list|dict,
        x_val   : list|dict|int|float|None,
        y_val   : list|dict|None,
        x_test  : list|dict|int|float|None,
        y_test  : list|dict|None,
        x_order : list|None=None,
        y_order : list|None=None,
        split   : dict|None=None,
        split_y : bool = True,
    ):
        if split is not None:
            d_split = split_by_idx(x_train, split)
            if d_split["train"] is not None:
                x_train = d_split["train"]
            if d_split["val"] is not None:
                x_val   = d_split["val"]
            if d_split["test"] is not None:
                x_test  = d_split["test"]

            if split_y:
                d_split = split_by_idx(y_train, split)
                if d_split["train"] is not None:
                    y_train = d_split["train"]
                if d_split["val"] is not None:
                    y_val   = d_split["val"]
                if d_split["test"] is not None:
                    y_test  = d_split["test"]
        else:
            if isinstance(x_test, (int, float)):
                if isinstance(x_test, float):
                    x_test = 1. - x_test
                if isinstance(x_val, float):
                    x_val = x_val / x_test  # balance x_val
                x_train, x_test, y_train, y_test = train_test_split(
                    x_train, y_train,
                    train_size=x_test,
                    shuffle=ENV[ENV__TRAIN__SPLIT_SHUFFLE_DEFAULT],
                    random_state=ENV[ENV__TRAIN__SPLIT_SHUFFLE_DEFAULT]
                )
            if isinstance(x_val, (int, float)):
                if isinstance(x_val, float):
                    x_val = 1. - x_val
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train, y_train,
                    train_size=x_val,
                    shuffle=ENV[ENV__TRAIN__SPLIT_SHUFFLE_DEFAULT],
                    random_state=ENV[ENV__TRAIN__SPLIT_SHUFFLE_DEFAULT]
                )

        self._x_train = x_train
        self._y_train = y_train
        self._x_val   = x_val
        self._y_val   = y_val
        self._x_test  = x_test
        self._y_test  = y_test
        self.x_order  = x_order
        self.y_order  = y_order

    def _x(self, data, order):
        order = order or self.x_order
        if order is None:
            return data
        if not isinstance(data, (dict, list, tuple)):
            raise ValueError("Ordered output only available for data organized as dict, list or tuple")
        return [data[k] for k in order]

    def _y(self, data, order):
        order = order or self.y_order
        if order is None:
            return data
        if not isinstance(data, (dict, list, tuple)):
            raise ValueError("Ordered output only available for data organized as dict, list or tuple")
        return [data[k] for k in order]

    @property
    def x_train(self):
        return self._x(self._x_train, None)

    @property
    def y_train(self):
        return self._y(self._y_train, None)

    @property
    def xy_train(self):
        return self.x_train, self.y_train

    @property
    def x_val(self):
        return self._x(self._x_val, None)

    @property
    def y_val(self):
        return self._y(self._y_val, None)

    @property
    def xy_val(self):
        return self.x_val, self.y_val

    @property
    def x_test(self):
        return self._x(self._x_test, None)

    @property
    def y_test(self):
        return self._y(self._y_test, None)

    @property
    def xy_test(self):
        return self.x_test, self.y_test

    def all_as_tuple(self, x_order=None, y_order=None):
        return (
            self._x(self.x_train, x_order),
            self._y(self.y_train, y_order),
            self._x(self.x_val  , x_order),
            self._y(self.y_val  , y_order),
            self._x(self.x_test , x_order),
            self._y(self.y_test , y_order),
        )


class TrainSequenceProvider(TrainDataProvider):

    def __init__(
        self,
        x_train : list,
        y_train : list,
        x_val   : list|int|float|None,
        y_val   : list|None,
        x_test  : list|int|float|None,
        y_test  : list|None,
        split   : SplitSequenceDef|None=None,
        split_y : bool=True,
        seq_len : int = None,
        stride  : int = 1,
        sampling_rate : int = 1,
        shuffle : bool = False,
        reverse : bool = False,
        batch_size : int = 1,
        soldify_val  : bool = True,
        soldify_test : bool = True,
    ):
        if seq_len is None:
            raise ValueError("Specify 'seq_length'!")

        self._batch_size = batch_size
        self._seq_len = seq_len
        self._stride = stride
        self._sampling_rate = sampling_rate

        super(TrainSequenceProvider, self).__init__(
            x_train = None,
            y_train = None,
            x_val   = None,
            y_val   = None,
            x_test  = None,
            y_test  = None,
        )

        if split is None:
            train_se, val_se, test_se = (0, None), (0, None), (0, None)
        else:
            train_se, val_se, test_se = train_val_test_boundaries(split, len(x_train))
            x_val  = x_train    # NOTE: start/end index would be used
            x_test = x_train    # NOTE: start/end index would be used

            if split_y:
                y_val   = y_train   # NOTE: start/end index would be used
                y_test  = y_train   # NOTE: start/end index would be used

        self._train_gen = TimeseriesGenerator(
            x_train,
            y_train,
            start_index = train_se[0],
            end_index   = train_se[1],
            length=self._seq_len,
            stride=self._stride,
            sampling_rate=self._sampling_rate,
            shuffle=shuffle,
            reverse=reverse,
            batch_size=self._batch_size
        )

        self._val_gen = TimeseriesGenerator(
            x_val,
            y_val,
            start_index = val_se[0],
            end_index   = val_se[1],
            length=self._seq_len,
            stride=self._stride,
            sampling_rate=self._sampling_rate,
            shuffle=False,
            reverse=reverse,
            batch_size=max(1, val_se[1]- val_se[0])
        )

        if soldify_val:
            self._val_xy = self._val_gen[0]
        else:
            self._val_xy = None

        self._test_gen = TimeseriesGenerator(
            x_test,
            y_test,
            start_index = test_se[0],
            end_index   = test_se[1],
            length=self._seq_len,
            stride=self._stride,
            sampling_rate=self._sampling_rate,
            shuffle=False,
            reverse=reverse,
            batch_size=max(1, test_se[1]- test_se[0])
        )

        if soldify_test:
            self._test_xy = self._test_gen[0]
        else:
            self._test_xy = None


    @property
    def seq_len(self):
        return self._seq_len

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def x_train(self):
        return self._train_gen

    @property
    def y_train(self):
        return None

    @property
    def xy_train(self):
        return self._train_gen

    @property
    def x_val(self):
        if self._val_xy is not None:
            return self._val_xy[0]
        else:
            return self._val_gen

    @property
    def y_val(self):
        if self._val_xy is not None:
            return self._val_xy[1]
        else:
            return None

    @property
    def xy_val(self):
        if self._val_xy is not None:
            return self._val_xy
        else:
            return self._val_gen

    @property
    def x_test(self):
        if self._test_xy is not None:
            return self._test_xy[0]
        else:
            return self._test_gen

    @property
    def y_test(self):
        if self._test_xy is not None:
            return self._test_xy[1]
        else:
            return self._test_gen

    @property
    def xy_test(self):
        if self._test_xy is not None:
            return self._test_xy
        else:
            return self._test_gen

    def all_as_tuple(self, x_order=None, y_order=None):
        raise NotImplementedError("'all_as_tuple' is not implemented for 'TrainSequenceProvider'!")


class ModelContext:
    """
    Holds basic model training context that is lightweight to dump with pickle
    (i.e. everything except model itself)
    Is designed for saving intermediate / best training checkpoints
    Also provides general reports printing / saving (text, graphs)
    Saves general data in YAML format as well so it easy to view and recover in case if pickle fails
    """

    _hist_figsize           = (8, 3)
    _extra_dump_vars        = tuple()
    _plot_cols              = 2

    def __init__(
        self,
        name                : str,
        model_class         : None,
        optimizer           : None,
        loss                : None,
        metrics             : list | dict | None,
        model_template      : list | None = None,
        model_variables     : dict | None = None,
        inputs_order        : list | None = None,
        reports             : list[str] | None = None,
        history             = None,
        test_pred           = None,
        eval_data           = None,
    ):
        self.name           = name
        self.model_class    = model_class
        self.optimizer      = optimizer
        self.loss           = loss
        self.metrics        = metrics
        self.model_template = model_template
        self.model_variables = model_variables
        self.inputs_order   = inputs_order
        self.reports        = reports
        self.history        = history
        self.report_history = None
        self.test_pred      = test_pred
        self.eval_data      = eval_data

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        if value is not None and not isinstance(value, (list, dict)):
            raise ValueError("'value' should be a list or a dict!")
        value = value or []
        self._metrics = [v for v in value if isinstance(v, str)]

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        self._history = value

    def get_metric_value(self, metric, fallback=False):
        if metric not in self._metrics:
            return None
        val_metric = 'val_'+metric
        if self._history is None or val_metric not in self._history or len(self._history[val_metric]) == 0:
            value = None
        else:
            value = self._history[val_metric][-1]
        if value is None and fallback:
            value = METRICS[metric].get(S_FALLBACK, METRICS_F[METRICS[metric].get(S_COMPARE, S_GE)])
        return value

    @property
    def metrics_data(self):
        result = {}
        for metric in self._metrics:
            result[metric] = self.get_metric_value(metric)
        return result

    @property
    def eval_data(self):
        return copy.deepcopy(self._eval_data)

    @eval_data.setter
    def eval_data(self, value):
        self._eval_data = {}
        for k in self._metrics:
            if value is None:
                self._eval_data[k] = None
            elif k in value:
                self._eval_data[k] = value[k]
            else:
                self._eval_data[k] = METRICS[k].get(S_FALLBACK, METRICS_F[METRICS[k].get(S_COMPARE, S_GE)])

    def get_eval_value(self, metric, fallback=False):
        if metric not in self._metrics:
            return None
        value = self._eval_data.get(metric, None)
        if value is None and fallback:
            value = METRICS[metric].get(S_FALLBACK, METRICS_F[METRICS[metric].get(S_COMPARE, S_GE)])
        return value

    @property
    def epoch(self):
        if self._history is None:
            return 0
        return len(self._history[S_LOSS])

    @property
    def val_loss(self):
        if self._history is None or len(self._history['val_'+S_LOSS])==0:
            return None
        return self._history['val_'+S_LOSS][-1]

    def _plot_images(self, plotter, plot_f, plot_hist_args):
        plot_history = self.report_history or self.history
        if plot_history is not None:
            print_metrics = [S_LOSS] + [m for m in self._metrics if m in plot_history]
            cols = self._plot_cols
            rows = (len(print_metrics)+cols-1) // cols
            figsize = (self._hist_figsize[0]*cols, self._hist_figsize[1]*rows)
            fig, subplots = plotter.subplots(rows, cols, figsize=figsize)
            fig.suptitle('График процесса обучения модели')
            s_col = 0
            s_row = 0
            for metric in print_metrics:
                if cols > 1 and rows > 1:
                    subplot = subplots[s_row, s_col]
                elif rows > 1:
                    subplot = subplots[s_row]
                else:
                    subplot = subplots[s_col]
                s_col += 1
                if s_col >= cols:
                    s_col = 0
                    s_row += 1
                metric_t = METRICS_T.get(metric, metric).capitalize()
                x_vals = list(range(1, len(plot_history[metric])+1))
                subplot.plot(x_vals, plot_history[metric],
                        label=metric_t + ' на обучающем наборе')
                subplot.plot(x_vals, plot_history['val_'+metric],
                        label=metric_t + ' на проверочном наборе')
                subplot.xaxis.get_major_locator().set_params(integer=True)
                subplot.set_xlabel('Эпоха обучения')
                subplot.set_xlim(1, len(plot_history[metric]))
                subplot.set_ylabel(metric_t)
                subplot.legend()
            plot_f(*plot_hist_args)

    def short_report(self):
        return "TODO:"

    def summary_to_disk(
        self,
        path: Path|str,
        extra_model_data : dict | None = None
    ):
        path = Path(path)
        if extra_model_data is None:
            extra_model_data = {}
        with open(path / f"epoch-{self.epoch}", "w") as f:
            pass
        metrics_data = self.metrics_data
        eval_data    = self.eval_data
        dump_data = {
            "name"          : self.name,
            "model_class"   : self.model_class,
            "optimizer"     : self.optimizer,
            "epoch"         : self.epoch,
            "loss"          : self.loss,
            "metrics"       : self._metrics,
            "val_loss"      : self.val_loss,
            "metrics_data"  :      metrics_data,
            "eval_data"     :      eval_data,
        }
        for metric, value in metrics_data.items():
            with open(path / safe_path(f"val_{metric}-{value}"), "w") as f:
                pass
        for metric, value in eval_data.items():
            with open(path / safe_path(f"test_{metric}-{value}"), "w") as f:
                pass

        for v in self._extra_dump_vars:
            dump_data[v] = getattr(self, v)
        dump_data['model_template'] = self.model_template
        dump_data['model_variable'] = self.model_variables
        dump_data['history']        = self.history
        dump_data = {
            **dump_data,
            **extra_model_data,
        }
        dump_data = safe_dict(dump_data)
        with open(path / "model.yaml", "w") as f:
            yaml.safe_dump(dump_data, f,allow_unicode=True, sort_keys=False)
        with open(path / "report.txt", "w") as f:
            f.write(self.short_report())
        self._plot_images(plt, plt.savefig, [path / "history.png"])

    def report_to_screen(self):
        print(self.short_report())
        print(f"train epoch={self.epoch}")
        for metric in self.metrics:
            print(f"'val_{metric}'  = {self.get_metric_value(metric)}")
            print(f"'test_{metric}' = {self.get_eval_value(metric)}")
        self._plot_images(plt, plt.show, [])


class ClassClassifierContext(ModelContext):
    """
    ModelContext extended to Class Classification task
    """

    _cm_figsize             = (5, 5)
    _cm_title_fontsize      = 14
    _cm_fontsize            = 12
    _extra_dump_vars        = (
        "class_labels",
    )

    def __init__(
        self,
        name                : str,
        model_class         : None,
        optimizer           : None,
        loss                : None,
        metrics             : list | dict | None,
        model_template      : list | None = None,
        model_variables     : dict | None = None,
        inputs_order        : list | None = None,
        reports             : list[str] | None = None,
        history             = None,
        test_pred           = None,
        eval_data           = None,

        class_labels        : list[str] | None = None,
        cm                  = None,
    ):
        super(ClassClassifierContext, self).__init__(
            name           = name,
            model_class    = model_class,
            optimizer      = optimizer,
            loss           = loss,
            metrics        = metrics,
            model_template = model_template,
            model_variables = model_variables,
            inputs_order   = inputs_order,
            reports        = reports,
            history        = history,
            test_pred      = test_pred,
            eval_data      = eval_data,
        )
        self.class_labels   = class_labels
        self.cm             = cm
        self.reports        = reports or [S_CM]

    def _plot_cm_images(self, plotter, plot_f, plot_cm_args):
        if self.cm is not None:
            fig, ax = plotter.subplots(figsize=self._cm_figsize)
            ax.set_title(f'Нейросеть {self.name}: матрица ошибок нормализованная', fontsize=self._cm_title_fontsize)
            disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, display_labels=self.class_labels)
            disp.plot(ax=ax)
            plotter.gca().images[-1].colorbar.remove()  # Стирание ненужной цветовой шкалы
            plotter.xlabel('Предсказанные классы', fontsize=self._cm_fontsize)
            plotter.ylabel('Верные классы', fontsize=self._cm_fontsize)
            fig.autofmt_xdate(rotation=45)          # Наклон меток горизонтальной оси при необходимости
            plot_f(*plot_cm_args)

    def short_report(self):
        report_data = {}
        report = []

        if self.cm is None and S_CM in self.reports:
            report.append["No data for Confusion Matrix! call update_data() first!"]

        if self.cm is not None:
            # Для каждого класса:
            for cls in range(len(self.class_labels)):
                # Определяется индекс класса с максимальным значением предсказания (уверенности)
                cls_pred = np.argmax(self.cm[cls])
                # Формируется сообщение о верности или неверности предсказания
                report_data[self.class_labels[cls]] = {
                    'top_rate'  : 100. * self.cm[cls, cls_pred],
                    'top_class' : self.class_labels[cls_pred],
                    'success'   : cls_pred == cls
                }

            report.append(('-'*100))
            report.append(report_from_dict(
                f'Нейросеть: {self.name}', report_data,
                'Класс: {:<20} {:3.0f}% сеть отнесла к классу {:<20} - Верно: {}', None
            ))
            # Средняя точность распознавания определяется как среднее диагональных элементов матрицы ошибок
            report.append('\nСредняя точность распознавания: {:3.0f}%'.format(100. * self.cm.diagonal().mean()))

        if S_REGRESSION in self.reports:
            pass    # TODO:

        return "\n".join(report)

    def summary_to_disk(
        self,
        path: Path|str,
        extra_model_data : dict | None = None
    ):
        if extra_model_data is None:
            extra_model_data = {}
        extra_model_data['class_labels'] = self.class_labels
        super(ClassClassifierContext, self).summary_to_disk(path, extra_model_data)
        self._plot_cm_images(plt, plt.savefig, [path / "cm.png"])

    def report_to_screen(self):
        super(ClassClassifierContext, self).report_to_screen()
        self._plot_cm_images(plt, plt.show, [])


class ModelHandler():
    """
    Abstraction layer on top of model to
    - remove complexity and unify models creation
    - provide unified way to save/load training context
    """

    _context_class = ModelContext

    def __init__(
        self,
        name            : str,
        model_class,
        optimizer,
        loss,
        metrics         : list | None = None,
        model_template  : list | dict = None,
        model_variables : dict | None = None,
        batch_size      : int  | None = 10,
        data_provider   : TrainDataProvider = None,
        load_weights_only : bool = False
    ):
        if metrics is None:
            metrics = [S_ACCURACY]
        self._context = self._context_class(
            name            = name,
            model_class     = model_class,
            optimizer       = optimizer,
            loss            = loss,
            metrics         = metrics,
            inputs_order    = None,
        )
        self.model_template  = model_template
        self.model_variables = model_variables
        self.batch_size      = batch_size
        self._model          = None
        self._named_layers   = {}
        self.data_provider   = data_provider
        self.load_weights_only = load_weights_only

    @property
    def context(self):
        return self._context

    @property
    def name(self):
        return self._context.name

    @name.setter
    def name(self, value):
        self._context.name = value

    @property
    def model_class(self):
        return self._context.model_class

    @model_class.setter
    def model_class(self, value):
        self._context.model_class = value

    @property
    def model_template(self):
        return self._context.model_template

    @model_template.setter
    def model_template(self, value):
        if value is None:
            self._context.model_template = []
        else:
            if isinstance(value, (list,tuple)):
                self._context.model_template = tuple(copy.deepcopy(list(value)))
            elif isinstance(value, dict):
                self._context.model_template = copy.deepcopy(value)
            else:
                raise ValueError("'model_template' should be a list, tuple or a dict!")

    @property
    def model_variables(self):
        return self._context.model_variables

    @model_variables.setter
    def model_variables(self, value):
        if value is None:
            self._context.model_variables = {}
        else:
            self._context.model_variables = copy.deepcopy(value)

    @property
    def model(self):
        return self._model

    @property
    def named_layers(self):
        return {**self._named_layers}

    @property
    def inputs_order(self):
        return self._context.inputs_order

    @property
    def metrics(self):
        return self._context.metrics

    @metrics.setter
    def metrics(self, value):
        for v in value:
            if v not in METRICS:
                raise ValueError(f"Unknown metric {v}!")
        self._context.metrics = value

    @property
    def history(self):
        return self._context.history

    @history.setter
    def history(self, value):
        if value is None:
            self._context.history = None
            return
        if self._context.history is None:
            self._context.history = {}
            for k in value:
                self._context.history[k] = [] + value[k]
        else:
            for k in self._context.history:
                self._context.history[k] += value[k]

    def create(self):
        mc = model_create(
            self.model_class,
            self.model_template,
            **self.model_variables
        )
        def get_metrics(metrics):
            result = []
            for metric in metrics:
                if S_FUNCTION in METRICS[metric]:
                    result.append(METRICS[metric][S_FUNCTION])
                else:
                    result.append(metric)
            return result
        self._model         = mc[S_MODEL]
        self._context.inputs_order  = mc[S_INPUTS]
        self._named_layers  = mc[S_NAMED_LAYERS]    # NOTE: named layers aren't restored on model load, use load_weights if named layers are used!
        if isinstance(self.context.optimizer, (list, tuple)):
            optimizer = self.context.optimizer[0](*self.context.optimizer[1], **self.context.optimizer[2])
        elif callable(self.context.optimizer):
            optimizer = self.context.optimizer()
        else:
            optimizer = self.context.optimizer
        self._model.compile(
            optimizer=optimizer,
            loss=self.context.loss,
            metrics=get_metrics(self.context.metrics),
        )

    def fit(self, epochs, initial_epoch=None, kwargs=None):
        if kwargs is None:
            kwargs = {}
        if initial_epoch is not None:
            kwargs['initial_epoch'] = initial_epoch
        self.history = self.model.fit(
            self.data_provider.x_train, self.data_provider.y_train,
            batch_size=self.batch_size,
            epochs=(initial_epoch or 0) + epochs,
            validation_data=(self.data_provider.x_val, self.data_provider.y_val),
            **kwargs,
        ).history
        self._context.test_pred = None
        self._context.eval_data = None
        return self.history

    def predict(self, data):
        return self.model.predict(data)

    def save(self, path):
        print(f"Saving model state to '{path}'")
        with open(path / "context.pickle", "wb") as f:
            pickle.dump(self._context, f)
        self._model.save(path / "model.keras")
        self._model.save_weights(path / "model.ckpt")
        with open(path / "model.txt", "w") as f:
            self._model.summary(print_fn=lambda x: f.write(x+"\n"))
        self._context.summary_to_disk(path)

    def load(self, path, dont_load_model=None, load_weights=None):
        if dont_load_model is None:
            if self.load_weights_only:
                dont_load_model = True
                load_weights = True
            else:
                dont_load_model = False
                load_weights = False
        if dont_load_model is True and load_weights is True:
            if self._model is None:
                self.create()
        print(f"Loading model state from '{path}'")
        with open(path / "context.pickle", "rb") as f:
            self._context = pickle.load(f)
        if dont_load_model is False:
            custom_objects = {}
            for metric in self.metrics:
                if S_FUNCTION in METRICS[metric]:
                    custom_objects[metric] = METRICS[metric][S_FUNCTION]
            self._model = load_model(
                path / "model.keras",
                custom_objects=custom_objects,
            )
        elif load_weights is True:
            self.load_weights(path)

    def load_weights(self, path):
        self._model.load_weights(path / "model.ckpt")

    def update_data(self, force=False):
        if self._context.test_pred is None or force:
            self._context.test_pred  = self.predict(self.data_provider.x_test)

        self._context.eval_data = None # TODO:

    def unload_model(self):
        self._model = None


class ClassClassifierHandler(ModelHandler):
    """
    ModelHandler extended to Class Classification task
    """

    _context_class = ClassClassifierContext
    _cm_round = 3

    def __init__(
        self,
        name            : str,
        model_class,
        optimizer,
        loss,
        metrics         : list | None = None,
        model_template  : list = None,
        model_variables : dict | None = None,
        batch_size      : int  | None = 10,
        data_provider   : TrainDataProvider = None,
        class_labels    : list[str] = None,
    ):
        super(ClassClassifierHandler, self).__init__(
            name,
            model_class,
            optimizer,
            loss,
            metrics,
            model_template,
            model_variables,
            batch_size,
            data_provider,
        )
        self.class_labels    = class_labels

    @property
    def class_labels(self):
        return self._context.class_labels

    @class_labels.setter
    def class_labels(self, value):
        if value is None:
            self._context.class_labels = []
        else:
            self._context.class_labels = tuple(copy.deepcopy(list(value)))

    def fit(self, epochs, initial_epoch=None, kwargs=None):
        result = super(ClassClassifierHandler, self).fit(epochs, initial_epoch=initial_epoch, kwargs=kwargs)
        self._context.cm = None
        return result

    def update_data(self):
        super(ClassClassifierHandler, self).update_data()

        if self._context.cm is None:
            self._context.cm = confusion_matrix(np.argmax(self.data_provider.y_test, axis=1),
                                np.argmax(self._context.test_pred, axis=1),
                                normalize='true')
            self._context.cm = np.around(self._context.cm, self._cm_round)

        if S_ACCURACY in self._context._eval_data:
            self._context._eval_data[S_ACCURACY] = self._context.cm.diagonal().mean()


class TrainHandler:
    """
    Provides unified and simple way to handle models training and saving/loading training checkpoints
    - lets stop training when desired accuracy reached
    - saves / loads best training result and last training result as well
    - saves intermediate training results so it's easy to continue when stopped for some reason
    """

    def __init__(
        self,
        data_path   : Path | str,
        data_name   : Path | str,
        mhd         : ModelHandler | None,
        mhd_class   = ModelHandler,
        on_model_update = None,
        fit_callbacks = None,
    ):
        self._data_path = data_path
        self.data_name = data_name
        self.mhd = mhd
        self._mhd_class = mhd_class
        self.on_model_update = on_model_update
        self.fit_callbacks = fit_callbacks

    @property
    def data_path(self):
        return self._data_path

    @data_path.setter
    def data_path(self, value):
        self._data_path = value
        self._update_paths()

    @property
    def data_name(self):
        return self._data_name

    @data_name.setter
    def data_name(self, value):
        self._data_name = value
        self._update_paths()

    def _update_paths(self):
        self._last_path = Path(self._data_path) / safe_path(self._data_name) / f"regular"
        self._back_path = Path(self._data_path) / safe_path(self._data_name) / f"regular.bak"
        self._best_path = Path(self._data_path) / safe_path(self._data_name) / f"best"
        self._bbak_path = Path(self._data_path) / safe_path(self._data_name) / f"best.bak"

    @property
    def mhd(self):
        return self._mhd

    @mhd.setter
    def mhd(self, value : ModelHandler):
        self._mhd = value

    def save(self, path=S_REGULAR):
        if path == S_REGULAR:
            save_path = self._last_path
            back_path = self._back_path
        elif path == S_BEST:
            save_path = self._best_path
            back_path = self._bbak_path
        else:
            raise ValueError(f"path should be one of {[S_REGULAR, S_BEST]}")

        if save_path.exists():
            if back_path.exists():
                shutil.rmtree(back_path)
            shutil.copytree(save_path, back_path)
            shutil.rmtree(save_path)

        os.makedirs(save_path)
        if self._mhd is not None:
            self._mhd.save(save_path)
        with open(save_path / S_COMPLETE, "w") as f:
            # Mark that save is complete
            pass

    def is_saved(self, path=S_REGULAR):
        if path == S_REGULAR:
            load_path = self._last_path
            back_path = self._back_path
        elif path == S_BEST:
            load_path = self._best_path
            back_path = self._bbak_path
        else:
            raise ValueError(f"path should be one of {[S_REGULAR, S_BEST]}")

        if (load_path.exists() and (load_path / S_COMPLETE).exists()) \
        or (back_path.exists() and (back_path / S_COMPLETE).exists()):
            return True
        return False

    def load(self, path=S_REGULAR, dont_load_model=None):
        if path == S_REGULAR:
            load_path = self._last_path
            back_path = self._back_path
        elif path == S_BEST:
            load_path = self._best_path
            back_path = self._bbak_path
        else:
            raise ValueError(f"path should be one of {[S_REGULAR, S_BEST]}")

        if not load_path.exists() or not (load_path / S_COMPLETE).exists():
            # If original path not exists or save is not complete
            if back_path.exists() and (back_path / S_COMPLETE).exists():
                # Try to load backup
                load_path = back_path
            else:
                raise FileNotFoundError(f"path '{load_path}' was not found")

        if self._mhd is None:
            self._mhd = self._mhd_class(
                name = self.data_name,
                model_class = None,
                optimizer = None,
                loss = None,
            )
        self._mhd.load(load_path, dont_load_model=dont_load_model)

    def load_last(self):
        if self.is_saved(S_REGULAR):
            self.load(S_REGULAR)
        elif self.is_saved(S_BEST):
            self.load(S_BEST)
        else:
            self.load(S_REGULAR)    # will raise error, as intended

    def load_best(self):
        if self.is_saved(S_BEST):
            self.load(S_BEST)
        elif self.is_saved(S_REGULAR):
            self.load(S_REGULAR)
        else:
            self.load(S_BEST)       # will raise error, as intended

    def clear_saved_data(self):
        for path in (self._last_path, self._back_path, self._best_path, self._bbak_path):
            if path.exists():
                shutil.rmtree(path)

    def is_enough(self, target):
        if target is None:
            return False
        if len(target) == 0:
            return False
        enough = True
        for metric in METRICS:
            if metric in target:
                cmp = METRICS[metric][S_COMPARE]
                value = self.mhd.context.get_metric_value(metric, fallback=True)
                if cmp == S_GE:
                    if value < target[metric]:
                        enough = False
                        break
                elif cmp == S_LE:
                    if value > target[metric]:
                        enough = False
                        break
                else:
                    raise ValueError(f"Unknown compare function '{cmp}' for metric '{metric}'!")
        return enough

    def train(
            self,
            from_scratch    : bool | None,
            epochs          : int  | None,
            target          : dict | None,
            train_step      : int = 1,  # With steps greater than 1 best value can be missed
            save_step       : int = 5,
            display_callback= print,
        ):

        best = TrainHandler(
            # NOTE: used only to load data and hold best value
            data_path       = self.data_path,
            data_name       = self.data_name,
            mhd             = self._mhd_class(
                name=self.data_name,
                model_class=None,
                optimizer=None,
                loss=None,
                metrics=self._mhd.metrics,
            ),
        )

        if from_scratch is not True:
            if self.is_saved(S_REGULAR):
                self.load(S_REGULAR)
                if best.is_saved(S_BEST):
                    best.load(S_BEST, dont_load_model=True)
                    if best.mhd.context.epoch >= self.mhd.context.epoch:
                        self.load(S_BEST)
            elif self.is_saved(S_BEST):
                self.load(S_BEST)
            elif from_scratch is False:
                self.load(S_REGULAR) # will raise error, as intended
            else:
                from_scratch = True

        if from_scratch is True:
            self.clear_saved_data()
            self.mhd.create()
        else:
            if best.is_saved(S_BEST):
                best.load(S_BEST, dont_load_model=True)

        if self.on_model_update is not None:
            self.on_model_update(self)

        if  self.mhd.context.epoch < epochs and not self.is_enough(target):
            display_callback(f"Starting to train from epoch {self.mhd.context.epoch}, max epoch: {epochs}, "
                f", target: {target}, metrics: {self.mhd.context.metrics_data}")

            next_save = self.mhd.context.epoch + save_step

            save_result = False # by default don't save if not trained
            while self.mhd.context.epoch < epochs and not self.is_enough(target):
                save_result = True

                display_metrics = []
                current_metrics = []
                best_metrics = []
                for metric in self.mhd.context.metrics:
                    display_metrics.append(metric)
                    current_metrics.append(str(self.mhd.context.get_metric_value(metric)))
                    best_metrics.append(str(best.mhd.context.get_metric_value(metric)))

                display_callback(
                    f"epoch/{'/'.join(display_metrics)}: "
                    f"current - {self.mhd.context.epoch}/{'/'.join(current_metrics)}, "
                    f"best - {best.mhd.context.epoch}/{'/'.join(best_metrics)}"
                )
                if self.mhd.context.epoch > 0:
                    initial_epoch = self.mhd.context.epoch
                else:
                    initial_epoch = None
                fit_kwargs = {}
                if self.fit_callbacks is not None:
                    fit_kwargs['callbacks'] = self.fit_callbacks
                self.mhd.fit(epochs=train_step, initial_epoch=initial_epoch, kwargs=fit_kwargs)
                # NOTE: fit() would affect .epoch, and other metrics fields

                is_better = False
                for metric in self.mhd.context.metrics:
                    current_value = self.mhd.context.get_metric_value(metric, fallback=True)
                    best_value = best.mhd.context.get_metric_value(metric, fallback=True)
                    cmp = METRICS[metric][S_COMPARE]
                    if cmp == S_GE:
                        if current_value > best_value:
                            is_better = True
                            break
                    elif cmp == S_LE:
                        if current_value < best_value:
                            is_better = True
                            break
                    else:
                        raise ValueError(f"Unknown compare function '{cmp}' for metric '{metric}'!")

                if is_better:
                    self.mhd.update_data()
                    self.save(S_BEST)
                    best.mhd.context.history = copy.deepcopy(self.mhd.context.history)

                if self.mhd.context.epoch == next_save:
                    self.mhd.update_data()
                    self.save(S_REGULAR)
                    next_save += save_step

                if self.is_enough(target):
                    # TODO: check metrics on test_data
                    break

            if save_result:
                self.mhd.update_data()
                self.save(S_REGULAR)



# Few usage examples
if STANDALONE:
    if False and __name__ == "__main__":
        ### Crate model handler
        mhd = ClassClassifierHandler(
            name        ="conv1",
            model_class =Sequential,
            optimizer   ="adam",
            loss        ="categorical_crossentropy",
            metrics     =[S_ACCURACY],
            model_template = [
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
            ],
            model_variables={
                "vocab_size"    : 10000,
                "chunk_size"    : 1000,
                "classes_count" : 100
            },
            # TODO: train, validation, test data
        )

        ###
        # Just Load and train
        thd = TrainHandler(
            data_path = ENV[ENV__MODEL__DATA_ROOT] / "some_additional_path",
            data_name = "some_specific_name",
            mhd = mhd
        )
        thd.load_last()
        thd.mhd.model.summary()
        thd.train(
            from_scratch    =FROM_SCRATCH,
            epochs          =EPOCHS,
            target          = {S_ACCURACY: TARGET_ACCURACY},
            save_step       =SAVE_STEP,
        )
        thd.mhd.context.to_screen()

        ###
        # Load and eval
        thd = TrainHandler(
            data_path = ENV[ENV__MODEL__DATA_ROOT] / "some_additional_path",
            data_name = "some_specific_name",
        )
        thd.load_best()
        mhd = thd.mhd
        mhd.predict(INPUT_DATA)
        ###

    if True and __name__ == "__main__":
        for predict_range in ((1,2), (5,6), (1, 6), ):
            data = [i for i in range(0, 150)]
            seq_len = 10
            split = SplitSequenceDef(
                val_size    = 0.1,
                test_size   = 0.1,
                margin      = seq_len*2,
                y_start_offset= predict_range[0],
                y_end_offset  = predict_range[1]-1
            )
            train_se, val_se, test_se = train_val_test_boundaries(
                split,
                len(data),
            )

            # Amount of last non-full items to be cropped
            data_crop = split.y_end_offset

            x_data = data[:-data_crop]
            y_data = [
                data[ i + split.y_start_offset : i + split.y_end_offset + 1 ]
                for i in range(len(data)-data_crop-1) # NOTE: -1 due to shift below
            ]

            y_data.insert(0, y_data[0]) # NOTE: Required to shift y_data by a single sample (with any valid value)

            data_provider = TrainSequenceProvider(
                x_train     = x_data,
                y_train     = y_data,
                x_val       = None,
                y_val       = None,
                x_test      = None,
                y_test      = None,
                split       = split,
                seq_len     = seq_len,
                stride      = 1,
                sampling_rate = 1,
                shuffle     = False,
                batch_size  = 50,
            )
            train_sample    = [x for x in data_provider.xy_train]
            val_sample      = [x for x in data_provider.xy_val]
            test_sample     = [x for x in data_provider.xy_test]
