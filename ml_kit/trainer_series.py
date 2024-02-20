import numpy as np
import keras.backend as K

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


ENV__CORRELATION_PEAKSEARCH_RANGE = "ENV__CORRELATION_PEAKSEARCH_RANGE"
ENV[ENV__CORRELATION_PEAKSEARCH_RANGE] = (-70, 70)


def correlation(a, b, method=1):
    """
    Функция расчета корреляционного коэффициента Пирсона для двух рядов
    """
    if method==1:
        return np.corrcoef(a, b)[0, 1]
    elif method==2:
        a = np.array(a)
        b = np.array(b)
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        ab_mean = np.mean(a*b)
        a_std = np.std(a)
        b_std = np.std(b)
        corr = (ab_mean - a_mean*b_mean)/(a_std*b_std)
        return corr
    else:
        raise ValueError(f"Unsupported method '{method}'")


def correlation_graph(a, b, data_start, data_end, graph_range, method=1):
    if isinstance(graph_range, int):
        graph_range = (0, graph_range)
    result = []
    negative_range = (graph_range[0], -1)
    positive_range = (0, graph_range[1])
    for start, end in (negative_range, positive_range,):
        if start < 0:
            aa = b
            bb = a
            mult = -1
        else:
            aa = a
            bb = b
            mult = 1
        for i in range(start, end+1):
            result.append(
                correlation(
                    aa[data_start:data_end - mult*i],
                    bb[data_start + mult*i:data_end],
                    method=method
                )
            )

    return result


def correlation_peak(a, b, method=1):
    peak_search_range = ENV[ENV__CORRELATION_PEAKSEARCH_RANGE]
    cv = correlation_graph(a, b, 0, -1, peak_search_range, method=method)
    peak = np.argmax(cv) + peak_search_range[0]
    return peak


# Специальные функции (с суффиксом _K) для формирования метрик при обучении

def correlation_K(a, b):
    """
    Функция расчета корреляционного коэффициента Пирсона для двух рядов
    """
    a_mean = K.mean(a)
    b_mean = K.mean(b)
    ab_mean = K.mean(a*b)
    a_std = K.std(a)
    b_std = K.std(b)
    corr = (ab_mean - a_mean*b_mean)/(a_std*b_std)
    return corr


def correlation_graph_K(a, b, data_start, data_end, graph_range):
    if isinstance(graph_range, int):
        graph_range = (0, graph_range)
    result = []
    negative_range = (graph_range[0], -1)
    positive_range = (0, graph_range[1])
    for start, end in (negative_range, positive_range,):
        if start < 0:
            aa = b
            bb = a
            mult = -1
        else:
            aa = a
            bb = b
            mult = 1
        for i in range(start, end+1):
            result.append(
                correlation_K(
                    aa[data_start:data_end - mult*i],
                    bb[data_start + mult*i:data_end]
                )
            )

    return result


def correlation_peak_K(a, b):
    peak_search_range = ENV[ENV__CORRELATION_PEAKSEARCH_RANGE]
    cv = correlation_graph_K(a, b, 0, -1, peak_search_range)
    peak = K.argmax(cv) + peak_search_range[0]
    return peak


S_CORRELATION = 'correlation_K'
METRICS[S_CORRELATION] = {
    S_COMPARE   : S_GE,
    S_FUNCTION  : correlation_K,
    S_FALLBACK  : 0.0
}
METRICS_T[S_CORRELATION] = "корреляция"

S_CORRELATION_PEAK = 'correlation_peak_K'
METRICS[S_CORRELATION_PEAK] = {
    S_COMPARE   : S_LE,
    S_FUNCTION  : correlation_peak_K,
    S_FALLBACK  : 70
}
METRICS_T[S_CORRELATION_PEAK] = "пик корреляции"


# TODO: differentiation
