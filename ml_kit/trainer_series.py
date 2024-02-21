import numpy as np
import keras.backend as K
from scipy import signal as S

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

S_COREL_METHOD__AUTO = "auto"
S_COREL_METHOD__NP_CORRCOEF = "np.corrcoef"
S_COREL_METHOD__NP = "np"
S_COREL_METHOD__SCIPY_CORRELATE = "scipy.correlate"

S_COREL_METHODS = [
    S_COREL_METHOD__AUTO,
    S_COREL_METHOD__NP_CORRCOEF,
    S_COREL_METHOD__NP,
    S_COREL_METHOD__SCIPY_CORRELATE,
]

def correlation(a, b, method=S_COREL_METHOD__AUTO, method_kwargs=None):
    """
    Функция расчета корреляционного коэффициента Пирсона для двух рядов
    """
    if method not in S_COREL_METHODS:
        raise ValueError(f"Unsupported method '{method}'")

    if method==S_COREL_METHOD__AUTO:
        method = S_COREL_METHOD__NP_CORRCOEF
    if method==S_COREL_METHOD__NP_CORRCOEF:
        return np.corrcoef(a, b)[0, 1]
    elif method==S_COREL_METHOD__NP:
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
        raise ValueError(f"Unsupported method '{method}' for correlation!")


def correlation_graph(a, b, data_start, data_end, graph_range, method=S_COREL_METHOD__AUTO, method_kwargs=None):
    if isinstance(graph_range, int):
        graph_range = (0, graph_range)
    if method != S_COREL_METHOD__SCIPY_CORRELATE:
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
                        method=method,
                        method_kwargs=method_kwargs,
                    )
                )
        return result
    else:
        corr = S.correlate(a[data_start:data_end], b[data_start:data_end], mode='same')
        lags = S.correlation_lags(data_end-data_start, data_end-data_start, mode='same')
        lags_min = lags[0]
        lags_max = lags[-1]
        if graph_range[0] < lags_min:
            raise ValueError("Not enough data!")
        if graph_range[1]-1 > lags_max:
            raise ValueError("Not enough data!")
        start_index = graph_range[0] - lags_min
        end_index   = graph_range[1] - lags_min
        return corr[start_index:end_index]


def correlation_peak(a, b, method=S_COREL_METHOD__AUTO, method_kwargs=None):
    peak_search_range = ENV[ENV__CORRELATION_PEAKSEARCH_RANGE]
    cv = correlation_graph(a, b, 0, -1, peak_search_range, method=method, method_kwargs=method_kwargs)
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
