import numpy as np


def correlation(a, b):
    """
    Функция расчета корреляционного коэффициента Пирсона для двух рядов
    """
    return np.corrcoef(a, b)[0, 1]


def correlation_graph(a, b, data_start, data_end, graph_range):
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
                    bb[data_start + mult*i:data_end]
                )
            )

    return result
