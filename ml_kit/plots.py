import numpy as np
import matplotlib.pyplot as plt
import copy


class GraphDef:

    def __init__(
        self,
        idx_label : dict,
        title   : str | None,
        x_label : str | None,
        y_label : str | None,
        plot_f
    ):
        self.idx_label = idx_label  # key is index in data source, label is label for legend
        self.title = title          # title for graph
        self.x_label = x_label      # x_label for plot
        self.y_label = y_label      # y_label for plot
        self.plot_f = plot_f          # plotting function (plot_sequences, plot_bars, etc)


def plot_sequences(subplot, data, label, kwargs=None):
    kwargs = kwargs or {}
    subplot.plot(data, label=label, **kwargs)

def plot_xy(subplot, data, label, kwargs=None):
    kwargs = kwargs or {}
    subplot.plot(data[0], data[1], label=label, **kwargs)

def plot_bars(subplot, data, label, kwargs=None):
    kwargs = kwargs or {}
    subplot.bar(x=np.arange(len(data)), height=data, label=label, **kwargs)


def plot_graph(
    subplot,
    graph_data  : list[list],
    graph       : GraphDef,
    start       : int | None = None,
    end         : int | None = None,
    legend      : bool = True,
):
    if graph.title is not None:
        subplot.set_title(graph.title)
    for idx, label in graph.idx_label.items():
        if isinstance (graph.plot_f, (list, tuple)):
            plot_f, kwargs = graph.plot_f
        else:
            plot_f = graph.plot_f
            kwargs = {}
        plot_f(subplot, graph_data[idx][start:end], label, **kwargs)
    if graph.y_label is not None:
        subplot.set_ylabel(graph.y_label)
    if graph.x_label is not None:
        subplot.set_xlabel(graph.x_label)
    if legend:
        subplot.legend()


def plot_to_multiple_rows(
        subplots,
        data2d  : np.array,
        rows    : list[GraphDef],
        start   : int|None=None,
        end     : int|None=None,
        x_label : str|None=None,
        legend  : bool  = True,
        sharex  : bool  = False,
    ):

    for i in range(len(rows)):
        graph = rows[i]
        subplot = subplots[i]
        if graph.title is not None:
            subplot.set_title(graph.title)

        graph_data = []
        graph_map = {}
        i = 0
        for idx, label in graph.idx_label.items():
            graph_data.append(
                data2d[:, idx]
            )
            graph_map[i] = label
            i+=1
        graph_def = copy.deepcopy(graph)
        graph_def.x_label = None
        graph_def.idx_label = graph_map
        plot_graph(
            subplot,
            graph_data,
            graph_def,
            start,
            end,
            legend,
        )
        if not sharex and graph.x_label is not None:
            subplot.set_xlabel(graph.x_label)
    if sharex:
        if x_label is not None:
            plt.xlabel(x_label)
        else:
            for graph in rows:
                if graph.x_label is not None:
                    plt.xlabel(graph.x_label)
                    break


    # NOTE: create subplots with something like this
    # fig, subplots = plt.subplots(len(rows), 1, figsize=(22,13), sharex=sharex)
