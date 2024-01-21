################################################################################
# General helper functions / classes

import time
import re

DEBUG_PRINT = True


class timex:
    """
    Контекстный менеджер для измерения времени операций
    Операция обертывается менеджером с помощью оператора with
    """

    def __init__(self, task_name, task_suffix="о"):
        self._task_name = task_name
        self._task_suffix = task_suffix

    def __enter__(self):
        # Фиксация времени старта процесса
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        # Вывод времени работы
        if DEBUG_PRINT:
            print('{} занял{}: {:.2f} с'.format(self._task_name, self._task_suffix, time.time() - self.t))


class ArbitraryClass:
    """Формирует класс из словаря (поля словаря становятся полями класса)
    """
    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)


def to_dict(**kwargs):
    return kwargs


def safe_dict(value):
    """Преобразует значение для безопасного сохранения в YAML файл
    """
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            result[safe_dict(k)] = safe_dict(v)
        return result
    elif isinstance(value, (list, tuple)):
        result = [safe_dict(v) for v in value]
        if isinstance(value, tuple):
            result = tuple(result)
        return result
    elif isinstance(value, (str, int, float)):
        return value
    else:
        return str(value)


def safe_path(path):
    """Преобразует строку в безопасную для файловой системы
    """
    return re.subn(r"[^\w -]", "_", path)[0]


def report_from_dict(title: str, data: dict, template: str, fields: list[str]):
    """Формирование отчёта из словаря

    Args:
        title (str): Заголовок отчёта
        data (dict): Данные для отчёта
        template (str): Строка-шаблон для вывода данных
        fields (list): Список полей, которые будут выводится. В порядке как в строке-шаблоне
    """

    report = [title]
    if fields is None:
        fields = ['_key_', ]
        for v in data.values():
            fields += list(v.keys())
            break
    for k, v in data.items():
        z = {**v}
        if '_key_' in fields:
            z['_key_'] = k
        report.append(template.format(*[z.get(field, None) for field in fields]))
    return "\n".join(report)


def chop_list_by_sliding_window(data, chunk_size, step):
    """
    Функция разбиения списка на отрезки скользящим окном
    На входе - последовательность список, размер окна, шаг окна
    """
    # Последовательность разбивается на части до последнего полного окна
    return [data[i:i + chunk_size] for i in range(0, len(data) - chunk_size + 1, step)]


def layer_template(layer_kind, *args, **kwargs):
    return (layer_kind, args, kwargs)


def layer_create(layer_template_data, **variables):
    layer_kind, args, kwargs = layer_template_data
    args = [*args]
    kwargs = {**kwargs}
    for i in range(len(args)):
        v = args[i]
        if isinstance(v, str) and v[:1] == "$":
            args[i] = variables[v[1:]]
    for k, v in kwargs.items():
        if isinstance(v, str) and v[:1] ==  "$":
            kwargs[k] = variables[v[1:]]
    return layer_kind(*args, **kwargs)


def model_create(model_class, *layers, **variables):
    """
    Помогатор для создания модели
    """
    model = model_class()
    for layer in layers:
        model.add(layer_create(layer, **variables))
    return model
