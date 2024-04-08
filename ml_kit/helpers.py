################################################################################
# General helper functions / classes

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

import copy
import time
import re

ENV__MODEL__CREATE_AUTOIMPORT = 'ENV__MODEL__CREATE_AUTOIMPORT'
ENV__MODEL__FALLBACK_ARGS = 'ENV__MODEL__FALLBACK_ARGS'

ENV[ENV__MODEL__CREATE_AUTOIMPORT] = False
ENV[ENV__MODEL__FALLBACK_ARGS] = {
    "<class 'keras.src.layers.regularization.dropout.Dropout'>": {'seed': 1}
}


S_CHAIN = 'chain'
S_INPUT = 'input'
S_OUTPUT = 'output'
S_OUTPUTS = 'outputs'
S_LAYER = 'layer'
S_LAYERS = 'layers'
S_MAKE_INSTANCE = 'make_instance'
S_MODEL = 'model'
S_MODEL_CLASS = 'model_class'
S_MODEL_TEMPLATE = 'model_template'
S_NAME = "name"
S_INPUTS = 'inputs'
S_INPUTS_ORDER = 'inputs_order'
S_PARENTS = 'parents'
S_CHILDREN = 'children'
S_IPARENTS = 'iparents'
S_ICHILDREN = 'ichildren'
S_NAMED_LAYERS = 'named_layers'
S_INSTANCE = 'instance'
S_KIND = 'kind'
S_SUBMODELS = 'submodels'
S_SIMPLE = 'simple'
S_COMPLEX = 'complex'
S_DATA = 'data'
S_VARS = 'vars'

S_ARGS = 'args'
S_KWARGS = 'kwargs'


def mult(*values):
    if len(values) == 1 and isinstance(values, (list, tuple)):
        values = values[0]
    result = 1
    for v in values:
        result *= v
    return result


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
        if ENV[ENV__DEBUG_PRINT]:
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
    return re.subn(r"[^\w .-]", "_", path)[0]


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


class LayerDummy:
    _id = 0
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.id = LayerDummy._id
        self.parent = None
        LayerDummy._id += 1
        print(f"Layer with id {self.id} were created"
              f" ( {', '.join([str(v) for v in args] + [str(k) + '=' + str(v) for k, v in kwargs.items()])})")

    def __call__(self, parent, *args, **kwargs):
        print(f"Layer with id {self.id} were called "
              f" ( parent={parent}, {', '.join([str(v) for v in args] + [str(k) + '=' + str(v) for k, v in kwargs.items()])})")
        self.parent = parent
        return self

    def __str__(self):
        if self.parent is None:
            return(f"Layer({self.id})")
        else:
            return f"{self.parent}.Layer({self.id})"

    def to_str(self):
        return f"[[ LayerDummy id={self.id}, parent={self.parent}, " \
               f"({', '.join([str(v) for v in self.args] + [str(k) + '=' + str(v) for k, v in self.kwargs.items()])}) ]]"


class ModelDummy:
    _id = 0
    def __init__(self, inputs, outputs, *args, **kwargs):
        self.inputs = inputs
        self.outputs = outputs
        self.args = args
        self.kwargs = kwargs
        self.id = ModelDummy._id
        self._fit_iteration = 0
        ModelDummy._id += 1
        print(f"Model with id {self.id} were created"
              f" ( inputs={inputs}, outputs={outputs}, {', '.join([str(v) for v in args] + [str(k) + '=' + str(v) for k, v in kwargs.items()])})")

    def __str__(self):
        return f"[[ ModelDummy id={self.id},  inputs={self.inputs}, outputs={self.outputs}]]"

    def __str__(self):
        return f"[[ LayerDummy id={self.id},  inputs={self.inputs}, outputs={self.outputs}, " \
               f"({', '.join([str(v) for v in self.args] + [str(k) + '=' + str(v) for k, v in self.kwargs.items()])}) ]]"

    def compile(self, *args, **kwargs):
        print(f"ModelDummy(id={self.id}.compile({', '.join([str(v) for v in args] + [str(k) + '=' + str(v) for k, v in kwargs.items()])})")
        self._fit_iteration = 0

    def fit(self, *args, **kwargs):
        print(f"ModelDummy(id={self.id}.fit({', '.join([str(v) for v in args] + [str(k) + '=' + str(v) for k, v in kwargs.items()])})")
        self._fit_iteration += 1
        return ArbitraryClass(
            history={
                "loss":         [2**31-self._fit_iteration],
                "val_loss":     [2**31-self._fit_iteration],
                "accuracy":     [0.001*self._fit_iteration],
                "val_accuracy": [0.001*self._fit_iteration],
                "mae":          [2**31-self._fit_iteration],
                "val_mae":      [2**31-self._fit_iteration],
            },
        )


def layer_template(layer_kind, *args, **kwargs):
    return (layer_kind, args, kwargs)

def subst_vars(value, variables, recurse=False, var_sign="$", raise_error_when_not_found=True, nested=False):
    result = value
    if isinstance(value, str):
        if value[:len(var_sign)] == var_sign:
            k = value[len(var_sign):]   # TODO: regex matching for var_sign{(\w+)}
            if k in variables:
                result = variables[k]
                if recurse:
                    result = subst_vars(result, variables, recurse, var_sign, raise_error_when_not_found, nested=True)
            elif raise_error_when_not_found:
                raise KeyError(f"Variable '{k}' was not found within variables!")
    elif recurse or not nested:
        if isinstance(value, list):
            result = [subst_vars(v, variables, recurse, var_sign, raise_error_when_not_found, nested=True) for v in value]
        elif isinstance(value, tuple):
            result = (subst_vars(v, variables, recurse, var_sign, raise_error_when_not_found, nested=True) for v in value)
        elif isinstance(value, dict):
            result = {
                subst_vars(k, variables, recurse, var_sign, raise_error_when_not_found, nested=True) :
                subst_vars(v, variables, recurse, var_sign, raise_error_when_not_found, nested=True)
                for k, v in value.items()
            }
    return result

def layer_create(layer_template_data, **variables):
    layer_kind, args, kwargs = layer_template_data
    layer_kind = subst_vars([layer_kind], variables, recurse=True)[0]
    if layer_kind is None:
        return None
    args = [*args]
    kwargs = {k:v for k,v in kwargs.items() if isinstance(k, str) and k[:1] != "_"}
    args = subst_vars(args, variables, recurse=True)
    kwargs = subst_vars(kwargs, variables, recurse=True)
    layer_kind_type = str(type(layer_kind))
    if layer_kind_type in ENV[ENV__MODEL__FALLBACK_ARGS]:
        for k, v in ENV[ENV__MODEL__FALLBACK_ARGS][layer_kind_type].items():
            if k not in kwargs:
                kwargs[k] = v
    if isinstance(layer_kind, str) and layer_kind[:1]=="<" and layer_kind[:-1]==">":
        if ENV[ENV__MODEL__CREATE_AUTOIMPORT]:
            pass # TODO: try to import according to str # NOTE: it's really unsafe but can be used with YAML
        else:
            raise ValueError("'layer_kind' should be a class!")
    return layer_kind(*args, **kwargs)


def _create_layers_chain(parent, name_suffix, *layers, **variables):
    layers_chain = []
    named = {}
    inputs_dict = {}
    outputs_dict = {}
    auto_names = {}
    for layer in layers:
        if layer[2].get('name', None) is not None and '_name_' not in layer[2]:
            layer[2]['_name_'] = f"{layer[2]['name']}"
        if layer[2].get('_name_', None) is not None and 'name' not in layer[2]:
                layer[2]['name'] = f"_{layer[2]['_name_']}_"
        if layer[2].get('_input_', None) is not None:
            layer[2]['_parent_'] = None
            if 'name' not in layer[2]:
                layer[2]['name'] = f"_input_{layer[2]['_input_']}_"
            if '_name_' not in layer[2]:
                layer[2]['_name_'] = f"_input_{layer[2]['_input_']}_"
        if layer[2].get('_output_', None) is not None:
            if 'name' not in layer[2]:
                layer[2]['name'] = f"_output_{layer[2]['_output_']}_"
            if '_name_' not in layer[2]:
                layer[2]['_name_'] = f"_output_{layer[2]['_output_']}_"
        if '_parent_' in layer[2]:
            layer_parent = subst_vars([layer[2]['_parent_']], {**variables, **named}, recurse=True)[0]
        else:
            layer_parent = parent
        if 'name' not in layer[2]:
            an = f"{layer.__name__}"
            if an not in auto_names:
                auto_names[an] = 0
            auto_names[an] += 1
            layer[2]['name'] = f"_{an}_{auto_names[an]}_"
        if layer[2]['name'] is None:
            del layer[2]['name']
        else:
            if name_suffix != "":
                layer[2]['name'] = f"{name_suffix}/{layer[2]['name']}"
        if layer_parent is None:
            layer_instance = layer_create(layer, **{**variables, **named})
        else:
            layer_instance = layer_create(layer, **{**variables, **named})(layer_parent)
        if layer_instance is None:
            continue
        if '_name_' in layer[2]:
            named[layer[2]['_name_']] = layer_instance
        layers_chain.append(layer_instance)
        if layer[2].get('_spinoff_', False) is not True:
            parent = layer_instance
        if layer[2].get('_input_', None) is not None:
            inputs_dict[layer[2]['_input_']] = layer_instance
        if layer[2].get('_output_', None) is not None:
            outputs_dict[layer[2]['_output_']] = layer_instance

    if len(inputs_dict) == 0:
        inputs = [layers_chain[0]]
    else:
        inputs = [inputs_dict[k] for k in sorted(inputs_dict.keys())]

    if len(outputs_dict) == 0:
        outputs = [parent]
    else:
        outputs = [outputs_dict[k] for k in sorted(outputs_dict.keys())]

    return layers_chain, named, inputs, outputs, parent

def _lookup_vars(value, result:list[str], lookup_vars:dict):
    if isinstance(value, (list, tuple)):
        for v in value:
            _lookup_vars(v, result, lookup_vars)
    if isinstance(value, (dict)):
        for v in list(value.keys()) + (value.values()):
            _lookup_vars(v, result, lookup_vars)
    else:
        if isinstance(value, str) and value in lookup_vars:
            if value[1:] not in result:
                result.append(value[1:])

def _get_iparents(templates, layers:list[str]):
    result = []
    lookup_vars = ["$"+k for k in layers]
    for _, args, kwargs in templates:
        for var_name in lookup_vars:
            if var_name in result:
                continue
            _lookup_vars(list(args) + list(kwargs.values()), result, lookup_vars)
    return result


def simple_model_create(model_class, templates, model_kwargs=None, **variables):
    """
    Помогатор для создания модели
    """
    if not isinstance(templates, (list, tuple, dict)):
        raise ValueError("'templates' should be a list, tuple or dict!")

    inputs = None
    outputs = None
    model = None
    named_layers = {}
    model_kwargs = model_kwargs or {}

    if isinstance(templates, (list, tuple)):
        if len(templates) < 2:
            raise ValueError("Create at least 2 layers!")
        if hasattr(model_class, 'add'): # NOTE: this is for Sequential model
            model = model_class()
            for layer in templates:
                layer_instance = layer_create(layer, **variables)
                model.add(layer_instance)
                if '_name_' in layer[2]:
                    named_layers[layer[2]['_name_']] = layer_instance
        else:
            layers_chain, named_layers, inputs, outputs, last_in_chain = _create_layers_chain(None, "", *templates, **{**variables, **named_layers})
            model = model_class(inputs, outputs)

    elif isinstance(templates, dict):
        branches = {}
        inputs = []
        outputs = []

        # Initialize inputs, output and branches
        for k, v in templates.items():
            if S_CHAIN in v:
                raise ValueError(f"Prohibited field '{S_CHAIN}' were found in branch '{k}'!")
            if not isinstance(v.get(S_LAYERS, None), (list, tuple)):
                raise ValueError(f"Expected to find '{S_LAYERS}' in branch '{k}' and it should be a list or tuple!")
            if len(v[S_LAYERS]) < 1:
                raise ValueError(f"Create at least 2 layers for branch '{k}'!")

            branches[k] = {
                S_PARENTS: None,
                S_CHILDREN:None,
                **copy.deepcopy(v),
                S_IPARENTS: [],
                S_ICHILDREN: [],
            }

            if branches[k][S_PARENTS] is not None:
                if not isinstance(branches[k][S_PARENTS], list):
                    if isinstance(branches[k][S_PARENTS], tuple):
                        branches[k][S_PARENTS] = list(branches[k][S_PARENTS])
                    else:
                        branches[k][S_PARENTS] = [branches[k][S_PARENTS]]
            else:
                branches[k][S_PARENTS] = []

            if branches[k][S_CHILDREN] is not None:
                if not isinstance(branches[k][S_CHILDREN], list):
                    if isinstance(branches[k][S_CHILDREN], tuple):
                        branches[k][S_CHILDREN] = list(branches[k][S_CHILDREN])
                    else:
                        branches[k][S_CHILDREN] = [branches[k][S_CHILDREN]]
            else:
                branches[k][S_CHILDREN] = []

            if branches[k].get(S_INPUT, False):
                inputs.append(k)

            if branches[k].get(S_OUTPUT, False):
                outputs.append(k)

        if len(inputs) == 0:
            raise ValueError(f"Specify at least one branch as input (set {S_INPUT} field to True)")

        if len(outputs) == 0:
            raise ValueError("Output branch is not set! Specify strictly one branch as output (set 'output' field to True)")

        # Chain branches (find all children) and do DRC

        ## First - update references using parents fields
        for k, v in branches.items():
            iparents = v[S_IPARENTS] = _get_iparents(v[S_LAYERS], branches)
            parents = v[S_PARENTS] = list(set(v[S_PARENTS]))
            v[S_IPARENTS] = list(set(v[S_IPARENTS]))
            if len(parents) + len(iparents) == 0:
                if k not in inputs:
                    raise ValueError(f"Non-input branch without parent set! ('{k}')")
            for p in parents:
                if p not in branches:
                    raise ValueError(f"Parent '{p}' wasn't found for branch '{k}'!")
            for p in parents:
                branches[p][S_CHILDREN].append(k)
            for p in iparents:
                branches[p][S_ICHILDREN].append(k)

        ## Make references distinct
        for v in branches.values():
            v[S_CHILDREN] = list(set(v[S_CHILDREN]))
            v[S_ICHILDREN] = list(set(v[S_ICHILDREN]))

        ## Second - update references using child fields
        for k, v in branches.items():
            if len(v[S_CHILDREN]) + len(v[S_ICHILDREN]) == 0:
                if k not in outputs:
                    # TODO: permit childless branches if they are named?
                    raise ValueError(f"Found childless branch '{k}', but it's not specified as output!")
            for c in v[S_CHILDREN]:
                if c not in branches:
                    raise ValueError(f"Children '{c}' wasn't found for branch '{k}'!")
                if c not in branches[c][S_PARENTS]:
                    branches[c][S_PARENTS].append(k)

        ## Make references distinct
        for v in branches.values():
            v[S_PARENTS] = list(set(v[S_PARENTS]))
            v[S_IPARENTS] = list(set(v[S_IPARENTS]))

        ## Make sure inputs don't have parents
        for k in inputs:
            if len(branches[k][S_PARENTS]) + len(branches[k][S_IPARENTS]) > 0:
                raise ValueError(
                    f"Parents were found for input branch '{k}'!"
                    f" Direct parents: {branches[k][S_PARENTS]}"
                    f" Indirect parents (specified via vars): {branches[k][S_IPARENTS]}"
                )

        # Check not supported cases:
        for k, v in branches.items():
            if len(v[S_PARENTS]) > 1:
                continue
                raise NotImplementedError(
                    f"Multiple parents were found for branch {k}! It's not supported (yet)"
                    f" ({v[S_PARENTS]})")

        # TODO: check for circular references

        # Create branches at last
        def _incomplete_branches(branches):
            return [k for k in branches if S_CHAIN not in branches[k]]

        incomplete_branches = _incomplete_branches(branches)
        created_branches = {}
        while len(incomplete_branches) > 0:
            for k in incomplete_branches:
                if k not in inputs:
                    for p in branches[k][S_PARENTS]:
                        if S_CHAIN not in branches[p]:
                            continue
                    for p in branches[k][S_IPARENTS]:
                        if S_CHAIN not in branches[p]:
                            continue

                v = branches[k]
                if 'variables' in v:
                    branch_vars = {**variables, **v['variables']}
                else:
                    branch_vars = variables
                overlapping = [k for k in created_branches if k in branch_vars]
                if len(overlapping) > 0:
                    raise ValueError(f"Error creating branch {k}: existing branches ({overlapping}) are overlapping with variables!")
                branch_vars = {**branch_vars, **created_branches}
                if len(v[S_PARENTS])==0 or k in inputs:
                    parents = None
                else:
                    parents = branches[v[S_PARENTS][0]][S_CHAIN][-1]
                branches[k][S_CHAIN], named, branches[k][S_INPUTS], branches[k][S_OUTPUTS], last_in_chain = \
                _create_layers_chain(
                    parents,
                    k,
                    *v[S_LAYERS],
                    **{**branch_vars, **named_layers})
                for nk, nv in named.items():
                    named_layers[k+"/"+nk] = nv

                created_branches[k] = last_in_chain

            prev_incomplete = incomplete_branches
            incomplete_branches = _incomplete_branches(branches)
            if len(prev_incomplete) == len(incomplete_branches):
                line_break = "\n"
                raise ValueError(
                    f"Those branches can't be created (check for circular references): {incomplete_branches}!"
                    f" {[line_break + k + ': [' + ', '.join([kk for kk in branches[k][S_PARENTS]  if S_CHAIN not in branches[k][S_PARENTS]] + [kk for kk in branches[k][S_IPARENTS] if S_CHAIN not in branches[k][S_IPARENTS]]) for k in incomplete_branches]}]"
                    )

        model_inputs = []
        for k in inputs:
            model_inputs += branches[k][S_INPUTS]

        model_outputs = []
        for k in outputs:
            model_outputs += branches[k][S_OUTPUTS]

        model = model_class(model_inputs, model_outputs, **model_kwargs)

        inputs_order = []
        for mi in model_inputs:
            inputs_order.append(mi.name)

    return {S_MODEL: model, S_INPUTS_ORDER: inputs_order, S_NAMED_LAYERS: named_layers,
            S_INPUTS: model_inputs, S_OUTPUTS: model_outputs, S_DATA: {},
            S_VARS: {**variables}, 'model_kwargs': {**model_kwargs}}


def complex_model_create(
    model_class, submodels, **variables
):
    # Create models
    data = {}
    for k, v in submodels.items():
        while True:

            if isinstance(k, str):
                if k[:1] == "_" and k[-1:] == "_":
                    break

            model_kwargs = v.get(S_KWARGS, {})
            if S_NAME not in model_kwargs:
                model_kwargs[S_NAME] = k

            if S_NAME in model_kwargs and model_kwargs[S_NAME] is None:
                del model_kwargs[S_NAME]

            if S_MODEL_TEMPLATE in v:

                data[k] = simple_model_create(
                    v[S_MODEL_TEMPLATE]['model_class'],
                    v[S_MODEL_TEMPLATE]['template'],
                    model_kwargs,
                    **{**v[S_MODEL_TEMPLATE].get(S_VARS, {}), **variables}
                )

            elif S_MODEL_CLASS in v:

                model_inputs = []
                for ep_path in v[S_INPUTS]:
                    ep = data
                    for ep_item in ep_path:
                        ep = ep[ep_item]
                    model_inputs.append(ep)

                model_outputs = []
                if S_OUTPUTS in v:
                    for ep_path in v[S_OUTPUTS]:
                        ep = data
                        for ep_item in ep_path:
                            ep = ep[ep_item]
                        model_outputs.append(ep)

                args = []

                args.append(model_inputs)

                if len(model_outputs) > 0:
                    args.append(model_outputs)

                if isinstance(v[S_MODEL_CLASS], str):
                    model_class = data[v[S_MODEL_CLASS]][S_MODEL]
                else:
                    model_class = v[S_MODEL_CLASS]

                data[k] = {}
                data[k][S_MODEL] = model_class(*args, **model_kwargs)
                data[k][S_INPUTS] = model_inputs
                data[k][S_OUTPUTS] = model_outputs
                data[k][S_INPUTS_ORDER] = [item.name for item in model_inputs]
                data[k][S_NAMED_LAYERS] = {}
                data[k][S_DATA] = {}

            else:
                raise ValueError("No supported scheme for model found!")

            # Recurse if necessary
            if v.get(S_MAKE_INSTANCE, False) is not True:
                break
            else:
                # Make default instance of this model
                # (will use `if S_MODEL_CLASS in v` code branch below)

                v = {
                    S_MODEL_CLASS: k,
                    S_KWARGS: to_dict(name = None),
                }
                v[S_INPUTS] = [
                    [k, S_INPUTS, i] for i in range(len(data[k][S_INPUTS]))
                ]

                k = f"{k}_i_"


    output_model = data[submodels[f"_{S_OUTPUT}_"]]
    named_layers = {}
    for k in submodels.keys():
        if isinstance(k, str):
            if k[:1] == "_" and k[-1:] == "_":
                continue
        if S_NAMED_LAYERS not in data[k]:
            continue
        for kk, vv in data[k][S_NAMED_LAYERS].items():
            named_layers[f"{k}/{kk}"] = vv

    result = {**output_model}
    result[S_NAMED_LAYERS] = named_layers
    result[S_DATA] = data
    result[S_VARS] = {**variables}
    return result


def model_create(model_class, templates, model_kwargs=None, **variables):
    if isinstance(templates, dict):
        kind = templates.get(f"_{S_KIND}_", None)
    else:
        kind = None

    if kind in (None, S_SIMPLE):
        return simple_model_create(model_class, templates, model_kwargs, **variables)
    elif kind == S_COMPLEX:
        return complex_model_create(model_class, templates, **variables)
    else:
        raise ValueError(f"Unsupported template kind: '{kind}'")


if STANDALONE:
    if __name__ == "__main__":
        # Test Cases:
        model_branches = to_dict(
            in1 = to_dict(
                input  = True,
                layers = [
                    layer_template(LayerDummy, "in_1__layer_1", data=11),
                    layer_template(LayerDummy, "in_1__layer_2", data=12),
                    layer_template(LayerDummy, "in_1__layer_3", data=13),
                    layer_template(LayerDummy, "in_1__layer_4", data=14),
                ],
            ),

            in2 = to_dict(
                input  = True,
                layers = [
                    layer_template(LayerDummy, "in_2__layer_1", data=21),
                    layer_template(LayerDummy, "in_2__layer_2", data=22),
                    layer_template(LayerDummy, "in_2__layer_3", data=23),
                ],
            ),

            branch_1 = to_dict(
                parents = "in1",
                layers = [
                    layer_template(LayerDummy, "br_1__layer_1", data=101),
                    layer_template(LayerDummy, "br_1__layer_2", data=102, aux="$var1"),
                    layer_template(LayerDummy, "br_1__layer_3", "$in1", "$in2", data=103),
                ]
            ),

            branch_2 = to_dict(
                parents = ["branch_1", ],
                output = True,
                layers = [
                    layer_template(LayerDummy, "br_2__layer_1", data=201, aux="$lvar1"),
                    layer_template(LayerDummy, "br_2__layer_2", data=202, aux="$in2"),
                    layer_template(LayerDummy, "br_2__layer_3", data=203),
                    layer_template(LayerDummy, "br_2__layer_4", ["$in1", "$in2"]),
                    layer_template(LayerDummy, "br_2__layer_5", ["$lvar2", "$in2"]),
                ],
                variables = to_dict(
                    lvar1 = "lvar1_value",
                    lvar2 = "$in1",
                )
            )
        )

        variables = to_dict(var1="var1_value")

        sequential_model = model_create(ModelDummy, model_branches['in2']['layers'], **variables)
        print(sequential_model)

        branched_model   = model_create(ModelDummy, model_branches, **variables)
        print(branched_model)
