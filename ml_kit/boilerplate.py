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
    from ml_kit.trainer_common import *

import copy

from IPython.display import clear_output, display
import ipywidgets as widgets
from functools import lru_cache

from tensorflow.keras import utils


def get_tab_run_default(tab_id, model_name, model_data, hp_name, hp):
    return tab_id, f"{model_name}--{hp_name}"


def make_tabs(models, hyper_params_sets, get_tab_run_call=get_tab_run_default,):
    tabs_dict = {}
    for model_name, model_data in models.items():
        for hp_name, hp in hyper_params_sets.items():
            for tab_id in hp['tabs']:
                tab_group, tab_i = get_tab_run_call(tab_id, model_name, model_data, hp_name, hp)
                if tab_group not in tabs_dict:
                    tabs_dict[tab_group] = {}
                widget = tabs_dict[tab_group][tab_i] = widgets.Output()
                with widget:
                    clear_output()
                    print(f"{model_name}--{hp_name}")

    tabs_objs = {k: widgets.Tab() for k in tabs_dict}
    for k, v in tabs_dict.items():
        tab_items_keys = list(sorted(v.keys()))
        tabs_objs[k].children = [v[kk] for kk in tab_items_keys]
        for i in range(0, len(tab_items_keys)):
            tabs_objs[k].set_title(i, f"{k}:{tab_items_keys[i]}")

    return tabs_dict, tabs_objs


def preparation_default(
    model_name,
    model_data,
    hp_name,
    hp,
):
    model_vars = hp['model_vars']
    data_vars = hp['data_vars']
    train_vars = hp['train_vars']
    data_provider = TrainDataProvider(
        x_train = None,
        y_train = None,

        x_val   = None,
        y_val   = None,

        x_test  = None,
        y_test  = None,
    )
    return data_provider


def model_create_default(
    model_name,
    model_data,
    hp_name,
    hp,
    run_name,
    data_provider,
):
    mhd_kwargs = model_data.get('mhd_kwargs', {})
    model_vars = hp['model_vars']
    data_vars = hp['data_vars']
    train_vars = hp['train_vars']
    mhd = ModelHandler(
        name            = run_name,
        model_class     = model_data['model_class'],
        optimizer       = model_data.get('optimizer', ENV[ENV__TRAIN__DEFAULT_OPTIMIZER]),
        loss            = model_data.get('loss',      ENV[ENV__TRAIN__DEFAULT_LOSS]),
        metrics         = model_data.get('metrics',   hp.get('metrics', ENV[ENV__TRAIN__DEFAULT_METRICS])),
        model_template  = model_data['template'],
        model_variables = model_vars,
        batch_size      = train_vars.get('batch_size',ENV[ENV__TRAIN__DEFAULT_BATCH_SIZE]),
        data_provider   = data_provider,
        **mhd_kwargs
    )
    return mhd, ModelHandler


def on_model_update_default(thd):
    pass

def trainer_create_default(
    model_name,
    model_data,
    hp_name,
    hp,
    run_name,
    mhd,
    mhd_class,
    on_model_update_call=on_model_update_default,
):
    model_vars = hp['model_vars']
    data_vars = hp['data_vars']
    train_vars = hp['train_vars']
    thd_kwargs = model_data.get('thd_kwargs', {})

    thd = TrainHandler(
        data_path       = ENV[ENV__MODEL__DATA_ROOT] / model_data.get("data_path", ENV[ENV__TRAIN__DEFAULT_DATA_PATH]),
        data_name       = run_name,
        mhd             = mhd,
        mhd_class       = mhd_class,
        on_model_update = on_model_update_call,
        **thd_kwargs
    )
    return thd, TrainHandler


def train_display_default(*args, **kwargs):
    pass


def print_to_tab_learn(
    model_name,
    model_data,
    hp_name,
    hp,
    run_name,
    mhd,
    thd,
    tab_group,
    tab_name,
    last_metrics,
    best_metrics,
):
    mhd.context.report_to_screen()
    mhd.model.summary()
    utils.plot_model(mhd.model, dpi=60)
    plt.show()


def print_to_tab_fallback(
    model_name,
    model_data,
    hp_name,
    hp,
    run_name,
    mhd,
    thd,
    tab_group,
    tab_name,
    last_metrics,
    best_metrics,
):
    print(f"No print code for tabs group '{tab_group}'")


def print_to_tab_default(
    model_name,
    model_data,
    hp_name,
    hp,
    run_name,
    mhd,
    thd,
    tab_group,
    tab_name,
    last_metrics,
    best_metrics,
):
    tab_print_map = {
        'learn' : print_to_tab_learn
    }
    print_call = tab_print_map.get(tab_group, None)
    if print_call is None:
        print_to_tab_fallback(
            model_name,
            model_data,
            hp_name,
            hp,
            run_name,
            mhd,
            thd,
            tab_group,
            tab_name,
            last_metrics,
            best_metrics,
        )
        return

    print(f"Модель         : {hp['model']}")
    print(f"Гиперпараметры : {hp_name}")
    print(f"Последняя эпоха: {last_metrics['epoch']}")
    print(f"Лучшая эпоха   : {best_metrics['epoch']}")
    print("")
    print_call(
        model_name,
        model_data,
        hp_name,
        hp,
        run_name,
        mhd,
        thd,
        tab_group,
        tab_name,
        last_metrics,
        best_metrics,
    )


def train_routine(
    models,
    hyper_params_sets,
    tabs_dict,
    get_tab_run_call=get_tab_run_default,
    preparation_call=preparation_default,
    on_model_update_call=on_model_update_default,
    model_create_call=model_create_default,
    trainer_create_call=trainer_create_default,
    train_display_call=train_display_default,
    print_to_tab_call=print_to_tab_default,
    ):
    dummy_output = widgets.Output()

    for model_name in models:
        for hp_name in hyper_params_sets:
            def main_logic(model_name, hp_name):

                hp = copy.deepcopy(hyper_params_sets[hp_name])
                if 'model' not in hp:
                    hp['model'] = model_name

                model_data = models[model_name]

                _, run_name = get_tab_run_call(None, model_name, model_data, hp_name, hp)
                print(f"Running {run_name}")

                model_vars = {**copy.deepcopy(model_data.get('vars', {})), **copy.deepcopy(hp.get('model_vars', {}))}

                hp['model_vars'] = model_vars

                if 'data_vars' not in hp:
                    hp['data_vars'] = {}

                if 'train_vars' not in hp:
                    hp['train_vars'] = {}
                train_vars = hp['train_vars']

                data_provider = preparation_call(
                    model_name,
                    model_data,
                    hp_name,
                    hp,
                )

                mhd, mhd_class = model_create_call(
                    model_name,
                    model_data,
                    hp_name,
                    hp,
                    run_name,
                    data_provider,
                )

                thd, thd_class = trainer_create_call(
                    model_name,
                    model_data,
                    hp_name,
                    hp,
                    run_name,
                    mhd,
                    mhd_class,
                    on_model_update_call,
                )

                # Check if saved results are enough even if model is not saved
                best_simple_avail = thd.can_load(S_BEST, dont_load_model=True)
                regular_simple_avail = thd.can_load(S_REGULAR, dont_load_model=True)
                enough = False
                can_pred = True
                if best_simple_avail or regular_simple_avail:
                    thd_tmp = TrainHandler(
                        # NOTE: used only to load and check metrics
                        data_path       = thd.data_path,
                        data_name       = thd.data_name,
                        mhd             = thd._mhd_class(
                            name=thd.data_name,
                            model_class=None,
                            optimizer=None,
                            loss=None,
                            metrics=thd._mhd.metrics,
                        ),
                    )
                    for load_path in S_REGULAR, S_BEST:
                        thd_tmp.load(load_path, dont_load_model=True)
                        if thd_tmp.mhd.context.epoch >= train_vars.get("epochs", ENV[ENV__TRAIN__DEFAULT_EPOCHS]) \
                        or thd.is_enough(train_vars.get("target", ENV[ENV__TRAIN__DEFAULT_TARGET])):
                            enough = True
                            break
                    del thd_tmp

                if not enough:
                # Train if not enough
                    thd.train(
                        from_scratch    = train_vars.get("from_scratch", ENV[ENV__TRAIN__DEFAULT_FROM_SCRATCH]),
                        epochs          = train_vars.get("epochs", ENV[ENV__TRAIN__DEFAULT_EPOCHS]),
                        target          = train_vars.get("target", ENV[ENV__TRAIN__DEFAULT_TARGET]),
                        save_step       = train_vars.get("save_step", ENV[ENV__TRAIN__DEFAULT_SAVE_STEP]),
                        display_callback= train_display_call,
                    )
                else:
                # Otherwise - load in following order
                    if thd.can_load(S_REGULAR):
                        thd.load(S_REGULAR)
                    elif regular_simple_avail:
                        thd.load(S_REGULAR, dont_load_model=True)
                        can_pred = False
                    elif thd.can_load(S_BEST):
                        thd.load(S_BEST)
                    else:
                        thd.load(S_BEST, dont_load_model=True)
                        can_pred = False

                last_metrics = {}
                best_metrics = {}

                # Display train results
                full_history = copy.deepcopy(mhd.context.history)

                last_metrics['epoch'] = mhd.context.epoch
                last_metrics['pred'] = None
                if can_pred:
                    mhd.update_data(force=True)
                    last_metrics['pred'] = mhd.context.test_pred
                else:
                    mhd.create()    # NOTE: required to print model info

                best_metrics['epoch'] = None
                best_metrics['pred'] = None

                if thd.can_load(S_BEST):
                    thd.load_best()
                    best_metrics['epoch'] = mhd.context.epoch
                    mhd.update_data(force=True)
                    best_metrics['pred'] = mhd.context.test_pred
                elif thd.can_load(S_BEST, dont_load_model=True):
                    thd_tmp = TrainHandler(
                        # NOTE: used only to load data and hold best value
                        data_path       = thd.data_path,
                        data_name       = thd.data_name,
                        mhd             = thd._mhd_class(
                            name=thd.data_name,
                            model_class=None,
                            optimizer=None,
                            loss=None,
                            metrics=thd._mhd.metrics,
                        ),
                    )
                    thd_tmp.load(S_BEST, dont_load_model=True)
                    best_metrics['epoch'] = thd_tmp.mhd.context.epoch
                    del thd_tmp

                mhd.context.report_history = full_history

                # Cleanup anything in buffer
                with dummy_output:
                    plt.show()
                    clear_output()

                for tab_id in hp['tabs']:
                    tab_group, tab_name = get_tab_run_call(tab_id, model_name, model_data, hp_name, hp)
                    with tabs_dict[tab_group][tab_name]:
                        clear_output()
                        print_to_tab_call(
                            model_name,
                            model_data,
                            hp_name,
                            hp,
                            run_name,
                            mhd,
                            thd,
                            tab_id,
                            tab_name,
                            last_metrics,
                            best_metrics,
                        )

                    # Cleanup anything that could be left in buffer
                    with dummy_output:
                        plt.show()
                        clear_output()

                mhd.context.report_history = None
                mhd.unload_model()

            main_logic(model_name, hp_name)
