from litebo.acquisition_function.acquisition import *
from litebo.acq_maximizer.ei_optimization import InterleavedLocalAndRandomSearch
from litebo.surrogate.base.rf_with_instances import RandomForestWithInstances
from litebo.surrogate.base.build_gp import create_gp_model
from litebo.utils.util_funcs import get_types
from litebo.utils.constants import MAXINT


def build_acq_func(func_str='ei', model=None):
    func_str = func_str.lower()
    if func_str == 'ei':
        acq_func = EI
    elif func_str == 'eips':
        acq_func = EIPS
    elif func_str == 'logei':
        acq_func = LogEI
    elif func_str == 'pi':
        acq_func = PI
    elif func_str == 'lcb':
        acq_func = LCB
    elif func_str == 'lpei':
        acq_func = LPEI
    else:
        raise ValueError('Invalid string %s for acquisition function!' % func_str)

    return acq_func(model=model)


def build_optimizer(func_str='local_random', acq_func=None, config_space=None, rng=None):
    assert config_space is not None
    func_str = func_str.lower()

    if func_str == 'local_random':
        optimizer = InterleavedLocalAndRandomSearch
    else:
        raise ValueError('Invalid string %s for acq_maximizer!' % func_str)

    return optimizer(acquisition_function=acq_func,
                     config_space=config_space,
                     rng=rng)


def build_surrogate(func_str='prf', config_space=None, rng=None, history_hpo_data=None):
    assert config_space is not None
    func_str = func_str.lower()
    types, bounds = get_types(config_space)
    seed = rng.randint(MAXINT)
    if func_str == 'prf':
        return RandomForestWithInstances(types=types, bounds=bounds, seed=seed)
    elif 'gp' in func_str:
        return create_gp_model(model_type=func_str,
                               config_space=config_space,
                               types=types,
                               bounds=bounds,
                               rng=rng)
    elif func_str.startswith('tlbo'):
        print('the current surrogate is', func_str)
        from litebo.surrogate.tlbo.rgpe import RGPE
        inner_surrogate_type = func_str[5:]
        return RGPE(config_space, history_hpo_data, seed,
                    surrogate_type=inner_surrogate_type, num_src_hpo_trial=-1)
    else:
        raise ValueError('Invalid string %s for surrogate!' % func_str)
