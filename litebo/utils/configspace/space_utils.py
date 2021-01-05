import numpy as np
from typing import List
from litebo.utils.configspace import Configuration, ConfigurationSpace
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    CategoricalHyperparameter
from ConfigSpace import EqualsCondition, ForbiddenEqualsClause, ForbiddenAndConjunction


def parse_bool(input_):
    if isinstance(input_, bool):
        return input
    elif isinstance(input_, str):
        if input_.lower == 'true':
            return True
        elif input_.lower() == 'false':
            return False
        else:
            raise ValueError("Expect string to be 'True' or 'False' but %s received!" % input_)
    else:
        ValueError("Expect a bool or str but %s received!" % type(input_))


def config_space2string(config_space: ConfigurationSpace):
    pass


def string2config_space(space_desc: str):
    pass


def get_config_from_dict(config_dict: dict, config_space: ConfigurationSpace):
    pass


def get_config_space_from_dict(space_dict: dict):
    cs = ConfigurationSpace()
    params_dict = space_dict['parameters']
    for key in params_dict:
        param_dict = params_dict[key]
        param_type = param_dict['type']
        if param_type in ['float', 'int']:
            bound = param_dict['bound']
            optional_args = dict()
            if 'default' in param_dict:
                optional_args['default_value'] = param_dict['default']
            elif 'log' in param_dict:
                optional_args['log'] = parse_bool(param_dict['log'])
            elif 'q' in param_dict:
                optional_args['q'] = param_dict['q']

            if param_type == 'float':
                param = UniformFloatHyperparameter(key, bound[0], bound[1], **optional_args)
            else:
                param = UniformIntegerHyperparameter(key, bound[0], bound[1], **optional_args)

        elif param_type == 'cat':
            choices = param_dict['choice']
            optional_args = dict()
            if 'default' in param_dict:
                optional_args['default_value'] = param_dict['default']
            param = CategoricalHyperparameter(key, choices, **optional_args)

        else:
            raise ValueError("Parameter type %s not supported!" % param_type)

        cs.add_hyperparameter(param)
    return cs
