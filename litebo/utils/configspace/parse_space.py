import json
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    CategoricalHyperparameter
from ConfigSpace import EqualsCondition, ForbiddenEqualsClause, ForbiddenAndConjunction


def parse_bool(input):
    if isinstance(input, bool):
        return input
    elif isinstance(input, str):
        if input.lower == 'true':
            return True
        elif input.lower() == 'false':
            return False
        else:
            raise ValueError("Expect string to be 'True' or 'False' but %s received!" % input)
    else:
        ValueError("Expect a bool or str but %s received!" % type(input))


def json_to_space(filepath):
    with open(filepath, 'r') as f:
        json_dict = json.load(f)
    cs = dict_to_space(json_dict)
    return cs


def dict_to_space(dictionary):
    cs = ConfigurationSpace()
    params_dict = dictionary['parameter']
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
