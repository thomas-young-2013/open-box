from typing import List
import numpy as np
from litebo.utils.config_space import Configuration, ConfigurationSpace


def sample_configurations(configuration_space: ConfigurationSpace, num: int) -> List[Configuration]:
    result = []
    cnt = 0
    while cnt < num:
        config = configuration_space.sample_configuration(1)
        if config not in result:
            result.append(config)
            cnt += 1
    return result


def expand_configurations(configs: List[Configuration], configuration_space: ConfigurationSpace, num: int):
    num_config = len(configs)
    num_needed = num - num_config
    config_cnt = 0
    while config_cnt < num_needed:
        config = configuration_space.sample_configuration(1)
        if config not in configs:
            configs.append(config)
            config_cnt += 1
    return configs


def minmax_normalization(x):
    min_value = min(x)
    delta = max(x) - min(x)
    if delta == 0:
        return [1.0]*len(x)
    return [(float(item)-min_value)/float(delta) for item in x]


def std_normalization(x):
    _mean = np.mean(x)
    _std = np.std(x)
    if _std == 0:
        return np.array([0.]*len(x))
    return (np.array(x) - _mean) / _std


def norm2_normalization(x):
    z = np.array(x)
    normalized_z = z / np.linalg.norm(z)
    return normalized_z
