from functools import partial

from ConfigSpace import ConfigurationSpace, Configuration, Constant,\
     CategoricalHyperparameter, UniformFloatHyperparameter, \
     UniformIntegerHyperparameter, InCondition
from ConfigSpace.read_and_write import pcs, pcs_new, json
from litebo.configspace.util import convert_configurations_to_array
from ConfigSpace.util import get_one_exchange_neighbourhood
# get_one_exchange_neighbourhood = partial(get_one_exchange_neighbourhood, stdev=0.05, num_neighbors=8)
