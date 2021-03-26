import os
import abc
import time
import numpy as np
from typing import List
from collections import OrderedDict
from tensorboardX import SummaryWriter
from litebo.utils.logging_utils import setup_logger, get_logger


class BOBase(object, metaclass=abc.ABCMeta):
    def __init__(self, objective_function, config_space, task_id='task_id', output_dir='logs/',
                 random_state=1, initial_runs=3, max_runs=50,
                 sample_strategy='bo', surrogate_type='prf',
                 history_bo_data: List[OrderedDict] = None,
                 time_limit_per_trial=600):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.task_id = task_id
        _time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        _logger_id = '%s' % task_id
        self.logger_name = None
        self.logger = self._get_logger(_logger_id)
        self.rng = np.random.RandomState(random_state)
        self.writer = SummaryWriter(log_dir='logs/%s' % task_id)

        self.config_space = config_space
        self.objective_function = objective_function
        self.init_num = initial_runs
        self.max_iterations = max_runs
        self.iteration_id = 0
        self.sample_strategy = sample_strategy
        self.history_bo_data = history_bo_data
        self.surrogate_type = surrogate_type
        self.time_limit_per_trial = time_limit_per_trial
        self.config_advisor = None

    def run(self):
        raise NotImplementedError()

    def iterate(self):
        raise NotImplementedError()

    def get_history(self):
        assert self.config_advisor is not None
        return self.config_advisor.history_container

    def get_incumbent(self):
        assert self.config_advisor is not None
        return self.config_advisor.history_container.get_incumbents()

    def _get_logger(self, name):
        logger_name = 'Lite-BO-%s' % name
        self.logger_name = os.path.join(self.output_dir, '%s.log' % str(logger_name))
        setup_logger(self.logger_name)
        return get_logger(logger_name)
