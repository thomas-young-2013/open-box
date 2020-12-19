import collections
from typing import List
from litebo.utils.constants import MAXINT
from litebo.config_space import Configuration
from litebo.utils.logging_utils import get_logger


Perf = collections.namedtuple(
    'perf', ['cost', 'time', 'status', 'additional_info'])


class HistoryContainer(object):
    def __init__(self, task_id):
        self.task_id = task_id
        self.data = collections.OrderedDict()
        self.config_counter = 0
        self.incumbent_value = MAXINT
        self.incumbents = list()
        self.logger = get_logger(self.__class__.__name__)

    def add(self, config: Configuration, perf: Perf):
        if config in self.data:
            self.logger.warning('Repeated configuration detected!')
            return

        self.data[config] = perf
        self.config_counter += 1

        if len(self.incumbents) > 0:
            if perf[0] < self.incumbent_value[0]:
                self.incumbents.clear()
            if perf[0] <= self.incumbent_value[0]:
                self.incumbents.append((config, perf))
                self.incumbent_value = perf
        else:
            self.incumbent_value = perf
            self.incumbents.append((config, perf))

    def get_perf(self, config: Configuration):
        return self.data[config]

    def get_all_configs(self):
        return list(self.data.keys())

    def empty(self):
        return self.config_counter == 0

    def get_incumbents(self):
        return self.incumbents


class MOHistoryContainer(object):
    """
    Multi-Objective History Container
    """
    def __init__(self, task_id):
        self.task_id = task_id
        self.data = collections.OrderedDict()
        self.config_counter = 0
        self.pareto = collections.OrderedDict()
        self.num_objs = None
        self.mo_incumbent_value = None
        self.mo_incumbents = None
        self.logger = get_logger(self.__class__.__name__)

    def add(self, config: Configuration, perf: List[Perf]):
        if self.num_objs is None:
            self.num_objs = len(perf)
            self.mo_incumbent_value = [MAXINT] * self.num_objs
            self.mo_incumbents = [list()] * self.num_objs
        assert self.num_objs == len(perf)

        if config in self.data:
            self.logger.warning('Repeated configuration detected!')
            return

        self.data[config] = perf
        self.config_counter += 1

        # update pareto
        remove_config = []
        for pareto_config, pareto_perf in self.pareto.items():  # todo efficient way?
            if all(pp <= p for pp, p in zip(pareto_perf, perf)):
                break
            elif all(p <= pp for pp, p in zip(pareto_perf, perf)):
                remove_config.append(pareto_config)
        else:
            self.pareto[config] = perf
            self.logger.info('Update pareto: %s, %s.' % (str(config), str(perf)))

        for conf in remove_config:
            self.logger.info('Remove from pareto: %s, %s.' % (str(conf), str(self.pareto[conf])))
            self.pareto.pop(conf)

        # update mo_incumbents
        for i in range(self.num_objs):
            if len(self.mo_incumbents[i]) > 0:
                if perf[i] < self.mo_incumbent_value[i]:
                    self.mo_incumbents[i].clear()
                if perf[i] <= self.mo_incumbent_value[i]:
                    self.mo_incumbents[i].append((config, perf[i], perf))
                    self.mo_incumbent_value[i] = perf[i]
            else:
                self.mo_incumbent_value[i] = perf[i]
                self.mo_incumbents[i].append((config, perf[i], perf))

    def get_perf(self, config: Configuration):
        return self.data[config]

    def get_all_configs(self):
        return list(self.data.keys())

    def get_all_perfs(self):
        return list(self.data.values())

    def empty(self):
        return self.config_counter == 0

    def get_incumbents(self):
        return self.get_pareto()

    def get_mo_incumbents(self):
        return self.mo_incumbents

    def get_mo_incumbent_value(self):
        return self.mo_incumbent_value

    def get_pareto(self):
        return list(self.pareto.items())

    def get_pareto_set(self):
        return list(self.pareto.keys())

    def get_pareto_front(self):
        return list(self.pareto.values())
