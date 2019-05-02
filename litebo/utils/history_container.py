import collections
from litebo.configspace import Configuration
from litebo.utils.constants import MAXINT


Perf = collections.namedtuple(
    'perf', ['cost', 'time', 'status', 'additional_info'])


class HistoryContainer(object):
    def __init__(self, task_id):
        self.task_id = task_id
        self.data = collections.OrderedDict()
        self.config_counter = 0
        self.incumbent_value = MAXINT
        self.incumbents = list()

    def add(self, config: Configuration, perf: Perf):
        if config in self.data:
            raise ValueError('Repeated configuration detected!')
        self.data[config] = perf
        self.config_counter += 1

        if len(self.incumbents) > 0:
            if perf < self.incumbent_value:
                self.incumbents.clear()
            if perf <= self.incumbent_value:
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
