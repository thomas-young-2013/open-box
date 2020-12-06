import collections
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
