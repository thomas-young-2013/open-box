import json
import collections
from typing import List
from litebo.utils.constants import MAXINT
from litebo.utils.config_space import Configuration, ConfigurationSpace
from litebo.utils.logging_utils import get_logger
from litebo.utils.multi_objective import Hypervolume


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

    def save_json(self, fn: str = "history_container.json"):
        """
        saves runhistory on disk

        Parameters
        ----------
        fn : str
            file name
        """
        data = [(k.get_dictionary(), list(v)) for k, v in self.data.items()]

        with open(fn, "w") as fp:
            json.dump({"data": data}, fp, indent=2)

    @staticmethod
    def load_history_from_json(self, fn: str, cs: ConfigurationSpace):
        """Load and runhistory in json representation from disk.
        Parameters
        ----------
        fn : str
            file name to load from
        cs : ConfigSpace
            instance of configuration space
        """
        try:
            with open(fn) as fp:
                all_data = json.load(fp)
        except Exception as e:
            self.logger.warning(
                'Encountered exception %s while reading runhistory from %s. '
                'Not adding any runs!', e, fn,
            )
            return

        _history_data = collections.OrderedDict()
        # important to use add method to use all data structure correctly
        for k, v in all_data["data"]:
            config = None
            perf = Perf(float(v[0]), float(v[1]), int(v[2]), v[3])
            _history_data[config] = perf
        return _history_data


class MOHistoryContainer(object):
    """
    Multi-Objective History Container
    """
    def __init__(self, task_id, ref_point):
        self.task_id = task_id
        self.data = collections.OrderedDict()
        self.config_counter = 0
        self.pareto = collections.OrderedDict()
        self.num_objs = None
        self.mo_incumbent_value = None
        self.mo_incumbents = None
        self.hypervolume = Hypervolume(ref_point=ref_point)
        self.hv_data = list()
        self.logger = get_logger(self.__class__.__name__)

    def add(self, config: Configuration, perf: List[Perf], hv=0):
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

        # calculate current hypervolume
        pareto_front = self.get_pareto_front()
        if pareto_front:
            hv = self.hypervolume.compute(pareto_front)
        else:
            hv = 0
        self.hv_data.append(hv)

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
