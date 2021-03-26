import sys
import json
import collections
from typing import List, Union
import numpy as np
from litebo.utils.constants import MAXINT
from litebo.utils.config_space import Configuration, ConfigurationSpace
from litebo.utils.logging_utils import get_logger
from litebo.utils.multi_objective import Hypervolume, get_pareto_front
from litebo.utils.config_space.space_utils import get_config_from_dict
from litebo.utils.visualization.plot_convergence import plot_convergence


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

    def get_all_perfs(self):
        return list(self.data.values())

    def get_all_configs(self):
        return list(self.data.keys())

    def empty(self):
        return self.config_counter == 0

    def get_incumbents(self):
        return self.incumbents

    def get_str(self):
        from terminaltables import AsciiTable
        incumbents = self.get_incumbents()
        if not incumbents:
            return 'No incumbents in history. Please run optimization process.'

        configs_table = []
        nil = "-"
        parameters = list(incumbents[0][0].get_dictionary().keys())
        for para in parameters:
            row = []
            row.append(para)
            for config, perf in incumbents:
                val = config.get(para, None)
                if val is None:
                    val = nil
                if isinstance(val, float):
                    val = "%.6f" % val
                elif not isinstance(val, str):
                    val = str(val)
                row.append(val)
            configs_table.append(row)
        configs_title = ["Parameters"] + ["" if i else "Optimal Value" for i, _ in enumerate(incumbents)]

        table_data = ([configs_title] +
                      configs_table +
                      [["Optimal Objective Value"] + [perf for config, perf in incumbents]] +
                      [["Num Configs"] + [str(self.config_counter)]]  # todo: no failed configs
                      )

        M = 2
        raw_table = AsciiTable(
            table_data
            # title="Result of Optimization"
        ).table
        lines = raw_table.splitlines()
        title_line = lines[1]
        st = title_line.index("|", 1)
        col = "Optimal Value"
        L = len(title_line)
        lines[0] = "+" + "-" * (L - 2) + "+"
        new_title_line = title_line[:st + 1] + (" " + col + " " * (L - st - 3 - len(col))) + "|"
        lines[1] = new_title_line
        bar = "\n" + lines.pop() + "\n"
        finals = lines[-M:]
        prevs = lines[:-M]
        render_table = "\n".join(prevs) + bar + bar.join(finals) + bar
        return render_table

    def __str__(self):
        return self.get_str()

    __repr__ = __str__

    def plot_convergence(
            self,
            xlabel="Number of iterations $n$",
            ylabel=r"Min objective value after $n$ iterations",
            ax=None, name=None, alpha=0.2, yscale=None,
            color=None, true_minimum=None,
            **kwargs):
        """Plot one or several convergence traces.

        Parameters
        ----------
        args[i] :  `OptimizeResult`, list of `OptimizeResult`, or tuple
            The result(s) for which to plot the convergence trace.

            - if `OptimizeResult`, then draw the corresponding single trace;
            - if list of `OptimizeResult`, then draw the corresponding convergence
              traces in transparency, along with the average convergence trace;
            - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
              an `OptimizeResult` or a list of `OptimizeResult`.

        ax : `Axes`, optional
            The matplotlib axes on which to draw the plot, or `None` to create
            a new one.

        true_minimum : float, optional
            The true minimum value of the function, if known.

        yscale : None or string, optional
            The scale for the y-axis.

        Returns
        -------
        ax : `Axes`
            The matplotlib axes.
        """
        losses = list(self.data.values())

        n_calls = len(losses)
        iterations = range(1, n_calls + 1)
        mins = [np.min(losses[:i]) for i in iterations]
        max_mins = max(mins)
        cliped_losses = np.clip(losses, None, max_mins)
        return plot_convergence(iterations, mins, cliped_losses, xlabel, ylabel, ax, name, alpha, yscale, color,
                                true_minimum, **kwargs)

    def visualize_jupyter(self):
        try:
            import hiplot as hip
        except ModuleNotFoundError:
            if sys.version_info < (3, 6):
                raise ValueError("HiPlot requires Python 3.6 or newer. "
                                 "See https://facebookresearch.github.io/hiplot/getting_started.html")
            self.logger.error("Please run 'pip install hiplot'. "
                              "HiPlot requires Python 3.6 or newer.")
            raise

        visualize_data_premature = self.data
        visualize_data = []
        for config, perf in visualize_data_premature.items():
            config_perf = config.get_dictionary()
            config_perf['perf'] = perf
            visualize_data.append(config_perf)
        hip.Experiment.from_iterable(visualize_data).display()
        return

    def save_json(self, fn: str = "history_container.json"):
        """
        saves runhistory on disk

        Parameters
        ----------
        fn : str
            file name
        """
        data = [(k.get_dictionary(), float(v)) for k, v in self.data.items()]

        with open(fn, "w") as fp:
            json.dump({"data": data}, fp, indent=2)

    def load_history_from_json(self, cs: ConfigurationSpace, fn: str = "history_container.json"):
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
            config = get_config_from_dict(k, cs)
            perf = float(v)
            _history_data[config] = perf
        return _history_data


class MOHistoryContainer(HistoryContainer):
    """
    Multi-Objective History Container
    """
    def __init__(self, task_id, ref_point=None):
        self.task_id = task_id
        self.data = collections.OrderedDict()
        self.config_counter = 0
        self.pareto = collections.OrderedDict()
        self.num_objs = None
        self.mo_incumbent_value = None
        self.mo_incumbents = None
        self.ref_point = ref_point
        self.hv_data = list()
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
            self.logger.info('Update pareto: config=%s, objs=%s.' % (str(config), str(perf)))

        for conf in remove_config:
            self.logger.info('Remove from pareto: config=%s, objs=%s.' % (str(conf), str(self.pareto[conf])))
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

        # Calculate current hypervolume if reference point is provided
        if self.ref_point is not None:
            pareto_front = self.get_pareto_front()
            if pareto_front:
                hv = Hypervolume(ref_point=self.ref_point).compute(pareto_front)
            else:
                hv = 0
            self.hv_data.append(hv)

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

    def compute_hypervolume(self, ref_point=None):
        if ref_point is None:
            ref_point = self.ref_point
        assert ref_point is not None
        pareto_front = self.get_pareto_front()
        if pareto_front:
            hv = Hypervolume(ref_point=ref_point).compute(pareto_front)
        else:
            hv = 0
        return hv


class MultiStartHistoryContainer(object):
    """
    History container for multistart algorithms.
    """
    def __init__(self, task_id, num_objs=1, ref_point=None):
        self.task_id = task_id
        self.num_objs = num_objs
        self.history_containers = []
        self.restart()

    def restart(self):
        if self.num_objs == 1:
            self.current = HistoryContainer(self.task_id)
        else:
            self.current = MOHistoryContainer(self.task_id, ref_point)
        self.history_containers.append(self.current)

    def get_configs_for_all_restarts(self):
        all_configs = []
        for history_container in self.history_containers:
            all_configs.extend(list(history_container.data.keys()))
        return all_configs

    def get_incumbents_for_all_restarts(self):
        best_incumbents = []
        best_incumbent_value = float('inf')
        if self.num_objs == 1:
            for hc in self.history_containers:
                incumbents = hc.get_incumbents()
                incumbent_value = hc.incumbent_value
                if incumbent_value > best_incumbent_value:
                    continue
                elif incumbent_value < best_incumbent_value:
                    best_incumbent_value = incumbent_value
                best_incumbents.extend(incumbents)
            return best_incumbents
        else:
            return self.get_pareto_front()

    def get_pareto_front(self):
        assert self.num_objs > 1
        Y = np.vstack([hc.get_pareto_front() for hc in self.history_containers])
        return get_pareto_front(Y).tolist()

    def add(self, config: Configuration, perf: Perf):
        self.current.add(config, perf)

    def get_perf(self, config: Configuration):
        for history_container in self.history_containers:
            if config in history_container.data:
                return self.data[config]
        raise KeyError

    def get_all_configs(self):
        return self.current.get_all_configs()

    def empty(self):
        return self.current.config_counter == 0

    def get_incumbents(self):
        if self.num_objs == 1:
            return self.current.incumbents
        else:
            return self.current.get_pareto()

    def get_mo_incumbents(self):
        assert self.num_objs > 1
        return self.current.mo_incumbents

    def get_mo_incumbent_value(self):
        assert self.num_objs > 1
        return self.current.mo_incumbent_value

    def get_pareto(self):
        assert self.num_objs > 1
        return self.current.get_pareto()

    def get_pareto_set(self):
        assert self.num_objs > 1
        return self.current.get_pareto_set()

    def compute_hypervolume(self, ref_point=None):
        assert self.num_objs > 1
        return self.current.compute_hypervolume(ref_point)

    def save_json(self, fn: str = "history_container.json"):
        """
        saves runhistory on disk

        Parameters
        ----------
        fn : str
            file name
        """
        self.current.save_json(fn)

    def load_history_from_json(self, cs: ConfigurationSpace, fn: str = "history_container.json"):
        """Load and runhistory in json representation from disk.
        Parameters
        ----------
        fn : str
            file name to load from
        cs : ConfigSpace
            instance of configuration space
        """
        self.current.load_history_from_json(cs, fn)
