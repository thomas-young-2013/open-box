import numpy as np
from math import log, ceil
from collections import OrderedDict

from litebo.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from litebo.core.advisor import Advisor


class HyperbandAdvisor(Advisor):
    def __init__(self, config_space,
                 R,
                 eta=3,
                 restart_needed=True,
                 skip_last=0,
                 initial_trials=10,
                 initial_configurations=None,
                 init_strategy='random_explore_first',  # todo
                 history_bo_data=None,
                 optimization_strategy='hyperband',
                 surrogate_type='prf',
                 output_dir='logs',
                 task_id=None,
                 rng=None):

        super().__init__(config_space,
                         initial_trials=initial_trials,
                         initial_configurations=initial_configurations,
                         init_strategy=init_strategy,
                         optimization_strategy=optimization_strategy,
                         history_bo_data=history_bo_data,
                         surrogate_type=surrogate_type,
                         output_dir=output_dir,
                         task_id=task_id,
                         rng=rng)
        self.R = R  # Maximum iterations per configuration
        self.eta = eta  # Define configuration downsampling rate (default = 3)
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R
        self.restart_needed = restart_needed
        self.skip_last = skip_last

        self.incumbent_configs = list()
        self.incumbent_perfs = list()
        self.n_configs = 0                  # running configs num
        self.n_observations = 0             # running observation got num
        self.running_stats = OrderedDict()  # config : perf
        self.running_ret_val = []
        self.running_early_stops = []
        self.generator = self.iterate()

    def setup_bo_basics(self, acq_type='ei', acq_optimizer_type='local_random'):
        return

    def get_suggestion(self):
        raise NotImplementedError('please call get_mf_suggestion()')

    def update_observation(self, observation):
        raise NotImplementedError('please call update_mf_observation()')

    def get_mf_suggestion(self):
        return next(self.generator)     # config, n_iter, R, extra_info

    def update_mf_observation(self, observation):
        config, perf, trial_state = observation
        # return_info, time_taken, trial_id, config = observation   # todo mf result
        assert config in self.running_stats.keys()
        if not isinstance(perf, (int, float)):
            perf = perf[-1]     # todo what does this mean?
        self.running_stats[config] = perf
        self.n_observations += 1
        if self.n_observations == int(self.n_configs):   # todo just temporary!!!
            self.running_ret_val = [dict(loss=p, ref_id=None) for p in self.running_stats.values()]
            self.running_early_stops = [False] * int(self.n_configs)
            next(self.generator)    # update observation or the last iter is not completed
        # todo no use
        if trial_state == SUCCESS and perf < MAXINT:
            if len(self.configurations) == 0:
                self.default_obj_value = perf

            self.configurations.append(config)
            self.perfs.append(perf)
            # self.history_container.add(config, perf)  # todo deal repeated. only record full?

            self.perc = np.percentile(self.perfs, self.scale_perc)
            self.min_y = np.min(self.perfs)
            self.max_y = np.max(self.perfs)
        else:
            self.failed_configurations.append(config)

    def iterate(self):
        """
        a suggestion generator
        """
        while True:
            for s in reversed(range(self.s_max + 1)):
                # Initial number of configurations
                n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
                # Initial number of iterations per config
                r = self.R * self.eta ** (-s)

                # Choose next n configurations.
                T = self.choose_next(n)
                incumbent_loss = np.inf
                extra_info = None
                last_run_num = None
                for i in range((s + 1) - int(self.skip_last)):  # Changed from s + 1
                    # Run each of the n configs for <iterations>
                    # and keep best (n_configs / eta) configurations.

                    self.n_configs = n * self.eta ** (-i)   # caution: covert to int when comparing
                    n_iterations = r * self.eta ** (i)
                    n_iter = n_iterations
                    if last_run_num is not None and not self.restart_needed:
                        n_iter -= last_run_num
                    last_run_num = n_iterations

                    self.logger.info("HB: %d configurations x %d iterations each"
                                     % (int(self.n_configs), int(n_iterations)))

                    # reset running stats
                    self.running_stats.clear()
                    self.running_ret_val.clear()
                    self.running_early_stops.clear()
                    self.n_observations = 0
                    # send suggestions
                    if extra_info is None:
                        extra_info = [None] * int(self.n_configs)
                    for idx, t in enumerate(T):
                        self.running_stats[t] = None
                        self.logger.info("config %d sent: %s, %d, %d, %s"
                                         % (idx+1, t, n_iter, self.R, extra_info[idx]))
                        yield t, n_iter, self.R, extra_info[idx]    # return one suggestion

                    # wait for observations
                    while any(v is None for v in self.running_stats.values()):
                        self.logger.warning("update_mf_observation is needed!")
                        yield None

                    # ret_val, early_stops = self.run_in_parallel(T, n_iter, extra_info)  # todo

                    val_losses = [item['loss'] for item in self.running_ret_val]
                    ref_list = [item['ref_id'] for item in self.running_ret_val]

                    # select a number of best configurations for the next loop
                    # filter out early stops, if any
                    indices = np.argsort(val_losses)
                    if len(T) == sum(self.running_early_stops):
                        break
                    if len(T) >= self.eta:
                        T = [T[i] for i in indices if not self.running_early_stops[i]]
                        extra_info = [ref_list[i] for i in indices if not self.running_early_stops[i]]
                        reduced_num = int(self.n_configs / self.eta)
                        T = T[0:reduced_num]
                        extra_info = extra_info[0:reduced_num]
                    else:
                        T = [T[indices[0]]]
                        extra_info = [ref_list[indices[0]]]
                    incumbent_loss = val_losses[indices[0]]
                    # self.add_stage_history(self.stage_id, min(self.global_incumbent, incumbent_loss)) # todo
                    # self.stage_id += 1
                    self.update_incumbent(T, val_losses, indices, n_iterations)
                    yield   # update_mf_observation called

    def choose_next(self, num_config):
        # Sample n configurations uniformly.
        return self.sample_random_configs(num_config)

    def update_incumbent(self, T, val_losses, indices, n_iterations):
        if int(n_iterations) < self.R:
            return
        incumbent_loss = val_losses[indices[0]]
        if not np.isnan(incumbent_loss):
            self.incumbent_configs.append(T[0])
            self.incumbent_perfs.append(incumbent_loss)

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_perfs) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_perfs)
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        perfs = [self.incumbent_perfs[i] for i in indices[0: num_inc]]
        return configs, perfs
