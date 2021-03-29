from math import ceil
from litebo.utils.config_space import ConfigurationSpace
from litebo.apps.multi_fidelity.utils import sample_configuration
from litebo.apps.multi_fidelity.async_mq_sh import async_mqSuccessiveHalving


class async_mqHyperband(async_mqSuccessiveHalving):
    """
    The implementation of Asynchronous Hyperband (extended of ASHA)
    """
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 skip_outer_loop=0,
                 random_state=1,
                 method_id='mqAsyncHyperband',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',):
        super().__init__(objective_func, config_space, R, eta=eta,
                         random_state=random_state, method_id=method_id, restart_needed=restart_needed,
                         time_limit_per_trial=time_limit_per_trial, runtime_limit=runtime_limit,
                         ip=ip, port=port, authkey=authkey)

        self.skip_outer_loop = skip_outer_loop

        # construct hyperband iteration list for initial configs
        self.hb_iter_id = 0
        self.hb_iter_list = list()
        self.B = (self.s_max + 1) * self.R
        for s in reversed(range(self.skip_outer_loop, self.s_max + 1)):
            # Initial number of configurations
            n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
            # Initial number of iterations per config
            r = self.R * self.eta ** (-s)
            self.hb_iter_list.extend([r] * n)
        self.logger.info('hyperband iteration list: %s' % self.hb_iter_list)

    def choose_next(self):
        """
        sample a random config. give iterations according to Hyperband strategy.
        """
        next_n_iteration = self.get_next_n_iteration()
        next_rung_id = self.get_rung_id(self.bracket, next_n_iteration)

        next_config = sample_configuration(self.config_space, excluded_configs=self.bracket[next_rung_id]['configs'])
        next_extra_conf = {}
        return next_config, next_n_iteration, next_extra_conf

    def get_next_n_iteration(self):
        next_n_iteration = self.hb_iter_list[self.hb_iter_id % len(self.hb_iter_list)]
        self.hb_iter_id += 1
        return next_n_iteration
