import os
import abc
import numpy as np

from openbox.utils.util_funcs import get_types
from openbox.utils.logging_utils import get_logger
from openbox.utils.history_container import HistoryContainer, MOHistoryContainer, \
    MultiStartHistoryContainer
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.utils.samplers import SobolSampler, LatinHypercubeSampler
from openbox.utils.multi_objective import get_chebyshev_scalarization, NondominatedPartitioning
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.core.base import build_acq_func, build_optimizer, build_surrogate
from openbox.core.base import Observation


class Advisor(object, metaclass=abc.ABCMeta):
    """
    Basic Advisor Class, which adopts a policy to sample a configuration.
    """

    def __init__(self, config_space,
                 task_info,
                 initial_trials=10,
                 initial_configurations=None,
                 init_strategy='random_explore_first',
                 history_bo_data=None,
                 optimization_strategy='bo',
                 surrogate_type=None,
                 acq_type=None,
                 acq_optimizer_type='local_random',
                 ref_point=None,
                 output_dir='logs',
                 task_id=None,
                 random_state=None):

        # Create output (logging) directory.
        # Init logging module.
        # Random seed generator.
        self.task_info = task_info
        self.num_objs = task_info['num_objs']
        self.num_constraints = task_info['num_constraints']
        self.init_strategy = init_strategy
        self.output_dir = output_dir
        self.rng = np.random.RandomState(random_state)
        self.logger = get_logger(self.__class__.__name__)

        history_folder = os.path.join(self.output_dir, 'bo_history')
        if not os.path.exists(history_folder):
            os.makedirs(history_folder)
        self.history_file = os.path.join(history_folder, 'bo_history_%s.json' % task_id)

        # Basic components in Advisor.
        self.optimization_strategy = optimization_strategy

        # Init the basic ingredients in Bayesian optimization.
        self.history_bo_data = history_bo_data
        self.surrogate_type = surrogate_type
        self.constraint_surrogate_type = None
        self.acq_type = acq_type
        self.acq_optimizer_type = acq_optimizer_type
        self.init_num = initial_trials
        self.config_space = config_space
        self.config_space_seed = self.rng.randint(MAXINT)
        self.config_space.seed(self.config_space_seed)
        self.ref_point = ref_point

        # init history container
        if self.num_objs == 1:
            self.history_container = HistoryContainer(task_id, self.num_constraints, config_space=self.config_space)
        else:  # multi-objectives
            self.history_container = MOHistoryContainer(task_id, self.num_objs, self.num_constraints, ref_point)

        # initial design
        if initial_configurations is not None and len(initial_configurations) > 0:
            self.initial_configurations = initial_configurations
            self.init_num = len(initial_configurations)
        else:
            self.initial_configurations = self.create_initial_design(self.init_strategy)
            self.init_num = len(self.initial_configurations)

        self.surrogate_model = None
        self.constraint_models = None
        self.acquisition_function = None
        self.optimizer = None
        self.check_setup()
        self.setup_bo_basics()

    def check_setup(self):
        """
        Check optimization_strategy, num_objs, num_constraints, acq_type, surrogate_type.
        Returns
        -------
        None
        """
        assert self.optimization_strategy in ['bo', 'random']
        assert isinstance(self.num_objs, int) and self.num_objs >= 1
        assert isinstance(self.num_constraints, int) and self.num_constraints >= 0

        # single objective
        if self.num_objs == 1:
            if self.num_constraints == 0:
                if self.acq_type is None:
                    self.acq_type = 'ei'
                assert self.acq_type in ['ei', 'eips', 'logei', 'pi', 'lcb', 'lpei', ]
                if self.surrogate_type is None:
                    self.surrogate_type = 'gp'
            else:  # with constraints
                if self.acq_type is None:
                    self.acq_type = 'eic'
                assert self.acq_type in ['eic', ]
                if self.surrogate_type is None:
                    self.surrogate_type = 'gp'
                if self.constraint_surrogate_type is None:
                    self.constraint_surrogate_type = 'gp'

        # multi-objective
        else:
            if self.num_constraints == 0:
                if self.acq_type is None:
                    self.acq_type = 'ehvi'
                assert self.acq_type in ['ehvi', 'mesmo', 'usemo', 'parego']
                if self.surrogate_type is None:
                    if self.acq_type == 'mesmo':
                        self.surrogate_type = 'gp_rbf'
                    else:
                        self.surrogate_type = 'gp'
                if self.acq_type == 'mesmo' and self.surrogate_type != 'gp_rbf':
                    self.surrogate_type = 'gp_rbf'
                    self.logger.warning('Surrogate model has changed to Gaussian Process with RBF kernel '
                                        'since MESMO is used. Surrogate_type should be set to \'gp_rbf\'.')
            else:  # with constraints
                if self.acq_type is None:
                    self.acq_type = 'ehvic'
                assert self.acq_type in ['ehvic', 'mesmoc', 'mesmoc2']
                if self.surrogate_type is None:
                    if self.acq_type == 'mesmoc':
                        self.surrogate_type = 'gp_rbf'
                    else:
                        self.surrogate_type = 'gp'
                if self.constraint_surrogate_type is None:
                    if self.acq_type == 'mesmoc':
                        self.constraint_surrogate_type = 'gp_rbf'
                    else:
                        self.constraint_surrogate_type = 'gp'
                if self.acq_type == 'mesmoc' and self.surrogate_type != 'gp_rbf':
                    self.surrogate_type = 'gp_rbf'
                    self.logger.warning('Surrogate model has changed to Gaussian Process with RBF kernel '
                                        'since MESMOC is used. Surrogate_type should be set to \'gp_rbf\'.')
                if self.acq_type == 'mesmoc' and self.constraint_surrogate_type != 'gp_rbf':
                    self.surrogate_type = 'gp_rbf'
                    self.logger.warning('Constraint surrogate model has changed to Gaussian Process with RBF kernel '
                                        'since MESMOC is used. Surrogate_type should be set to \'gp_rbf\'.')

            # Check reference point is provided for EHVI methods
            if 'ehvi' in self.acq_type and self.ref_point is None:
                raise ValueError('Must provide reference point to use EHVI method!')

    def setup_bo_basics(self):
        """
        Prepare the basic BO components.
        Returns
        -------
        An optimizer object.
        """
        if self.num_objs == 1 or self.acq_type == 'parego':
            self.surrogate_model = build_surrogate(func_str=self.surrogate_type,
                                                   config_space=self.config_space,
                                                   rng=self.rng,
                                                   history_hpo_data=self.history_bo_data)
        else:  # multi-objectives
            self.surrogate_model = [build_surrogate(func_str=self.surrogate_type,
                                                    config_space=self.config_space,
                                                    rng=self.rng,
                                                    history_hpo_data=self.history_bo_data)
                                    for _ in range(self.num_objs)]

        if self.num_constraints > 0:
            self.constraint_models = [build_surrogate(func_str=self.constraint_surrogate_type,
                                                      config_space=self.config_space,
                                                      rng=self.rng) for _ in range(self.num_constraints)]

        if self.acq_type in ['mesmo', 'mesmoc', 'mesmoc2', 'usemo']:
            types, bounds = get_types(self.config_space)
            self.acquisition_function = build_acq_func(func_str=self.acq_type,
                                                       model=self.surrogate_model,
                                                       constraint_models=self.constraint_models,
                                                       types=types,
                                                       bounds=bounds)
        else:
            self.acquisition_function = build_acq_func(func_str=self.acq_type,
                                                       model=self.surrogate_model,
                                                       constraint_models=self.constraint_models,
                                                       ref_point=self.ref_point)
        if self.acq_type == 'usemo':
            self.acq_optimizer_type = 'usemo_optimizer'
        elif self.acq_type.startswith('mesmo'):
            self.acq_optimizer_type = 'mesmo_optimizer'
        self.optimizer = build_optimizer(func_str=self.acq_optimizer_type,
                                         acq_func=self.acquisition_function,
                                         config_space=self.config_space,
                                         rng=self.rng)

    def create_initial_design(self, init_strategy='default'):
        """
        Create several configurations as initial design.
        Parameters
        ----------
        init_strategy: str

        Returns
        -------
        Initial configurations.
        """
        default_config = self.config_space.get_default_configuration()
        num_random_config = self.init_num - 1
        if init_strategy == 'random':
            initial_configs = self.sample_random_configs(self.init_num)
            return initial_configs
        elif init_strategy == 'default':
            initial_configs = [default_config] + self.sample_random_configs(num_random_config)
            return initial_configs
        elif init_strategy == 'random_explore_first':
            candidate_configs = self.sample_random_configs(100)
            return self.max_min_distance(default_config, candidate_configs, num_random_config)
        elif init_strategy == 'sobol':
            sobol = SobolSampler(self.config_space, num_random_config, random_state=self.rng)
            initial_configs = [default_config] + sobol.generate(return_config=True)
            return initial_configs
        elif init_strategy == 'latin_hypercube':
            lhs = LatinHypercubeSampler(self.config_space, num_random_config, criterion='maximin')
            initial_configs = [default_config] + lhs.generate(return_config=True)
            return initial_configs
        else:
            raise ValueError('Unknown initial design strategy: %s.' % init_strategy)

    def max_min_distance(self, default_config, src_configs, num):
        min_dis = list()
        initial_configs = list()
        initial_configs.append(default_config)

        for config in src_configs:
            dis = np.linalg.norm(config.get_array() - default_config.get_array())
            min_dis.append(dis)
        min_dis = np.array(min_dis)

        for i in range(num):
            furthest_config = src_configs[np.argmax(min_dis)]
            initial_configs.append(furthest_config)
            min_dis[np.argmax(min_dis)] = -1

            for j in range(len(src_configs)):
                if src_configs[j] in initial_configs:
                    continue
                updated_dis = np.linalg.norm(src_configs[j].get_array() - furthest_config.get_array())
                min_dis[j] = min(updated_dis, min_dis[j])

        return initial_configs

    def get_suggestion(self, history_container=None):
        """
        Generate a configuration (suggestion) for this query.
        Returns
        -------
        A configuration.
        """
        if history_container is None:
            history_container = self.history_container

        num_config_evaluated = len(history_container.configurations)
        num_config_successful = len(history_container.successful_perfs)

        if num_config_evaluated < self.init_num:
            return self.initial_configurations[num_config_evaluated]

        if self.optimization_strategy == 'random':
            return self.sample_random_configs(1, history_container)[0]

        X = convert_configurations_to_array(history_container.configurations)
        Y = history_container.get_transformed_perfs()
        cY = history_container.get_transformed_constraint_perfs()

        if self.optimization_strategy == 'bo':
            if num_config_successful < max(self.init_num, 1):
                self.logger.warning('No enough successful initial trials! Sample random configuration.')
                return self.sample_random_configs(1, history_container)[0]

            # train surrogate model
            if self.num_objs == 1:
                self.surrogate_model.train(X, Y)
            elif self.acq_type == 'parego':
                weights = self.rng.random_sample(self.num_objs)
                weights = weights / np.sum(weights)
                scalarized_obj = get_chebyshev_scalarization(weights, Y)
                self.surrogate_model.train(X, scalarized_obj(Y))
            else:  # multi-objectives
                for i in range(self.num_objs):
                    self.surrogate_model[i].train(X, Y[:, i])

            # train constraint model
            for i in range(self.num_constraints):
                self.constraint_models[i].train(X, cY[:, i])

            # update acquisition function
            if self.num_objs == 1:
                incumbent_value = history_container.get_incumbents()[0][1]
                self.acquisition_function.update(model=self.surrogate_model,
                                                 constraint_models=self.constraint_models,
                                                 eta=incumbent_value,
                                                 num_data=num_config_evaluated)
            else:  # multi-objectives
                mo_incumbent_value = history_container.get_mo_incumbent_value()
                if self.acq_type == 'parego':
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     eta=scalarized_obj(np.atleast_2d(mo_incumbent_value)),
                                                     num_data=num_config_evaluated)
                elif self.acq_type.startswith('ehvi'):
                    partitioning = NondominatedPartitioning(self.num_objs, Y)
                    cell_bounds = partitioning.get_hypercell_bounds(ref_point=self.ref_point)
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     cell_lower_bounds=cell_bounds[0],
                                                     cell_upper_bounds=cell_bounds[1])
                else:
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     constraint_perfs=cY,  # for MESMOC
                                                     eta=mo_incumbent_value,
                                                     num_data=num_config_evaluated,
                                                     X=X, Y=Y)

            # optimize acquisition function
            challengers = self.optimizer.maximize(runhistory=history_container,
                                                  num_points=5000)
            is_repeated_config = True
            repeated_time = 0
            cur_config = None
            while is_repeated_config:
                cur_config = challengers.challengers[repeated_time]  # todo: test small space
                if cur_config in history_container.configurations:
                    repeated_time += 1
                else:
                    is_repeated_config = False
            return cur_config
        else:
            raise ValueError('Unknown optimization strategy: %s.' % self.optimization_strategy)

    def update_observation(self, observation: Observation):
        """
        Update the current observations.
        Parameters
        ----------
        observation

        Returns
        -------

        """
        return self.history_container.update_observation(observation)

    def sample_random_configs(self, num_configs=1, history_container=None):
        """
        Sample a batch of random configurations.
        Parameters
        ----------
        num_configs

        history_container

        Returns
        -------

        """
        if history_container is None:
            history_container = self.history_container

        configs = list()
        sample_cnt = 0
        max_sample_cnt = 1000
        while len(configs) < num_configs:
            config = self.config_space.sample_configuration()
            sample_cnt += 1
            if config not in (history_container.configurations + configs):
                configs.append(config)
                sample_cnt = 0
                continue
            if sample_cnt >= max_sample_cnt:
                self.logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                configs.append(config)
                sample_cnt = 0
        return configs

    def save_history(self):
        """
        Save the history into a json file.
        Returns
        -------

        """
        self.history_container.save_json(self.history_file)

    def load_history_from_json(self):
        """
        Load history from a json file.
        Returns
        -------

        """
        return self.history_container.load_history_from_json(self.config_space, self.history_file)

    def get_suggestions(self):
        raise NotImplementedError
