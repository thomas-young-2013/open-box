from openbox.core.base import Observation
from openbox.core.generic_advisor import Advisor


class RandomAdvisor(Advisor):
    """
    Random Advisor Class, which adopts the random policy to sample a configuration.
    """

    def __init__(self, config_space,
                 task_info,
                 initial_trials=10,
                 initial_configurations=None,
                 init_strategy='random_explore_first',
                 history_bo_data=None,
                 surrogate_type=None,
                 acq_type=None,
                 acq_optimizer_type='local_random',
                 ref_point=None,
                 output_dir='logs',
                 task_id=None,
                 random_state=None):

        super().__init__(config_space, task_info, initial_trials, initial_configurations,
                         init_strategy, history_bo_data, 'random', surrogate_type,
                         acq_type, acq_optimizer_type, ref_point, output_dir, task_id, random_state)

    def get_suggestion(self, history_container=None):
        """
        Generate a configuration (suggestion) for this query.
        Returns
        -------
        A configuration.
        """
        if history_container is None:
            history_container = self.history_container
        return self.sample_random_configs(1, history_container)[0]

    def update_observation(self, observation: Observation):
        return self.history_container.update_observation(observation)
