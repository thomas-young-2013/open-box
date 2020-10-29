import numpy as np
from typing import List
from collections import OrderedDict

from litebo.core.sync_batch_advisor import SyncBatchAdvisor
from litebo.core.async_batch_advisor import AsyncBatchAdvisor
from litebo.core.distributed.master import Master


class DistributedSMBO(Master):
    def __init__(self, task_id, config_space,
                 sample_strategy='bo',
                 batch_size=4,
                 parallel_strategy='async',
                 time_limit_per_trial=180,
                 max_runs=200,
                 logging_dir='logs',
                 initial_configurations=None,
                 init_strategy='random_explore_first',
                 history_bo_data: List[OrderedDict] = None,
                 initial_runs=10,
                 random_state=1):
        self.rng = np.random.RandomState(random_state)
        if parallel_strategy == 'sync':
            self.config_advisor = SyncBatchAdvisor(config_space,
                                                   initial_trials=initial_runs,
                                                   initial_configurations=initial_configurations,
                                                   init_strategy=init_strategy,
                                                   optimization_strategy=sample_strategy,
                                                   batch_size=batch_size,
                                                   task_id=task_id,
                                                   output_dir=logging_dir,
                                                   rng=self.rng)
        elif parallel_strategy == 'async':
            self.config_advisor = AsyncBatchAdvisor(config_space,
                                                    initial_trials=initial_runs,
                                                    initial_configurations=initial_configurations,
                                                    init_strategy=init_strategy,
                                                    optimization_strategy=sample_strategy,
                                                    task_id=task_id,
                                                    output_dir=logging_dir,
                                                    rng=self.rng)
        else:
            raise ValueError('Invalid parallel strategy - %s.' % parallel_strategy)
        self.total_trials = max_runs
        super().__init__(task_id, self.config_advisor, self.total_trials)

    def get_incumbent(self):
        assert self.config_advisor is not None
        return self.config_advisor.history_container.get_incumbents()
