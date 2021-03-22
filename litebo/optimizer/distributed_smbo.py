import numpy as np
from typing import List
from collections import OrderedDict

from litebo.core.sync_batch_advisor import SyncBatchAdvisor
from litebo.core.async_batch_advisor import AsyncBatchAdvisor
from litebo.core.distributed.master import Master


class DistributedSMBO(Master):
    def __init__(self, task_id, config_space,
                 parallel_strategy='async',
                 batch_size=4,
                 batch_strategy='median_imputation',
                 num_constraints=0,
                 num_objs=1,
                 sample_strategy: str = 'bo',
                 max_runs=200,
                 time_limit_per_trial=180,
                 surrogate_type=None,
                 acq_type=None,
                 acq_optimizer_type='local_random',
                 initial_runs=3,
                 init_strategy='random_explore_first',
                 initial_configurations=None,
                 ref_point=None,
                 history_bo_data: List[OrderedDict] = None,
                 logging_dir='logs',
                 random_state=1,):
        self.task_info = {'num_constraints': num_constraints, 'num_objs': num_objs}
        if parallel_strategy == 'sync':
            self.config_advisor = SyncBatchAdvisor(config_space, self.task_info,
                                                   batch_size=batch_size,
                                                   batch_strategy=batch_strategy,
                                                   initial_trials=initial_runs,
                                                   initial_configurations=initial_configurations,
                                                   init_strategy=init_strategy,
                                                   history_bo_data=history_bo_data,
                                                   optimization_strategy=sample_strategy,
                                                   surrogate_type=surrogate_type,
                                                   acq_type=acq_type,
                                                   acq_optimizer_type=acq_optimizer_type,
                                                   ref_point=ref_point,
                                                   task_id=task_id,
                                                   output_dir=logging_dir,
                                                   random_state=random_state)
        elif parallel_strategy == 'async':
            self.config_advisor = AsyncBatchAdvisor(config_space, self.task_info,
                                                    batch_size=batch_size,
                                                    batch_strategy=batch_strategy,
                                                    initial_trials=initial_runs,
                                                    initial_configurations=initial_configurations,
                                                    init_strategy=init_strategy,
                                                    history_bo_data=history_bo_data,
                                                    optimization_strategy=sample_strategy,
                                                    surrogate_type=surrogate_type,
                                                    acq_type=acq_type,
                                                    acq_optimizer_type=acq_optimizer_type,
                                                    ref_point=ref_point,
                                                    task_id=task_id,
                                                    output_dir=logging_dir,
                                                    random_state=random_state)
        else:
            raise ValueError('Invalid parallel strategy - %s.' % parallel_strategy)
        self.total_trials = max_runs
        super().__init__(task_id, self.config_advisor, self.total_trials)

    def get_incumbent(self):
        assert self.config_advisor is not None
        return self.config_advisor.history_container.get_incumbents()
