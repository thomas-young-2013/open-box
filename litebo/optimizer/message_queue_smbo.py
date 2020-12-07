import time
from typing import List
from collections import OrderedDict

from litebo.core.sync_batch_advisor import SyncBatchAdvisor
from litebo.core.async_batch_advisor import AsyncBatchAdvisor
from litebo.optimizer.base import BOBase
from litebo.core.message_queue.master_messager import MasterMessager


class mqSMBO(BOBase):
    def __init__(self,
                 objective_function,
                 config_space,
                 batch_size=4,
                 sample_strategy='bo',
                 parallel_strategy='async',
                 time_limit_per_trial=180,
                 max_runs=200,
                 logging_dir='logs',
                 initial_configurations=None,
                 init_strategy='random_explore_first',
                 history_bo_data: List[OrderedDict] = None,
                 initial_runs=10,
                 task_id=None,
                 random_state=1,
                 ip="",
                 port=13579,):

        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         sample_strategy=sample_strategy, time_limit_per_trial=time_limit_per_trial,
                         history_bo_data=history_bo_data)
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

        self.parallel_strategy = parallel_strategy
        self.batch_size = batch_size
        max_queue_len = max(100, 3 * batch_size)
        self.master_messager = MasterMessager(ip, port, max_queue_len, max_queue_len)

    def async_run(self):
        config_num = 0
        result_num = 0
        while result_num < self.max_iterations:
            # Add jobs to masterQueue.
            while len(self.config_advisor.running_configs) < self.batch_size and config_num < self.max_iterations:
                config_num += 1
                _config = self.config_advisor.get_suggestion()
                _msg = [_config, self.time_limit_per_trial]
                self.logger.info("Master: Add config %d." % config_num)
                self.master_messager.send_message(_msg)

            # Get results from workerQueue.
            while True:
                observation = self.master_messager.receive_message()
                if observation is None:
                    # Wait for workers.
                    # self.logger.info("Master: wait for worker results. sleep 1s.")
                    time.sleep(1)
                    break
                # Report result.
                result_num += 1
                self.config_advisor.update_observation(observation)  # config, perf, trial_state
                self.logger.info('Master: Get %d result: %.3f' % (result_num, observation[1]))

    def sync_run(self):
        batch_num = (self.max_iterations + self.batch_size - 1) // self.batch_size
        if self.batch_size > self.config_advisor.init_num:
            batch_num += 1  # fix bug
        batch_id = 0
        while batch_id < batch_num:
            configs = self.config_advisor.get_suggestions()
            # Add batch configs to masterQueue.
            for config in configs:
                msg = [config, self.time_limit_per_trial]
                self.master_messager.send_message(msg)
            self.logger.info('Master: %d-th batch. %d configs sent.' % (batch_id, len(configs)))
            # Get batch results from workerQueue.
            result_num = 0
            result_needed = len(configs)
            while True:
                observation = self.master_messager.receive_message()
                if observation is None:
                    # Wait for workers.
                    # self.logger.info("Master: wait for worker results. sleep 1s.")
                    time.sleep(1)
                    continue
                # Report result.
                result_num += 1
                self.config_advisor.update_observation(observation)  # config, perf, trial_state
                self.logger.info('Master: In the %d-th batch [%d], result is: %.3f'
                                 % (batch_id, result_num, observation[1]))
                if result_num == result_needed:
                    break
            batch_id += 1

    def run(self):
        if self.parallel_strategy == 'async':
            self.async_run()
        else:
            self.sync_run()
