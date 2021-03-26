from litebo.optimizer.message_queue_smbo import mqSMBO
import time


class mqSMBO_modified(mqSMBO):

    def async_run_with_limit(self, runtime_limit):
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
                    time.sleep(self.sleep_time)
                    break
                # Report result.
                result_num += 1
                self.config_advisor.update_observation(observation)  # config, perf, trial_state
                self.logger.info('Master: Get %d result: %.3f' % (result_num, observation[1]))

                config, perf, trial_state = observation
                global_time = time.time() - self.global_start_time
                self.config_list.append(config)
                self.perf_list.append(perf)
                self.time_list.append(global_time)

            global_time = time.time() - self.global_start_time
            if global_time >= runtime_limit:
                return

    def sync_run_with_limit(self, runtime_limit):
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
                    time.sleep(self.sleep_time)
                    continue
                # Report result.
                result_num += 1
                self.config_advisor.update_observation(observation)  # config, perf, trial_state
                self.logger.info('Master: In the %d-th batch [%d], result is: %.3f'
                                 % (batch_id, result_num, observation[1]))

                config, perf, trial_state = observation
                global_time = time.time() - self.global_start_time
                self.config_list.append(config)
                self.perf_list.append(perf)
                self.time_list.append(global_time)

                if result_num == result_needed:
                    break
            batch_id += 1

            global_time = time.time() - self.global_start_time
            if global_time >= runtime_limit:
                return

    def run_with_limit(self, runtime_limit):
        self.max_iterations = max(self.max_iterations, 10000)
        self.sleep_time = 0.1
        self.global_start_time = time.time()
        self.config_list = []
        self.perf_list = []
        self.time_list = []
        if self.parallel_strategy == 'async':
            self.async_run_with_limit(runtime_limit)
        else:
            self.sync_run_with_limit(runtime_limit)
