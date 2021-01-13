# This file is adapted from github repo: https://github.com/automl/HpBandSter.
import os
import time
import logging
import threading
import configparser

from litebo.core.distributed.dispatcher import Dispatcher
from litebo.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT


class Master(object):
	def __init__(self, run_id, config_advisor, total_trials,
				config_directory=None, ping_interval=60, host=None, job_queue_sizes=(-1,0),
				dynamic_queue_size=True, logger=None, result_logger=None):

		"""The Master class is responsible for the book keeping and to decide what to run next. Optimizers are
                instantiations of Master, that handle the important steps of deciding what configurations to run on what
                budget when.
		Parameters
		----------
		run_id : string
			A unique identifier of that Hyperband run. Use, for example, the cluster's JobID when running multiple
			concurrent runs to separate them
		config_advisor: hpbandster.config_advisors object
			An object that can generate new configurations and registers results of executed runs
		working_directory: string
			The top level working directory accessible to all compute nodes(shared filesystem).
		ping_interval: int
			number of seconds between pings to discover new nodes. Default is 60 seconds.
		host: str
			ip (or name that resolves to that) of the network interface to use
		shutdown_workers: bool
			flag to control whether the workers are shutdown after the distributed is done
		job_queue_size: tuple of ints
			min and max size of the job queue. During the run, when the number of jobs in the queue
			reaches the min value, it will be filled up to the max size. Default: (0,1)
		dynamic_queue_size: bool
			Whether or not to change the queue size based on the number of workers available.
			If true (default), the job_queue_sizes are relative to the current number of workers.
		logger: logging.logger like object
			the logger to output some (more or less meaningful) information
		result_logger: hpbandster.api.results.util.json_result_logger object
			a result logger that writes live results to disk
		previous_result: hpbandster.core.result.Result object
			previous run to warmstart the run
		"""

		self.config_directory = config_directory

		if logger is None:
			self.logger = logging.getLogger('Lite-BO[MASTER]')
		else:
			self.logger = logger
		self.logger.setLevel(logging.DEBUG)

		self.result_logger = result_logger

		self.config_advisor = config_advisor
		self.total_iterations = total_trials
		self.time_ref = None
		self.iteration_id = 0
		self.config_history = dict()

		self.iterations = list()
		self.jobs = list()

		self.num_running_jobs = 0
		self.job_queue_sizes = job_queue_sizes
		self.user_job_queue_sizes = job_queue_sizes
		self.dynamic_queue_size = dynamic_queue_size

		if job_queue_sizes[0] >= job_queue_sizes[1]:
			raise ValueError("The queue size range needs to be (min, max) with min<max!")

		# condition to synchronize the job_callback and the queue
		self.thread_cond = threading.Condition()

		self.config = {'time_ref': self.time_ref}

		if self.config_directory is None:
			self.config_directory = 'conf'
		config_path = os.path.join(self.config_directory, 'distrib.config')
		config = configparser.ConfigParser()
		config.read(config_path)
		name_server = dict(config.items('name_server'))
		self.nameserver = name_server['nameserver']
		self.nameserver_port = int(name_server['nameserver_port'])

		self.dispatcher = Dispatcher(self.job_callback, queue_callback=self.adjust_queue_size,
									 run_id=run_id, ping_interval=ping_interval,
									 nameserver=self.nameserver, nameserver_port=self.nameserver_port, host=host)

		self.dispatcher_thread = threading.Thread(target=self.dispatcher.run)
		self.dispatcher_thread.start()

	def shutdown(self, shutdown_workers=False):
		self.logger.debug('Lite-BO[MASTER]: shutdown initiated, shutdown_workers = %s' % (str(shutdown_workers)))
		self.dispatcher.shutdown(shutdown_workers)
		self.dispatcher_thread.join()

	def wait_for_workers(self, min_n_workers=1):
		"""
		helper function to hold execution until some workers are active

		Parameters
		----------
		min_n_workers: int
			minimum number of workers present before the run starts		
		"""
		self.logger.debug('wait_for_workers trying to get the condition')
		with self.thread_cond:
			while self.dispatcher.number_of_workers() < min_n_workers:
				self.logger.info('Lite-BO[MASTER]: only %i worker(s) available, waiting for at least %i.' % (self.dispatcher.number_of_workers(), min_n_workers))
				self.thread_cond.wait(1)
				self.dispatcher.trigger_discover_worker()

		self.logger.debug('Enough workers to start this run!')

	def get_next_iteration(self, iteration, iteration_kwargs):
		"""
		instantiates the next iteration

		Overwrite this to change the iterations for different optimizers

		Parameters
		----------
			iteration: int
				the index of the iteration to be instantiated
			iteration_kwargs: dict
				additional kwargs for the iteration class

		Returns
		-------
			HB_iteration: a valid HB iteration object
		"""
		
		raise NotImplementedError('implement get_next_iteration for %s' % type(self).__name__)

	def run(self, min_n_workers=1):
		"""
			Parallel implementation in a distributed environment.
			1. sync batch parallel.
			2. async parallel.
		Parameters
		----------
		n_iterations: int
			number of iterations to be performed in this run
		min_n_workers: int
			minimum number of workers before starting the run
		"""

		self.wait_for_workers(min_n_workers)

		if self.time_ref is None:
			self.time_ref = time.time()
			self.config['time_ref'] = self.time_ref
			self.logger.info('Lite-BO[MASTER]: starting run at %s' % str(self.time_ref))

		self.thread_cond.acquire()

		while True:
			self._queue_wait()

			_config = self.config_advisor.get_suggestion()
			_config_id = self.iteration_id
			self.config_history[_config_id] = _config
			self.logger.info('Lite-BO[MASTER]: schedule new run for iteration %i' % self.iteration_id)
			self._submit_job(_config_id, _config)

			self.iteration_id += 1

			if self.iteration_id >= self.total_iterations:
				break

		self.thread_cond.release()

		# Implementation needed! return the result.

	def adjust_queue_size(self, number_of_workers=None):
		self.logger.debug('Lite-BO[MASTER]: number of workers changed to %s' % str(number_of_workers))
		with self.thread_cond:
			self.logger.debug('adjust_queue_size: lock accquired')
			if self.dynamic_queue_size:
				nw = self.dispatcher.number_of_workers() if number_of_workers is None else number_of_workers
				self.job_queue_sizes = (self.user_job_queue_sizes[0] + nw, self.user_job_queue_sizes[1] + nw)
				self.logger.info('Lite-BO[MASTER]: adjusted queue size to %s'%str(self.job_queue_sizes))
			self.thread_cond.notify_all()

	def job_callback(self, job):
		"""
		method to be called when a job has finished

		this will do some book keeping and call the user defined
		new_result_callback if one was specified
		"""
		self.logger.info('job_callback for %s started' % str(job.id))
		with self.thread_cond:
			self.logger.debug('job_callback for %s got condition' % str(job.id))
			self.num_running_jobs -= 1

			if self.result_logger is not None:
				self.result_logger(job)

			result = job.result
			_config_id = job.id
			_config = self.config_history[_config_id]
			config_dict = result['config']
			assert _config.get_dictionary() == config_dict
			_perf = result['objective_value']
			_trial_state = SUCCESS if job.exception is None else FAILED
			_observation = [_config, _perf, _trial_state]

			# Report the result, and remove the config from the running queue.
			self.config_advisor.update_observation(_observation)

			if self.num_running_jobs <= self.job_queue_sizes[0]:
				self.logger.debug("Lite-BO[MASTER]: Trying to run another job!")
				self.thread_cond.notify()

		self.logger.debug('job_callback for %s finished' % str(job.id))

	def _queue_wait(self):
		"""
		helper function to wait for the queue to not overflow/underload it
		"""
		if self.num_running_jobs >= self.job_queue_sizes[1]:
			while self.num_running_jobs > self.job_queue_sizes[0]:
				self.logger.debug('Lite-BO[MASTER]: running jobs: %i, queue sizes: %s -> wait' %
								  (self.num_running_jobs, str(self.job_queue_sizes)))
				self.thread_cond.wait()

	def _submit_job(self, config_id, config, budget=1.):
		"""
		hidden function to submit a new job to the dispatcher

		This function handles the actual submission in a
		(hopefully) thread save way
		"""
		self.logger.debug('Lite-BO[MASTER]: trying submitting job %s to dispatcher' % str(config_id))
		self.config_advisor.running_configs.append(config)
		with self.thread_cond:
			self.logger.info('Lite-BO[MASTER]: submitting job %s to dispatcher' % str(config_id))
			self.dispatcher.submit_job(config_id, config=config.get_dictionary(), budget=budget)
			self.num_running_jobs += 1

		# shouldn't the next line be executed while holding the condition?
		self.logger.debug("Lite-BO[MASTER]: job %s submitted to dispatcher"%str(config_id))

	def __del__(self):
		pass
