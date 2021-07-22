import sys
import time
import traceback
import random
import copy
from openbox.optimizer.nsga_base import NSGABase
from openbox.utils.constants import MAXINT
from openbox.core.base import Observation
from openbox.utils.platypus_utils import get_variator, set_problem_types, objective_wrapper
from platypus import Problem, NSGAII
from platypus import nondominated as _nondominated

"""
    The objective function returns a dictionary that has --- config, constraints, objs ---.
"""


class NSGAOptimizer(NSGABase):
    def __init__(self, objective_function: callable,
                 config_space,
                 num_constraints=0,
                 num_objs=1,
                 max_runs=2500,
                 algorithm='nsgaii',
                 logging_dir='logs',
                 task_id=None,
                 random_state=1,
                 **kwargs):

        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

        self.num_inputs = len(config_space.get_hyperparameters())
        self.num_constraints = num_constraints
        self.num_objs = num_objs
        self.algo = algorithm
        self.FAILED_PERF = [MAXINT] * num_objs
        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, max_runs=max_runs)
        random.seed(self.rng.randint(MAXINT))

        # prepare objective function for platypus algorithm
        self.nsga_objective = objective_wrapper(objective_function, config_space, num_constraints)

        # set problem
        self.problem = Problem(self.num_inputs, num_objs, num_constraints)
        set_problem_types(config_space, self.problem)
        if num_constraints > 0:
            self.problem.constraints[:] = "<=0"
        self.problem.function = self.nsga_objective

        # set algorithm
        if self.algo == 'nsgaii':
            population_size = kwargs.get('population_size', 100)
            if self.max_iterations <= population_size:
                self.logger.warning('max_runs <= population_size! Please check.')
                population_size = min(max_runs, population_size)
            variator = get_variator(config_space)
            self.algorithm = NSGAII(self.problem, population_size=population_size, variator=variator)
        else:
            raise ValueError('Unsupported algorithm: %s' % self.algo)

    def run(self):
        self.logger.info('Start optimization. max_iterations: %d' % (self.max_iterations,))
        start_time = time.time()
        self.algorithm.run(self.max_iterations)
        end_time = time.time()
        self.logger.info('Optimization is complete. Time: %.2fs.' % (end_time - start_time))
        return self

    def get_incumbent(self):
        raise

    def get_pareto_set(self):
        raise

    def get_pareto_front(self):
        raise

    def get_result(self):
        return self.get_solutions(feasible=True, nondominated=True, decode=True)

    def get_solutions(self, feasible=False, nondominated=False, decode=True):
        solutions = copy.deepcopy(self.algorithm.result)
        if feasible:
            solutions = [s for s in solutions if s.feasible]
        if nondominated:
            solutions = _nondominated(solutions)
        if decode:
            for s in solutions:
                s.variables[:] = [self.problem.types[i].decode(s.variables[i]) for i in range(self.problem.nvars)]
        return solutions
