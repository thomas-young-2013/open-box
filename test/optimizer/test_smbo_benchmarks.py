from litebo.benchmark.objective_functions.synthetic import Ackley, Beale, Branin, DTLZ2
from litebo.optimizer.generic_smbo import SMBO


num_trials = 10
for trial_id in range(num_trials):
    problem = DTLZ2(dim=3, constrained=True)
    bo = SMBO(problem.evaluate,
              problem.config_space,
              num_objs=2,
              num_constraints=1,
              ref_point=problem.ref_point,
              max_runs=100,
              random_state=trial_id)
    bo.run()
