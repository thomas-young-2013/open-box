from litebo.benchmark.objective_functions.synthetic import *
from litebo.optimizer.generic_smbo import SMBO


problem = Bukin()
bo = SMBO(problem.evaluate,
            problem.config_space,
            surrogate_type='gp',
            initial_runs=10,
            max_runs=60,
            task_id=1)
bo.run()

c_problem = Ackley(constrained=True)
cbo = SMBO(c_problem.evaluate,
            c_problem.config_space,
            num_constraints=2,
            surrogate_type='gp',
            initial_runs=10,
            max_runs=110,
            task_id=2)
cbo.run()

cbor = SMBO(c_problem.evaluate,
            c_problem.config_space,
            num_constraints=2,
            sample_strategy='random',
            initial_runs=10,
            max_runs=110,
            task_id=3,
            random_state=trial_id)
cbor.run()

mix_problem = Rao()
mbo = SMBO(mix_problem.evaluate,
            mix_problem.config_space,
            num_constraints=2,
            surrogate_type='gp',
            initial_runs=10,
            max_runs=60,
            task_id=4,
            random_state=trial_id)
mbo.run()

mbor = SMBO(mix_problem.evaluate,
            mix_problem.config_space,
            num_constraints=2,
            sample_strategy='random',
            initial_runs=10,
            max_runs=60,
            task_id=5,
            random_state=trial_id)
mbor.run()
