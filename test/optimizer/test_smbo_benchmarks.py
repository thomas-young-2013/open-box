import pandas as pd

from litebo.benchmark.objective_functions.synthetic import *
from litebo.optimizer.generic_smbo import SMBO

results = []
columns = ['trial_id', 'problem', 'method', 'iteration', 'value']


def log_history(trial_id, name, method, problem: BaseTestProblem, bo: SMBO):
    ys = list(bo.get_history().data.values())
    for i, y in enumerate(ys):
        results.append([trial_id, name, method, i, y - problem.optimal_value])


num_trials = 10
for trial_id in range(num_trials):
    problem = Bukin()
    bo = SMBO(problem.evaluate,
              problem.config_space,
              surrogate_type='gp',
              initial_runs=10,
              max_runs=60,
              random_state=trial_id)
    bo.run()
    log_history(trial_id, 'bukin', 'ei', problem, bo)

    c_problem = Ackley(constrained=True)
    cbo = SMBO(c_problem.evaluate,
               c_problem.config_space,
               num_constraints=2,
               surrogate_type='gp',
               initial_runs=10,
               max_runs=110,
               random_state=trial_id)
    cbo.run()
    log_history(trial_id, 'ackley', 'eic', c_problem, cbo)

    cbor = SMBO(c_problem.evaluate,
                c_problem.config_space,
                num_constraints=2,
                sample_strategy='random',
                initial_runs=10,
                max_runs=110,
                random_state=trial_id)
    cbor.run()
    log_history(trial_id, 'ackley', 'rs', c_problem, cbor)

    mix_problem = Rao()
    mbo = SMBO(mix_problem.evaluate,
               mix_problem.config_space,
               num_constraints=2,
               surrogate_type='gp',
               initial_runs=10,
               max_runs=60,
               random_state=trial_id)
    mbo.run()
    log_history(trial_id, 'rao', 'eic', mix_problem, mbo)

    mbor = SMBO(mix_problem.evaluate,
                mix_problem.config_space,
                num_constraints=2,
                sample_strategy='random',
                initial_runs=10,
                max_runs=60,
                random_state=trial_id)
    mbor.run()
    log_history(trial_id, 'rao', 'rs', mix_problem, mbor)

df = pd.DataFrame(results, columns=columns)
df.to_csv('results.csv', index=False)
