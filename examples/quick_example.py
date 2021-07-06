import numpy as np
import matplotlib.pyplot as plt
from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from openbox.optimizer.generic_smbo import SMBO


# Define Configuration Space
config_space = ConfigurationSpace()
x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
config_space.add_hyperparameters([x1, x2])


# Define Objective Function
def branin(config):
    import numpy as np
    config_dict = config.get_dictionary()
    x1 = config_dict['x1']
    x2 = config_dict['x2']

    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

    ret = dict(
        objs=(y, )
    )
    return ret


if __name__ == "__main__":
    # Run Optimization
    bo = SMBO(branin,
              config_space,
              num_objs=1,
              num_constraints=0,
              max_runs=50,
              surrogate_type='gp',
              time_limit_per_trial=180,
              task_id='quick_start')
    history = bo.run()

    print(history)

    # print(history.get_importance())

    history.plot_convergence(true_minimum=0.397887)
    plt.show()

    # history.visualize_jupyter()
