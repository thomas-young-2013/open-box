import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from openbox.utils.tuning import get_config_space, get_objective_function
from openbox.optimizer.generic_smbo import SMBO


if __name__ == "__main__":
    # prepare your data
    X, y = load_digits(return_X_y=True)
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

    # get config_space and objective_function
    config_space = get_config_space('xgboost')
    objective_function = get_objective_function('xgboost', x_train, x_val, y_train, y_val)

    # run
    bo = SMBO(objective_function,
              config_space,
              max_runs=100,
              time_limit_per_trial=180,
              task_id='tuning_xgboost')
    history = bo.run()

    print(history)

    history.plot_convergence()
    plt.show()

    # history.visualize_jupyter()
