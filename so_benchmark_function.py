import numpy as np
import time

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    Constant, CategoricalHyperparameter, InCondition, EqualsCondition, UnParametrizedHyperparameter, \
    ForbiddenEqualsClause, ForbiddenInClause, ForbiddenAndConjunction


def get_problem(problem_str, **kwargs):
    if problem_str == 'branin':
        problem = branin
    elif problem_str == 'townsend':
        problem = townsend
    elif 'lgb' in problem_str:
        problem = lgb
        kwargs['dataset'] = '_'.join(problem_str.split('_')[1:])
    elif 'svc' in problem_str:
        problem = svc
        kwargs['dataset'] = '_'.join(problem_str.split('_')[1:])
    else:
        raise ValueError('Unknown problem_str %s.' % problem_str)
    return problem(**kwargs)


class BaseSingleObjectiveProblem:
    def __init__(self, **kwargs):
        self.dim = kwargs.get('dim', None)

    def evaluate_config(self, config, optimizer='smac'):
        raise NotImplementedError

    def evaluate(self, X: np.ndarray):
        raise NotImplementedError

    @staticmethod
    def get_config_dict(config, optimizer='smac'):
        if optimizer == 'smac':
            config_dict = config.get_dictionary()
        elif optimizer == 'tpe':
            config_dict = config
        else:
            raise ValueError('Unknown optimizer %s' % optimizer)
        return config_dict

    @staticmethod
    def checkX(X: np.ndarray):
        X = np.atleast_2d(X)
        assert len(X.shape) == 2 and X.shape[0] == 1
        X = X.flatten()
        return X

    @staticmethod
    def get_configspace(optimizer='smac'):
        raise NotImplementedError

    def load_data(self, **kwargs):
        from test.test_utils import load_data
        from sklearn.model_selection import train_test_split
        dataset = kwargs['dataset']
        try:
            data_dir = kwargs.get('data_dir', '../soln-ml/data/cls_datasets/')
            x, y = load_data(dataset, data_dir)
        except Exception as e:
            data_dir = '../../soln-ml/data/cls_datasets/'
            x, y = load_data(dataset, data_dir)
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(x, y, stratify=y, random_state=1,
                                                                              test_size=0.3)


class ackley(BaseSingleObjectiveProblem):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate_config(self, config, optimizer='smac'):
        config_dict = self.get_config_dict(config, optimizer)
        x1 = config_dict['x1']
        x2 = config_dict['x2']
        X = np.array([x1, x2])
        return self.evaluate(X)

    def evaluate(self, X: np.ndarray):
        X = self.checkX(X)
        res = dict()
        res['objs'] = (-(np.cos((X[0] - 0.1) * X[1]) ** 2 + X[0] * np.sin(3 * X[0] + X[1])),)
        res['constraints'] = (
            -(-np.cos(1.5 * X[0] + np.pi) * np.cos(1.5 * X[1]) + np.sin(1.5 * X[0] + np.pi) * np.sin(1.5 * X[1])),)
        return res

    @staticmethod
    def get_configspace(optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            x1 = UniformFloatHyperparameter("x1", -2.25, 2.5)
            x2 = UniformFloatHyperparameter("x2", -2.5, 1.75)
            cs.add_hyperparameters([x1, x2])
            return cs
        elif optimizer == 'gpflowopt':
            import gpflowopt
            domain = gpflowopt.domain.ContinuousParameter('x1', -2.25, 2.5) + \
                     gpflowopt.domain.ContinuousParameter('x2', -2.5, 1.75)
            return domain
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


class townsend(BaseSingleObjectiveProblem):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate_config(self, config, optimizer='smac'):
        config_dict = self.get_config_dict(config, optimizer)
        x1 = config_dict['x1']
        x2 = config_dict['x2']
        X = np.array([x1, x2])
        return self.evaluate(X)

    def evaluate(self, X: np.ndarray):
        X = self.checkX(X)
        res = dict()
        res['objs'] = (-(np.cos((X[0] - 0.1) * X[1]) ** 2 + X[0] * np.sin(3 * X[0] + X[1])),)
        res['constraints'] = (
            -(-np.cos(1.5 * X[0] + np.pi) * np.cos(1.5 * X[1]) + np.sin(1.5 * X[0] + np.pi) * np.sin(1.5 * X[1])),)
        return res

    @staticmethod
    def get_configspace(optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            x1 = UniformFloatHyperparameter("x1", -2.25, 2.5)
            x2 = UniformFloatHyperparameter("x2", -2.5, 1.75)
            cs.add_hyperparameters([x1, x2])
            return cs
        elif optimizer == 'gpflowopt':
            import gpflowopt
            domain = gpflowopt.domain.ContinuousParameter('x1', -2.25, 2.5) + \
                     gpflowopt.domain.ContinuousParameter('x2', -2.5, 1.75)
            return domain
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


class branin(BaseSingleObjectiveProblem):
    """
    y = (x(2)-(5.1/(4*pi^2))*x(1)^2+5*x(1)/pi-6)^2+10*(1-1/(8*pi))*cos(x(1))+10
    """
    optimal_value = 0.397887
    optimal_point = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate_config(self, config, optimizer='smac'):
        config_dict = self.get_config_dict(config, optimizer)
        x1 = config_dict['x1']
        x2 = config_dict['x2']
        X = np.array([x1, x2])
        return self.evaluate(X)

    def evaluate(self, X: np.ndarray):
        X = self.checkX(X)
        x1 = X[0]
        x2 = X[1]
        y = (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(
            x1) + 10
        return y

    @staticmethod
    def get_configspace(optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            x1 = UniformFloatHyperparameter("x1", -5, 10)
            x2 = UniformFloatHyperparameter("x2", 0, 15)
            cs.add_hyperparameters([x1, x2])
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'x1': hp.uniform('x1', -5, 10),
                     'x2': hp.uniform('x2', 0, 15),
                     }
            return space
        elif optimizer == 'gpflowopt':
            import gpflowopt
            domain = gpflowopt.domain.ContinuousParameter('x1', -5, 10) + \
                     gpflowopt.domain.ContinuousParameter('x2', 0, 15)
            return domain
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


class lgb(BaseSingleObjectiveProblem):
    def __init__(self, n_jobs=3, **kwargs):
        super().__init__(dim=7, **kwargs)
        self.n_jobs = n_jobs
        self.load_data(**kwargs)
        self.bounds = [
            (100, 1000),
            (31, 2047),
            (15, 16),
            (1e-3, 0.3),
            (5, 30),
            (0.7, 1),
            (0.7, 1),
        ]

    def evaluate_config(self, config, optimizer='smac'):
        config_dict = self.get_config_dict(config, optimizer)
        n_estimators = int(config_dict['n_estimators'])
        num_leaves = int(config_dict['num_leaves'])
        learning_rate = config_dict['learning_rate']
        min_child_samples = int(config_dict['min_child_samples'])
        subsample = config_dict['subsample']
        colsample_bytree = config_dict['colsample_bytree']
        from lightgbm import LGBMClassifier
        from sklearn.metrics.scorer import balanced_accuracy_scorer
        lgbc = LGBMClassifier(n_estimators=n_estimators,
                              num_leaves=num_leaves,
                              max_depth=15,
                              learning_rate=learning_rate,
                              min_child_samples=min_child_samples,
                              subsample=subsample,
                              colsample_bytree=colsample_bytree,
                              n_jobs=self.n_jobs)
        lgbc.fit(self.train_x, self.train_y)
        return -balanced_accuracy_scorer(lgbc, self.val_x, self.val_y)

    def evaluate(self, x):
        x = self.checkX(x)
        from lightgbm import LGBMClassifier
        from sklearn.metrics.scorer import balanced_accuracy_scorer
        lgbc = LGBMClassifier(n_estimators=int(x[0]),
                              num_leaves=int(x[1]),
                              max_depth=int(x[2]),
                              learning_rate=x[3],
                              min_child_samples=int(x[4]),
                              subsample=x[5],
                              colsample_bytree=x[6],
                              n_jobs=self.n_jobs)
        lgbc.fit(self.train_x, self.train_y)
        return -balanced_accuracy_scorer(lgbc, self.val_x, self.val_y)

    def get_configspace(self, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            n_estimators = UniformFloatHyperparameter("n_estimators", 100, 1000, default_value=500, q=50)
            num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 2047, default_value=128)
            learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
            min_child_samples = UniformIntegerHyperparameter("min_child_samples", 5, 30, default_value=20)
            subsample = UniformFloatHyperparameter("subsample", 0.7, 1, default_value=1, q=0.1)
            colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.7, 1, default_value=1, q=0.1)
            cs.add_hyperparameters([n_estimators, num_leaves, learning_rate, min_child_samples, subsample,
                                    colsample_bytree])
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'n_estimators': (hp.randint('lgb_n_estimators', 19) + 2) * 50,
                     'num_leaves': hp.randint('lgb_num_leaves', 2017) + 31,
                     'max_depth': 15,
                     'learning_rate': hp.loguniform('lgb_learning_rate', np.log(1e-3), np.log(0.3)),
                     'min_child_samples': hp.randint('lgb_min_child_samples', 26) + 5,
                     'subsample': (hp.randint('lgb_subsample', 4) + 7) * 0.1,
                     'colsample_bytree': (hp.randint('lgb_colsample_bytree', 4) + 7) * 0.1,
                     }
            return space
        elif optimizer == 'gpflowopt':
            from gpflowopt.domain import ContinuousParameter
            domain = (
                    ContinuousParameter('n_estimators', 100, 1000) +
                    ContinuousParameter('num_leaves', 31, 2047) +
                    ContinuousParameter('max_depth', 15, 16) +
                    ContinuousParameter("learning_rate", 1e-3, 0.3) +
                    ContinuousParameter("min_child_samples", 5, 30) +
                    ContinuousParameter("subsample", 0.7, 1) +
                    ContinuousParameter("colsample_bytree", 0.7, 1)
            )
            return domain
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


class svc(BaseSingleObjectiveProblem):
    def __init__(self, **kwargs):
        dataset = kwargs['dataset']
        from litebo.test_utils import load_data
        from sklearn.model_selection import train_test_split
        x, y = load_data(dataset, '../soln-ml/data/cls_datasets/')
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(x, y, stratify=y, random_state=1,
                                                                              test_size=0.3)
        super().__init__(**kwargs)

    def evaluate_config(self, config, optimizer='smac'):
        config_dict = self.get_config_dict(config, optimizer)
        penalty = config_dict['penalty']
        loss = config_dict.get('loss', None)
        dual = config_dict.get('dual', None)
        C = config_dict['C']
        tol = config_dict['tol']
        fit_intercept = config_dict['fit_intercept']
        intercept_scaling = config_dict['intercept_scaling']
        if isinstance(penalty, dict):
            combination = penalty
            penalty = combination['penalty']
            loss = combination['loss']
            dual = combination['dual']

        from sklearn.svm import LinearSVC
        from sklearn.metrics.scorer import balanced_accuracy_scorer
        if dual == 'True':
            dual = True
        elif dual == 'False':
            dual = False

        svcc = LinearSVC(penalty=penalty,
                         loss=loss,
                         dual=dual,
                         tol=tol,
                         C=C,
                         fit_intercept=fit_intercept,
                         intercept_scaling=intercept_scaling,
                         multi_class='ovr',
                         random_state=1)
        svcc.fit(self.train_x, self.train_y)
        return -balanced_accuracy_scorer(svcc, self.val_x, self.val_y)

    @staticmethod
    def get_configspace(optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()

            penalty = CategoricalHyperparameter(
                "penalty", ["l1", "l2"], default_value="l2")
            loss = CategoricalHyperparameter(
                "loss", ["hinge", "squared_hinge"], default_value="squared_hinge")
            dual = CategoricalHyperparameter("dual", ['True', 'False'], default_value='True')
            # This is set ad-hoc
            tol = UniformFloatHyperparameter(
                "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
            C = UniformFloatHyperparameter(
                "C", 0.03125, 32768, log=True, default_value=1.0)
            multi_class = Constant("multi_class", "ovr")
            # These are set ad-hoc
            fit_intercept = Constant("fit_intercept", "True")
            intercept_scaling = Constant("intercept_scaling", 1)
            cs.add_hyperparameters([penalty, loss, dual, tol, C, multi_class,
                                    fit_intercept, intercept_scaling])

            penalty_and_loss = ForbiddenAndConjunction(
                ForbiddenEqualsClause(penalty, "l1"),
                ForbiddenEqualsClause(loss, "hinge")
            )
            constant_penalty_and_loss = ForbiddenAndConjunction(
                ForbiddenEqualsClause(dual, "False"),
                ForbiddenEqualsClause(penalty, "l2"),
                ForbiddenEqualsClause(loss, "hinge")
            )
            penalty_and_dual = ForbiddenAndConjunction(
                ForbiddenEqualsClause(dual, "True"),
                ForbiddenEqualsClause(penalty, "l1")
            )
            cs.add_forbidden_clause(penalty_and_loss)
            cs.add_forbidden_clause(constant_penalty_and_loss)
            cs.add_forbidden_clause(penalty_and_dual)
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'penalty': hp.choice('liblinear_combination',
                                          [{'penalty': "l1", 'loss': "squared_hinge", 'dual': "False"},
                                           {'penalty': "l2", 'loss': "hinge", 'dual': "True"},
                                           {'penalty': "l2", 'loss': "squared_hinge", 'dual': "True"},
                                           {'penalty': "l2", 'loss': "squared_hinge", 'dual': "False"}]),
                     'loss': None,
                     'dual': None,
                     'tol': hp.loguniform('liblinear_tol', np.log(1e-5), np.log(1e-1)),
                     'C': hp.loguniform('liblinear_C', np.log(0.03125), np.log(32768)),
                     'multi_class': hp.choice('liblinear_multi_class', ["ovr"]),
                     'fit_intercept': hp.choice('liblinear_fit_intercept', ["True"]),
                     'intercept_scaling': hp.choice('liblinear_intercept_scaling', [1])}
            return space
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)
