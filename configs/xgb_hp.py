from scipy.stats import uniform, randint
from hyperopt.pyll import scope
from hyperopt import hp
import numpy as np


def get_xgb_space():
    return {
        "colsample_bytree": uniform(0.0, 1.0),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3),  # default 0.1
        "max_depth": randint(2, 8),  # default 3
        "n_estimators": randint(100, 150),  # default 100
        "subsample": uniform(0.6, 0.4),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.0, 1.0),
        "min_child_weight": uniform(1.0, 10),
    }


def get_hyperopt_xgb_space():
    return {
        "learning_rate": hp.uniform("learning_rate", 0.03, 1.0),  # default 0.1
        "max_depth": hp.choice("max_depth", np.arange(2, 8, dtype=int)),  # default 3
        "gamma": hp.uniform("gamma", 0.0, 1.0), # default 0. Min loss red to continue partitioning. increasing makes model more conservative
        "reg_alpha": hp.uniform("reg_alpha", 0.0, 5.0), # default 0. increasing makes model more conservative
        "reg_lambda": hp.uniform("reg_lambda", 0.0, 5.0), # default 1. increasing makes model more conservative
        "n_estimators": hp.choice("n_estimators", np.arange(10, 2000, dtype=int)), # default 100
        "colsample_bytree": hp.uniform("colsample_bytree", 0.1, 1.0), # fraction of features to sample. prevent overfitting
        "subsample": hp.uniform("subsample", 0.1, 1.0), # fraction of data to sample. prevent overfitting
        "min_child_weight": hp.uniform("min_child_weight", 1.0, 100), # num of samples below which to stop partitioning
    }


def get_hyperopt_xgb_space_old():
    return {
        "colsample_bytree": hp.uniform("colsample_bytree", 0.01, 1.0),
        "gamma": hp.uniform("gamma", 0.0, 0.5),
        "learning_rate": hp.uniform("learning_rate", 0.03, 1.0),  # default 0.1

        ## Unable to restrict max_depth and n_estimators at the same time for the debug dataset

        # Works normally by itself
        "max_depth": hp.choice("max_depth", np.arange(2, 8, dtype=int)),  # default 3
        
        ## Unable to restrict n_estimators. The default is 100, but if below is uncommented it can go to 28
        "n_estimators": 200,
        # "n_estimators": hp.choice(
        #     "n_estimators", np.arange(100, 200, dtype=int)
        # ),  # default 100
        
        "subsample": hp.uniform("subsample", 0.4, 0.6),
        "reg_alpha": hp.uniform("reg_alpha", 0.0, 1.0),
        "reg_lambda": hp.uniform("reg_lambda", 0.0, 1.0),
        "min_child_weight": hp.uniform("min_child_weight", 1.0, 10),

        # "max_depth": scope.int(hp.quniform("max_depth", 2, 8, 1)),  # default 3
        # "n_estimators": scope.int(
        #     hp.quniform("n_estimators", 100, 150, 1)
        # ),  # default 100


        # "max_depth": scope.int(hp.quniform('max_depth', 3, 18, q=1)),
        # # "max_depth": hp.quniform("max_depth", 3, 18, 1),
        # "gamma": hp.uniform("gamma", 1, 9),
        # "reg_alpha": scope.int(hp.quniform("reg_alpha", 40, 180, 1)),
        # "reg_lambda": hp.uniform("reg_lambda", 0, 1),
        # "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        # "min_child_weight": scope.int(hp.quniform("min_child_weight", 0, 10, 1)),
        # "n_estimators": 180,
        # "seed": 0,
    }
