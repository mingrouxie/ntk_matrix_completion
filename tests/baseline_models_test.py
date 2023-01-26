import numpy as np
import xgboost

from utils.utilities import report_best_scores
from sklearn.linear_model import (
    LogisticRegressionCV,
    LinearRegression,
    RidgeCV,
    SGDRegressor,
    LassoCV,
)
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    train_test_split,
)
from scipy.stats import uniform, randint

from ntk_matrix_completion.utils.random_seeds import (
    HYPPARAM_SEED,
    ISOMER_SEED,
    MODEL_SEED,
)


def thoughts():
    # get ground truth data with get_ground_truth_energy_matrix
    # also these models, if I am only feeding in OSDA priors then it would just
    # predict the mean of the binding energies, right?
    # If I train on only one zeolite, that's way too little data
    # we could feed in the zeolite priors as well, that would help differentiate between the two

    # okay on the test splits
    # nothing to tune for NTK, can just use the results I alrdy have
    # for this, can implement the data splits w the isomers, save them if you paranoid
    # and then yknow yeah do the hyp param optimization
    # and report final best train, and test performance
    # Steal code if needed from psets

    # if i want to code in similar fashion then
    pass


def test_xgboost():
    X, y = load_diabetes(return_X_y=True)
    xgb_whole_dataset(X, y)
    xgb_cv(X, y, cv=10)
    xgb_kfold(X, y, n_splits=10)
    # xgb_hyp_search(X, y)
    test_get_tuned_xgb(X, y, cv=3)


def xgb_whole_dataset(X, y, random_seed=HYPPARAM_SEED):
    # step-by-step
    xgb = xgboost.XGBRegressor(objective="reg:squarederror", random_state=random_seed)
    xgb.fit(X, y)
    y_pred = xgb.predict(X)
    print("Whole dataset:", np.sqrt(mean_squared_error(y, y_pred)))


def xgb_cv(X, y, cv=10, random_seed=HYPPARAM_SEED):
    # CV
    scores = cross_val_score(
        xgboost.XGBRegressor(objective="reg:squarederror", random_state=random_seed),
        X,
        y,
        scoring="neg_mean_squared_error",
        cv=cv,
    )
    print("CV:", np.mean(np.sqrt(-scores)), np.std(np.sqrt(-scores)))


def xgb_kfold(X, y, n_splits=10, random_seed=HYPPARAM_SEED):
    # with k-fold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    scores = []
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        xgb = xgboost.XGBRegressor(
            objective="reg:squarederror", random_state=random_seed
        )
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        scores.append(mean_squared_error(y_test, y_pred))
    print("KFold:", np.mean(scores), np.std(scores))


def test_get_tuned_xgb(X, y, cv=3):
    # Hyperparameter search
    xgb = xgboost.XGBRegressor()
    params = {
        "colsample_bytree": uniform(0.7, 0.3),  # default 1.0
        "gamma": uniform(0, 0.5),  # default 0.0
        "learning_rate": uniform(0.03, 0.3),  # default 0.1
        "max_depth": randint(2, 8),  # default 3
        "n_estimators": randint(100, 150),  # default 100
        "subsample": uniform(0.6, 0.4),  # default 1.0
    }
    search = RandomizedSearchCV(
        xgb,
        param_distributions=params,
        random_state=HYPPARAM_SEED,
        n_iter=200,
        cv=cv,  # can be a cv generator with split and get_n_splits methods
        verbose=1,
        n_jobs=1,
        return_train_score=True,
    )
    search.fit(X, y)
    report_best_scores(search.cv_results_, 1)
    tuned_xgb = xgboost.XGBRegressor(*search.best_params_)
    return tuned_xgb
