from pyexpat import model

from sklearn import cluster
from path_constants import (
    BINDING_CSV,
    OSDA_CONFORMER_PRIOR_FILE_CLIPPED,
    OSDA_CONFORMER_PRIOR_FILE_SIEVED,
    OSDA_PRIOR_FILE,
    XGBOOST_MODEL_FILE,
)
from utilities import get_isomer_chunks, report_best_scores, cluster_isomers
from utilities import IsomerKFold
from package_matrix import Energy_Type, get_ground_truth_energy_matrix
from prior import make_prior
from ntk import create_iterator, SplitType
import os
import sys
import time
import pandas as pd
import numpy as np
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
import xgboost
from random_seeds import HYPPARAM_SEED, ISOMER_SEED, MODEL_SEED


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
    xgb_hyp_search(X, y)


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


def get_tuned_model(
    model,
    params,
    X,
    y,
    cv=5,
    split_type=SplitType.OSDA_ISOMER_SPLITS,
    random_seed=MODEL_SEED,
    objective="reg:squarederror",
    nthread=5,
    model_file=XGBOOST_MODEL_FILE,
):
    # Hyperparameter search
    print(f"Doing hyperparameter optimization")
    start = time.time()
    if not model:
        model = xgboost.XGBRegressor(
            objective=objective, random_state=random_seed, nthread=nthread
        )
        params = {
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
    if split_type == SplitType.OSDA_ISOMER_SPLITS:
        # cv_generator = IsomerKFold(n_splits=cv)
        cv_generator = get_isomer_chunks(
            X.reset_index("Zeolite"),
            metrics_mask=None,
            k_folds=cv,  # TODO: cv=5 arbitrary
        )
    search = RandomizedSearchCV(
        # TODO: bayesian opt with hyper opt
        # https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook
        model,
        param_distributions=params,
        random_state=HYPPARAM_SEED,
        n_iter=100,  # TODO: arbitrary
        cv=cv_generator,
        verbose=3,
        n_jobs=1,
        return_train_score=True,
        error_score="raise",
    )
    y = y.reset_index().set_index(["SMILES", "Zeolite"])
    search.fit(X, y)
    report_best_scores(search.cv_results_, 1)
    tuned_xgb = xgboost.XGBRegressor(
        objective=objective,
        random_state=random_seed,
        nthread=nthread,
        **search.best_params_,
    )
    print(f"Time taken: {(time.time()-start)/60} minutes taken")
    return tuned_xgb, search


def main(
    energy_type=Energy_Type.BINDING,
    split_type=SplitType.OSDA_ISOMER_SPLITS,
    k_folds=10,
    seed=MODEL_SEED,
    model=None,
    prior="CustomOSDAVector",
    stack_combined_priors="all",
    osda_prior_file=OSDA_CONFORMER_PRIOR_FILE_CLIPPED,  # TODO: generalize to substrate too
    optimize_hyperparameters=True,
    model_file=XGBOOST_MODEL_FILE,
):
    sieved_priors_index = pd.read_pickle(OSDA_CONFORMER_PRIOR_FILE_CLIPPED).index
    sieved_priors_index.name = "SMILES"
    ground_truth = pd.read_csv(BINDING_CSV)
    ground_truth = ground_truth[ground_truth["SMILES"].isin(sieved_priors_index)]
    ground_truth = ground_truth.set_index(["SMILES", "Zeolite"])
    ground_truth = ground_truth[["Binding (SiO2)"]]

    X_osda_handcrafted_prior, X_osda_getaway_prior, X_zeolite_prior = make_prior(
        test=None,
        train=None,
        method="CustomOSDAandZeoliteAsRows",
        normalization_factor=0,
        all_data=ground_truth,
        stack_combined_priors=False,
        osda_prior_file=OSDA_CONFORMER_PRIOR_FILE_CLIPPED,
    )
    print(
        f"Check prior shapes:",
        X_osda_handcrafted_prior.shape,
        X_osda_getaway_prior.shape,
        X_zeolite_prior.shape,
    )
    ### what to do with the retrieved priors
    if stack_combined_priors == "all":
        X = np.concatenate(
            [X_osda_handcrafted_prior, X_osda_getaway_prior, X_zeolite_prior], axis=1
        )
    elif stack_combined_priors == "osda":
        X = np.concatenate([X_osda_handcrafted_prior, X_osda_getaway_prior], axis=1)
    elif stack_combined_priors == "zeolite":
        X = X_zeolite_prior
    else:
        print(f"What do you want to do with the priors??")
        breakpoint()
    X = pd.DataFrame(X, index=ground_truth.index)
    print("Final prior X shape:", X.shape)
    ground_truth = ground_truth.reset_index("Zeolite")
    # get train_test_split by isomers
    clustered_isomers = pd.Series(cluster_isomers(ground_truth.index).values())
    clustered_isomers = clustered_isomers.sample(frac=1, random_state=ISOMER_SEED)
    clustered_isomers_train, clustered_isomers_test = train_test_split(
        clustered_isomers, test_size=0.1, random_state=ISOMER_SEED
    )
    smiles_train = list(set().union(*clustered_isomers_train))
    smiles_test = list(set().union(*clustered_isomers_test))
    ground_truth_train = (
        ground_truth.loc[smiles_train].reset_index().set_index(["SMILES", "Zeolite"])
    )
    ground_truth_test = (
        ground_truth.loc[smiles_test].reset_index().set_index(["SMILES", "Zeolite"])
    )
    X_train = X.loc[smiles_train]
    X_test = X.loc[smiles_test]
    # hyperparameter tuning
    if optimize_hyperparameters:
        model, search = get_tuned_model(
            model,
            params=None,
            X=X_train,
            y=ground_truth_train,
            model_file=model_file,
        )
    # fit and predict tuned model
    breakpoint()
    model.fit(X_train, ground_truth_train)
    model.save_model(model_file)
    y_pred = model.predict(X_train)
    print("Train score: ", np.sqrt(mean_squared_error(ground_truth_train, y_pred)))
    y_pred = model.predict(X_test)
    print("Test score: ", np.sqrt(mean_squared_error(ground_truth_test, y_pred)))
    breakpoint()


if __name__ == "__main__":
    # test_xgboost()
    main(
        model=None,
        optimize_hyperparameters=True,
        stack_combined_priors="osda"
    )

    # params = {
    #     "colsample_bytree": 0.3181689154948444,
    #     "gamma": 0.1829854912145143,
    #     "learning_rate": 0.06989883691979182,
    #     "max_depth": 4,
    #     "min_child_weight": 2.6129103665660702,
    #     "n_estimators": 116,
    #     "reg_alpha": 0.8264051782292202,
    #     "reg_lambda": 0.47355178063941394,
    #     "subsample": 0.9532706328098646,
    # }
    # model = xgboost.XGBRegressor(
    #     random_state=MODEL_SEED,
    #     objective="reg:squarederror",
    #     nthread=5,
    #     **params
    # )
    # main(model=model, optimize_hyperparameters=False, stack_combined_priors="osda")

# print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))
# 553 isomer groups from 1096 data points. very interesting
