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
import matplotlib.pyplot as plt



def get_tuned_model(
    params,
    X,
    y,
    k_folds=5,
    split_type=SplitType.OSDA_ISOMER_SPLITS,
    random_seed=MODEL_SEED,
    objective="reg:squarederror", # "neg_root_mean_squared_error"
    nthread=5,
    search_type="random",
    # test_cross_val=False, # change here
):
    """Hyperparameter search for XGBRegressor"""
    print(f"Doing hyperparameter optimization")
    start = time.time()

    # get iterator for train-evaluation splits
    if split_type == SplitType.OSDA_ISOMER_SPLITS:
        # cv_generator = IsomerKFold(n_splits=cv)
        cv_generator = get_isomer_chunks(
            X.reset_index("Zeolite"),
            metrics_mask=None,
            k_folds=k_folds,  # TODO: cv=5 arbitrary
        )
    else:
        print("What iterator do you want?")
        breakpoint()

    # tuning
    if search_type == "random":
        # IZC 2022 results were from this
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
        search = RandomizedSearchCV(
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
    elif search_type == "hyperopt":
        from test_hyperopt import HyperoptSearchCV

        fixed_params = {
            "objective": objective,
            "random_state": random_seed,
            "nthread": nthread,
        }
        search = HyperoptSearchCV(X, y, cv=cv_generator, fixed_params=fixed_params)
        search.fit()

    # results
    breakpoint()
    report_best_scores(search, n_top=1, search_type=search_type)
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
    k_folds=5,
    prior_method="CustomOSDAandZeoliteAsRows",  # "CustomOSDAVector",
    stack_combined_priors="all",
    osda_prior_file=OSDA_CONFORMER_PRIOR_FILE_CLIPPED,  # TODO: generalize to substrate too
    optimize_hyperparameters=True,
    model_file=XGBOOST_MODEL_FILE,
    search_type="random",
    model=None,
):
    # clean data
    sieved_priors_index = pd.read_pickle(OSDA_CONFORMER_PRIOR_FILE_CLIPPED).index
    sieved_priors_index.name = "SMILES"
    if energy_type == Energy_Type.BINDING:
        ground_truth = pd.read_csv(BINDING_CSV)
    else:
        print("Please work only with binding energies")
        breakpoint()
    ground_truth = ground_truth[ground_truth["SMILES"].isin(sieved_priors_index)]
    ground_truth = ground_truth.set_index(["SMILES", "Zeolite"])
    ground_truth = ground_truth[["Binding (SiO2)"]]

    # get priors
    X_osda_handcrafted_prior, X_osda_getaway_prior, X_zeolite_prior = make_prior(
        test=None,
        train=None,
        method=prior_method,
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

    # split data
    ground_truth = ground_truth.reset_index("Zeolite")
    if split_type == SplitType.OSDA_ISOMER_SPLITS:
        # get train_test_split by isomers
        clustered_isomers = pd.Series(cluster_isomers(ground_truth.index).values())
        clustered_isomers = clustered_isomers.sample(frac=1, random_state=ISOMER_SEED)
    else:
        print("What data splits do you want?")
        breakpoint()
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
            params=None,
            X=X_train,
            y=ground_truth_train,
            k_folds=k_folds,
            search_type=search_type,
        )

    # fit and predict tuned model
    print("Using tuned model")
    breakpoint()
    model.fit(X_train, ground_truth_train)
    model.save_model(model_file)
    y_pred = model.predict(X_train)
    print("Train score: ", np.sqrt(mean_squared_error(ground_truth_train, y_pred)))
    y_pred = model.predict(X_test)
    print("Test score: ", np.sqrt(mean_squared_error(ground_truth_test, y_pred)))
    y_pred - model.predict(X)
    print("Overall score: ", np.sqrt(mean_squared_error(ground_truth, y_pred)))
    breakpoint()
    xgboost.plot_importance(model)
    plt.figure(figsize = (16, 12))
    plt.savefig("data/output/baseline_model/xgb_feature_importance.png")
    breakpoint()


if __name__ == "__main__":
    # test_xgboost()

    main(
        optimize_hyperparameters=True,
        stack_combined_priors="osda",
        search_type="hyperopt",
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
