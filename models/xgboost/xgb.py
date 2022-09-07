import os
import sys
import time
import pandas as pd
import numpy as np
import xgboost
import matplotlib; matplotlib.use('Agg')

import matplotlib.pyplot as plt
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

from ntk_matrix_completion.utils.path_constants import (
    BINDING_CSV,
    BINDING_NB_CSV,
    MASK_CSV,
    OSDA_CONFORMER_PRIOR_FILE_CLIPPED,
    OSDA_CONFORMER_PRIOR_FILE_SIEVED,
    OSDA_PRIOR_FILE,
    XGBOOST_OUTPUT_DIR,
)
from ntk_matrix_completion.utils.utilities import (
    get_isomer_chunks,
    report_best_scores,
    cluster_isomers,
    IsomerKFold,
)
from ntk_matrix_completion.utils.package_matrix import (
    Energy_Type,
    get_ground_truth_energy_matrix,
)
from ntk_matrix_completion.features.prior import make_prior
from ntk_matrix_completion.models.neural_tangent_kernel.ntk import (
    create_iterator,
    SplitType,
)
from ntk_matrix_completion.utils.random_seeds import (
    HYPPARAM_SEED,
    ISOMER_SEED,
    MODEL_SEED,
)
from ntk_matrix_completion.configs.xgb_hp import get_xgb_space
from utils.hyperopt import HyperoptSearchCV


def get_tuned_model(
    params,
    X,
    y,
    k_folds=5,  # TODO: cv=5 arbitrary
    split_type=SplitType.OSDA_ISOMER_SPLITS,
    random_seed=MODEL_SEED,
    objective="reg:squarederror",  # "neg_root_mean_squared_error"
    nthread=5,
    search_type="random",
    mask=None
    # test_cross_val=False, # change here
):
    """Hyperparameter search for XGBRegressor"""
    print(f"[XGB] Doing hyperparameter optimization")
    start = time.time()

    # mask=None for IZC 2022 results. It is also not possible to implement for random and grid searches
    if not search_type == 'hyperopt':
        mask = None

    # get iterator for train-evaluation splits
    if split_type == SplitType.OSDA_ISOMER_SPLITS:
        # cv_generator = IsomerKFold(n_splits=cv)
        cv_generator = get_isomer_chunks(
            X.reset_index("Zeolite"),
            metrics_mask=mask,
            k_folds=k_folds,
        )
    else:
        print("[XGB] Other splits are not yet implemented")
        breakpoint()

    # tuning
    if search_type == "random":
        # IZC 2022 results were from this (no non-binding entries, no mask)
        model = xgboost.XGBRegressor(
            objective=objective, random_state=random_seed, nthread=nthread
        )
        params = get_xgb_space()
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

    elif search_type == "grid":
        model = xgboost.XGBRegressor(
            objective=objective, random_state=random_seed, nthread=nthread
        )
        params = get_xgb_space()
        search = GridSearchCV(
            model,
            param_grid=params,
            random_state=HYPPARAM_SEED,
            cv=cv_generator,
            verbose=3,
            n_jobs=1,
            return_train_score=True,
            error_score="raise",
        )
        y = y.reset_index().set_index(["SMILES", "Zeolite"])
        search.fit(X, y)

    elif search_type == "hyperopt":
        fixed_params = {
            "objective": objective,
            "random_state": random_seed,
            "nthread": nthread,
        }
        search = HyperoptSearchCV(X, y, cv=cv_generator, fixed_params=fixed_params, mask=mask)
        breakpoint()
        search.fit()

    # results
    report_best_scores(search, n_top=1, search_type=search_type)
    tuned_xgb = xgboost.XGBRegressor(
        objective=objective,
        random_state=random_seed,
        nthread=nthread,
        **search.best_params_,
    )
    print(f"[XGB] Time taken: {(time.time()-start)/60} minutes taken")
    return tuned_xgb, search


def main(
    energy_type=Energy_Type.BINDING,
    split_type=SplitType.OSDA_ISOMER_SPLITS,
    k_folds=5,
    # prior_method="CustomOSDAandZeoliteAsRows",
    prior_method="CustomOSDAVector",
    stack_combined_priors="all",
    osda_prior_file=OSDA_CONFORMER_PRIOR_FILE_CLIPPED,  # TODO: generalize to substrate too
    optimize_hyperparameters=True,
    output_dir=XGBOOST_OUTPUT_DIR,
    search_type="random",
    model=None,
):
    # clean data
    sieved_priors_index = pd.read_pickle(OSDA_CONFORMER_PRIOR_FILE_CLIPPED).index
    sieved_priors_index.name = "SMILES"
    if energy_type == Energy_Type.BINDING:
        # ground_truth = pd.read_csv(BINDING_CSV) # binding values only
        ground_truth = pd.read_csv(BINDING_NB_CSV) # binding with non-binding values
    else:
        print("[XGB] Please work only with binding energies")
        breakpoint()
    ground_truth = ground_truth[ground_truth["SMILES"].isin(sieved_priors_index)]
    ground_truth = ground_truth.set_index(["SMILES", "Zeolite"])
    ground_truth = ground_truth[["Binding (SiO2)"]]
    mask = pd.read_csv(MASK_CSV)

    # get priors
    print("[XGB] prior_method used is", prior_method)
    prior = make_prior(
        test=None,
        train=None,
        method=prior_method,
        normalization_factor=0,
        all_data=ground_truth,
        stack_combined_priors=False,
        osda_prior_file=OSDA_CONFORMER_PRIOR_FILE_CLIPPED,
    )

    if prior_method == "CustomOSDAVector":
        X = prior
        print(f"[XGB] Prior of shape {prior.shape}")

    elif prior_method == "CustomOSDAandVectorAsRows":
        X_osda_handcrafted_prior = prior[0] 
        X_osda_getaway_prior = prior[1]
        X_zeolite_prior = prior[2]
        
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
            print(f"[XGB] What do you want to do with the priors??")
            breakpoint()

    X = pd.DataFrame(X, index=ground_truth.index)
    print("[XGB] Final prior X shape:", X.shape)

    # split data
    ground_truth = ground_truth.reset_index("Zeolite")

    if split_type == SplitType.OSDA_ISOMER_SPLITS:
        # get train_test_split by isomers
        clustered_isomers = pd.Series(cluster_isomers(ground_truth.index).values())
        clustered_isomers = clustered_isomers.sample(frac=1, random_state=ISOMER_SEED)
    else:
        print("[XGB] What data splits do you want?")
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

    mask_train = mask.set_index('SMILES').loc[smiles_train]
    mask_test = mask.set_index('SMILES').loc[smiles_test]

    print("[XGB] DEBUG: Check X and mask have been created properly")
    breakpoint()
    
    # hyperparameter tuning
    if optimize_hyperparameters:
        print("[XGB] Tuning hyperparameters")
        model, search = get_tuned_model(
            params=None,
            X=X_train,
            y=ground_truth_train,
            k_folds=k_folds,
            search_type=search_type,
            mask=mask_train
        )

    # fit and predict tuned model
    print("[XGB] Using tuned model")
    # model.set_params(eval_metric=) # TODO: figure this out 
    # disable_default_eval_metric - do we need this?
    model.fit(
        X_train, 
        ground_truth_train,
        # eval_set=[(X_train, ground_truth_train), (X_test, ground_truth_test)],
        verbose=True,
        # feature_weights=
        )
    model.save_model(os.path.join(XGBOOST_OUTPUT_DIR, "xgboost.json"))

    y_pred_train = model.predict(X_train)
    df = pd.DataFrame(y_pred_train, columns=["Prediction"], index=ground_truth_train.index)
    df["Binding (SiO2)"] = ground_truth_train["Binding (SiO2)"]
    df.to_pickle(os.path.join(XGBOOST_OUTPUT_DIR, "train.pkl"))

    y_pred_test = model.predict(X_test)
    df = pd.DataFrame(y_pred_test, columns=["Prediction"], index=ground_truth_test.index)
    df["Binding (SiO2)"] = ground_truth_test["Binding (SiO2)"]
    df.to_pickle(os.path.join(XGBOOST_OUTPUT_DIR, "test.pkl"))

    y_pred_all = model.predict(X)
    g = ground_truth.reset_index().set_index(["SMILES", "Zeolite"])
    df = pd.DataFrame(y_pred_all, columns=["Prediction"], index=g.index)
    df["Binding (SiO2)"] = g["Binding (SiO2)"]
    df.to_pickle(os.path.join(XGBOOST_OUTPUT_DIR, "all.pkl"))

    # TODO: dumb way to check the RMSE is going down
    print("[XGB] Sanity check RMSE for train/test, includes non-bind:")
    print("Train score: ", np.sqrt(mean_squared_error(ground_truth_train, y_pred_train)))
    print("Test score: ", np.sqrt(mean_squared_error(ground_truth_test, y_pred_test)))

    # feature importance
    breakpoint()
    print("[XGB] model.feature_importances", model.feature_importances_)
    print(
        "[XGB] model.feature_importances shape", model.feature_importances_.shape
    )  # handcrafted 16 + getaway 273
    # fig, ax = plt.subplots(figsize=(16, 12))
    # ax = xgboost.plot_importance(model, max_num_features=10, ax=ax)
    # fig.savefig(os.path.join(XGBOOST_OUTPUT_DIR, "xgb_feature_importance.png"), dpi=300)

if __name__ == "__main__":
    # test_xgboost()

    # TODO
    # import argparse
    # parser = argparse.ArgumentParser(
    #     description='XGBoost model')
    # parser.add_argument(
    #     '--opt_hp',
    #     type=int,
    #     action='store_true',
    #     default=1,
    #     help='True to tune hyperparameters')
    # args = parser.parse_args()
    # main(args)

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

# print("[XGB] Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))
# 553 isomer groups from 1096 data points. very interesting
