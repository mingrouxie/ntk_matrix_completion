import os
import sys
import time
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import xgboost
import argparse
from sklearn import preprocessing


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ntk_matrix_completion.configs.xgb_hp import get_xgb_space
from ntk_matrix_completion.features.prior import make_prior
from ntk_matrix_completion.models.neural_tangent_kernel.ntk import (
    SplitType,
    create_iterator,
)
from ntk_matrix_completion.utils.hyperopt import HyperoptSearchCV
from ntk_matrix_completion.utils.loss import masked_rmse
from ntk_matrix_completion.utils.package_matrix import (
    Energy_Type,
    get_ground_truth_energy_matrix,
)
from ntk_matrix_completion.utils.path_constants import (
    BINDING_CSV,
    BINDING_NB_CSV,
    HANDCRAFTED_ZEOLITE_PRIOR_FILE,
    MASK_CSV,
    OSDA_CONFORMER_PRIOR_FILE_CLIPPED,
    OSDA_CONFORMER_PRIOR_FILE_SIEVED,
    OSDA_PRIOR_FILE,
    XGBOOST_OUTPUT_DIR,
    ZEOLITE_PRIOR_LOOKUP,
    OSDA_PRIOR_LOOKUP,
)
from ntk_matrix_completion.utils.random_seeds import (
    HYPPARAM_SEED,
    ISOMER_SEED,
    MODEL_SEED,
)
from ntk_matrix_completion.utils.utilities import (
    IsomerKFold,
    cluster_isomers,
    get_isomer_chunks,
    report_best_scores,
)
from sklearn.linear_model import (
    LassoCV,
    LinearRegression,
    LogisticRegressionCV,
    RidgeCV,
    SGDRegressor,
)
from sklearn.metrics import accuracy_score, auc, confusion_matrix, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)


def get_tuned_model(
    params,
    X,
    y,
    k_folds=5,  # TODO: cv=5 arbitrary
    split_type=SplitType.OSDA_ISOMER_SPLITS,
    model_seed=MODEL_SEED,
    objective="reg:squarederror",  # "neg_root_mean_squared_error"
    nthread=5,
    search_type="random",
    mask=None,
    debug=False,
    # test_cross_val=False, # change here
):
    """Hyperparameter search for XGBRegressor"""
    print(f"[XGB] Doing hyperparameter optimization")
    start = time.time()

    # mask=None for IZC 2022 results. It is also not possible to implement for random and grid searches
    if not search_type == "hyperopt":
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
            objective=objective, random_state=model_seed, nthread=nthread
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
            objective=objective, random_state=model_seed, nthread=nthread
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
            "random_state": model_seed,
            "nthread": nthread,
        }
        search = HyperoptSearchCV(
            X,
            y,
            # cv=cv_generator,
            fixed_params=fixed_params,
            mask=mask,
            output=kwargs["output"],
            seed=kwargs["split_seed"],  # not very sure
        )
        # breakpoint()
        search.fit(debug=debug)

    # results
    report_best_scores(search, n_top=1, search_type=search_type)
    tuned_xgb = xgboost.XGBRegressor(
        objective=objective,
        random_state=model_seed,
        nthread=nthread,
        **search.best_params_,
    )
    print(f"[XGB] Time taken: {(time.time()-start)/60} minutes taken")
    return tuned_xgb, search


def main(kwargs):
    """
    sieved_file: Pickle file that is read into a DataFrame,
        where the index contains the entries of interest
    """
    # make output directory
    if os.path.isdir(kwargs["output"]):
        print("[XGB] Output directory already exists, adding time to directory name")
        now = "_%d%d%d_%d%d%d" % (
            datetime.now().year,
            datetime.now().month,
            datetime.now().day,
            datetime.now().hour,
            datetime.now().minute,
            datetime.now().second,
        )
        kwargs["output"] = kwargs["output"] + now
    print("[XGB] Output file is", kwargs["output"])
    os.mkdir(kwargs["output"])

    # clean data
    sieved_priors_index = pd.read_pickle(kwargs["sieved_file"]).index
    sieved_priors_index.name = "SMILES"
    if kwargs["energy_type"] == Energy_Type.BINDING:
        # ground_truth = pd.read_csv(BINDING_CSV) # binding values only
        ground_truth = pd.read_csv(kwargs["truth"])  # binding with non-binding values
    else:
        print("[XGB] Please work only with binding energies")
        breakpoint()
    ground_truth = ground_truth[ground_truth["SMILES"].isin(sieved_priors_index)]
    ground_truth = ground_truth.set_index(["SMILES", "Zeolite"])
    ground_truth = ground_truth[["Binding (SiO2)"]]
    mask = pd.read_csv(kwargs["mask"])

    # get priors
    print("[XGB] prior_method used is", kwargs["prior_method"])
    prior = make_prior(
        test=None,
        train=None,
        method=kwargs["prior_method"],
        normalization_factor=0,
        all_data=ground_truth,
        stack_combined_priors=False,
        osda_prior_file=kwargs["osda_prior_file"],
        zeolite_prior_file=kwargs["zeolite_prior_file"],
        osda_prior_map=kwargs["osda_prior_map"],
        zeolite_prior_map=kwargs["zeolite_prior_map"],
    )
    # TODO: THIS IF THREAD IS RATHER UNKEMPT. WHEN WE GENERALIZE TO ZEOLITES....
    if kwargs["prior_method"] == "CustomOSDAVector":
        X = prior
        print(f"[XGB] Prior of shape {prior.shape}")

    elif kwargs["prior_method"] == "CustomOSDAandZeoliteAsRows":
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
        if kwargs["stack_combined_priors"] == "all":
            X = np.concatenate(
                [X_osda_handcrafted_prior, X_osda_getaway_prior, X_zeolite_prior],
                axis=1,
            )
        elif kwargs["stack_combined_priors"] == "osda":
            X = np.concatenate([X_osda_handcrafted_prior, X_osda_getaway_prior], axis=1)
        elif kwargs["stack_combined_priors"] == "zeolite":
            X = X_zeolite_prior
        else:
            print(f"[XGB] What do you want to do with the priors??")
            breakpoint()

    else:
        print(f"[XGB] prior_method {kwargs['prior_method']} not implemented")
        breakpoint()

    X = pd.DataFrame(X, index=ground_truth.index)
    print("[XGB] Final prior X shape:", X.shape)
    
    # split data
    ground_truth = ground_truth.reset_index("Zeolite")

    if kwargs["split_type"] == SplitType.OSDA_ISOMER_SPLITS:
        # get train_test_split by isomers
        clustered_isomers = pd.Series(cluster_isomers(ground_truth.index).values())
        clustered_isomers = clustered_isomers.sample(frac=1, random_state=ISOMER_SEED)
    else:
        print("[XGB] What data splits do you want?")
        breakpoint()
    clustered_isomers_train, clustered_isomers_test = train_test_split(
        clustered_isomers, test_size=0.1, shuffle=False, random_state=ISOMER_SEED
    )
    smiles_train = sorted(list(set().union(*clustered_isomers_train)))
    smiles_test = sorted(list(set().union(*clustered_isomers_test)))

    ground_truth_train = (
        ground_truth.loc[smiles_train].reset_index().set_index(["SMILES", "Zeolite"])
    )
    ground_truth_test = (
        ground_truth.loc[smiles_test].reset_index().set_index(["SMILES", "Zeolite"])
    )

    X_train = X.loc[smiles_train]
    X_test = X.loc[smiles_test]

    # scale inputs
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    mask_train = mask.set_index("SMILES").loc[smiles_train]
    mask_train.to_pickle(os.path.join(kwargs["output"], "mask_train.pkl"))
    mask_test = mask.set_index("SMILES").loc[smiles_test]
    mask_test.to_pickle(os.path.join(kwargs["output"], "mask_test.pkl"))

    # print("[XGB] DEBUG: Check X and mask have been created properly")
    # breakpoint()

    # hyperparameter tuning
    if kwargs["tune"]:
        print("[XGB] Tuning hyperparameters")
        model, search = get_tuned_model(
            params=None,
            X=X_train_scaled, #X_train,
            y=ground_truth_train,
            k_folds=kwargs["k_folds"],
            search_type=kwargs["search_type"],
            mask=mask_train,
            debug=kwargs["debug"],
            nthread=kwargs["nthread"],
            objective=kwargs["objective"],
            model_seed=kwargs["model_seed"],
        )
    else:
        model = xgboost.XGBRegressor()
        model.load_model(kwargs["model_file"])

    # fit and predict tuned model
    print("[XGB] Using tuned model")
    # model.set_params(eval_metric=) # TODO: figure this out
    # disable_default_eval_metric - do we need this?

    # TODO: save the information from here in order to plot train and test loss curves
    # eval_metric does not seem to allow custom eval_metrics from the most recent documentation, but
    # https://datascience.stackexchange.com/questions/99505/xgboost-fit-wont-recognize-my-custom-eval-metric-why

    model.set_params(
        eval_metric="rmse",  
        # TODO: AHHHHHHHHHHHhhhhhhhhhhhhhhhhhhhhhhhhhh
        # TODO: change to a custom eval_metric fn? Currently this means NB also included. 
        # Also change this to a parser argument if not custom
        # disable_default_eval_metric=
        early_stopping_rounds=10,
    )
    model.fit(
        X_train_scaled, #X_train,
        ground_truth_train,
        eval_set=[(X_train_scaled, ground_truth_train), (X_test_scaled, ground_truth_test)],
        # [(X_train, ground_truth_train), (X_test, ground_truth_test)],
        verbose=True,
        # feature_weights=
    )
    model.save_model(os.path.join(kwargs["output"], "xgboost.json"))
    results = model.evals_result()  #
    np.save(kwargs["output"] + "/" + "train_loss.npy", results["validation_0"]["rmse"])
    np.save(kwargs["output"] + "/" + "test_loss.npy", results["validation_1"]["rmse"])

    y_pred_train = model.predict(X_train_scaled)
    df = pd.DataFrame(
        y_pred_train, columns=["Prediction"], index=ground_truth_train.index
    )
    df["Binding (SiO2)"] = ground_truth_train["Binding (SiO2)"]
    df.to_pickle(os.path.join(kwargs["output"], "train.pkl"))

    y_pred_test = model.predict(X_test_scaled)
    df = pd.DataFrame(
        y_pred_test, columns=["Prediction"], index=ground_truth_test.index
    )
    df["Binding (SiO2)"] = ground_truth_test["Binding (SiO2)"]
    df.to_pickle(os.path.join(kwargs["output"], "test.pkl"))

    y_pred_all = model.predict(X)
    g = ground_truth.reset_index().set_index(["SMILES", "Zeolite"])
    df = pd.DataFrame(y_pred_all, columns=["Prediction"], index=g.index)
    df["Binding (SiO2)"] = g["Binding (SiO2)"]
    df.to_pickle(os.path.join(kwargs["output"], "all.pkl"))

    # performance
    print(
        "[XGB] Train score:",
        masked_rmse(
            ground_truth_train["Binding (SiO2)"].values, y_pred_train, mask_train.exists
        ),
    )
    print(
        "[XGB] Test score:",
        masked_rmse(
            ground_truth_test["Binding (SiO2)"].values, y_pred_test, mask_test.exists
        ),
    )
    print(
        "[XBG] Unmasked train score:",
        np.sqrt(mean_squared_error(ground_truth_train, y_pred_train)),
    )
    print(
        "[XBG] Unmasked test score:",
        np.sqrt(mean_squared_error(ground_truth_test, y_pred_test)),
    )

    # feature importance
    print(
        "[XGB] model.feature_importances:",
        model.feature_importances_.min(),
        model.feature_importances_.mean(),
        model.feature_importances_.max(),
    )  # handcrafted 16 + getaway 273 = 289 x 1
    np.save(
        os.path.join(kwargs["output"], "feature_importance"), model.feature_importances_
    )

    # fig, ax = plt.subplots(figsize=(16, 12))
    # ax = xgboost.plot_importance(model, max_num_features=10, ax=ax)
    # fig.savefig(os.path.join(XGBOOST_OUTPUT_DIR, "xgb_feature_importance.png"), dpi=300)

    print(f"[XGB] Finished. Output directory is {kwargs['output']}")


def preprocess(input):
    """
    Preprocess argparsed arguments into readable format
    """
    args = {
        "energy_type": Energy_Type.BINDING,
        "split_type": SplitType.OSDA_ISOMER_SPLITS,
        "k_folds": 5,
        # "model": None,
    }

    input = input.__dict__

    input["energy_type"] = (
        Energy_Type.BINDING
        if input.get("energy_type", None) == "binding"
        else Energy_Type.TEMPLATING
    )

    if input.get("split_type", None) == "naive":
        input["split_type"] = SplitType.NAIVE_SPLITS
    elif input.get("split_type", None) == "zeolite":
        input["split_type"] = SplitType.ZEOLITE_SPLITS
    elif input.get("split_type", None) == "osda":
        input["split_type"] = SplitType.OSDA_ISOMER_SPLITS

    args.update(input)
    return args


if __name__ == "__main__":
    # test_xgboost()

    parser = argparse.ArgumentParser(description="XGBoost")
    parser.add_argument("--output", help="Output directory", type=str, required=True)
    parser.add_argument(
        "--tune",
        help="Tune hyperparameters",
        action="store_true",
        dest="tune",
    )
    parser.add_argument(
        "--prior_method",
        help="method var in make_prior",
        type=str,
        default="CustomOSDAVector",
    )
    parser.add_argument("--debug", action="store_true", dest="debug")
    parser.add_argument(
        "--energy_type",
        help="Binding or templating energy",
        type=str,
        default="binding",
    )
    parser.add_argument(
        "--stack_combined_priors",
        help="Treatment for stacking priors",
        type=str,
        required=True,
        default="all",
    )
    parser.add_argument(
        "--search_type",
        help="Hyperparameter tuning method",
        type=str,
        required=True,
        default="hyperopt",
    )
    parser.add_argument("--truth", help="Ground truth file", type=str, required=True)
    parser.add_argument("--mask", help="Mask file", type=str, required=True)
    parser.add_argument(
        "--sieved_file",
        help="Dataframe whose index is used to sieve for desired data points",
        type=str,
        default=OSDA_CONFORMER_PRIOR_FILE_CLIPPED,  # TODO: generalize to substrate
    )
    parser.add_argument(
        "--osda_prior_file",
        help="OSDA prior file, only read if prior_method is CustomOSDAVector or CustomOSDAandZeoliteAsRows",
        type=str,
        default=OSDA_CONFORMER_PRIOR_FILE_CLIPPED,
    )
    parser.add_argument(
        "--zeolite_prior_file",
        help="Zeolite prior file, only read if prior_method is CustomOSDAandZeoliteAsRows",
        type=str,
        default=HANDCRAFTED_ZEOLITE_PRIOR_FILE,
    )
    parser.add_argument(
        "--osda_prior_map",
        help="Path to json file containing weights for OSDA descriptors",
        type=str,
        default=OSDA_PRIOR_LOOKUP,
    )
    parser.add_argument(
        "--zeolite_prior_map",
        help="Path to json file containing weights for zeolite descriptors",
        type=str,
        default=ZEOLITE_PRIOR_LOOKUP,
    )
    parser.add_argument(
        "--split_seed", help="Data split seed", type=str, default=ISOMER_SEED
    )
    parser.add_argument(
        "--nthread", help="Number of threads for XGBoost", type=int, default=5
    )
    parser.add_argument(
        "--objective",
        help="Objective for XGBoost",
        type=str,
        default="reg:squarederror",
    )
    parser.add_argument(
        "--model_seed", help="Seed for model", type=int, default=MODEL_SEED
    )
    parser.add_argument("--model_file", help="file to load model from", type=str)
    args = parser.parse_args()
    kwargs = preprocess(args)
    main(kwargs)

    # main(
    #     tune=True,
    #     stack_combined_priors="osda",
    #     search_type="hyperopt",
    # )

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
    # main(model=model, tune=False, stack_combined_priors="osda")

# print("[XGB] Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))
# 553 isomer groups from 1096 data points. very interesting
