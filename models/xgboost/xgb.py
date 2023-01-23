import os
import sys
import time
from datetime import datetime
import yaml
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xgboost
import argparse
from sklearn import preprocessing
import json


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
    scale_data,
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

    # get labels
    if kwargs["energy_type"] == Energy_Type.BINDING:
        # truth = pd.read_csv(BINDING_CSV) # binding values only
        truth = pd.read_csv(kwargs["truth"])  # binding with non-binding values
    else:
        print("[XGB] Please work only with binding energies")
        breakpoint()

    if kwargs["sieved_file"]:
        sieved_priors_index = pd.read_pickle(kwargs["sieved_file"]).index
        sieved_priors_index.name = "SMILES"
        truth = truth[truth["SMILES"].isin(sieved_priors_index)]

    truth = truth.set_index(["SMILES", "Zeolite"])
    truth = truth[["Binding (SiO2)"]]
    mask = pd.read_csv(kwargs["mask"])

    # get features
    print("[XGB] prior_method used is", kwargs["prior_method"])
    prior = make_prior(
        test=None,
        train=None,
        method=kwargs["prior_method"],
        normalization_factor=0,
        all_data=truth,
        stack_combined_priors=False,
        osda_prior_file=kwargs["osda_prior_file"],
        zeolite_prior_file=kwargs["zeolite_prior_file"],
        osda_prior_map=kwargs["osda_prior_map"],
        zeolite_prior_map=kwargs["zeolite_prior_map"],
        other_prior_to_concat=kwargs["other_prior_to_concat"],
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
            f"[XGB] Check prior shapes:",
            X_osda_handcrafted_prior.shape,
            X_osda_getaway_prior.shape,
            X_zeolite_prior.shape,
        )

        ### what to do with the retrieved priors
        if kwargs["prior_treatment"] == 1:
            X = X_osda_handcrafted_prior
        elif kwargs["prior_treatment"] == 2:
            X = np.concatenate([X_osda_handcrafted_prior, X_osda_getaway_prior], axis=1)
        elif kwargs["prior_treatment"] == 3:
            X = np.concatenate([X_osda_handcrafted_prior, X_zeolite_prior], axis=1)
        elif kwargs["prior_treatment"] == 4:
            X = np.concatenate([X_osda_getaway_prior, X_zeolite_prior], axis=1)
        elif kwargs["prior_treatment"] == 5:
            X = np.concatenate(
                [X_osda_handcrafted_prior, X_osda_getaway_prior, X_zeolite_prior],
                axis=1,
            )
        elif kwargs["prior_treatment"] == 6:
            X = X_zeolite_prior
        else:
            # if kwargs["stack_combined_priors"] == "all":
            #     X = np.concatenate(
            #         [X_osda_handcrafted_prior, X_osda_getaway_prior, X_zeolite_prior],
            #         axis=1,
            #     )
            # elif kwargs["stack_combined_priors"] == "osda":
            #     X = np.concatenate([X_osda_handcrafted_prior, X_osda_getaway_prior], axis=1)
            # elif kwargs["stack_combined_priors"] == "zeolite":
            #     X = X_zeolite_prior
            # else:
            print(f"[XGB] What do you want to do with the priors??")
            breakpoint()

    else:
        print(f"[XGB] prior_method {kwargs['prior_method']} not implemented")
        breakpoint()

    X = pd.DataFrame(X, index=truth.index)
    print("[XGB] Final prior X shape:", X.shape)

    # split data
    truth = truth.reset_index("Zeolite")

    if kwargs["split_type"] == SplitType.OSDA_ISOMER_SPLITS:
        # get train_test_split by isomers
        clustered_isomers = pd.Series(cluster_isomers(truth.index).values())
        clustered_isomers = clustered_isomers.sample(frac=1, random_state=ISOMER_SEED)
    else:
        print("[XGB] What data splits do you want?")
        breakpoint()
    clustered_isomers_train, clustered_isomers_test = train_test_split(
        clustered_isomers, test_size=0.1, shuffle=False, random_state=ISOMER_SEED
    )
    smiles_train = sorted(list(set().union(*clustered_isomers_train)))
    smiles_test = sorted(list(set().union(*clustered_isomers_test)))

    truth_train = truth.loc[smiles_train].reset_index().set_index(["SMILES", "Zeolite"])
    truth_test = truth.loc[smiles_test].reset_index().set_index(["SMILES", "Zeolite"])

    # scale ground truth if specified
    if kwargs["truth_scaler"]:
        truth_train_scaled, truth_test_scaled = scale_data(
            kwargs["truth_scaler"], truth_train, truth_test, kwargs["output"], "truth"
        )
    else:
        truth_train_scaled = truth_train
        truth_test_scaled = truth_test

    truth_train_scaled.to_pickle(
        os.path.join(kwargs["output"], "truth_train_scaled.pkl")
    )
    truth_test_scaled.to_pickle(os.path.join(kwargs["output"], "truth_test_scaled.pkl"))

    # split inputs
    X_train = X.loc[smiles_train]
    X_test = X.loc[smiles_test]

    # scale inputs
    X_train_scaled, X_test_scaled = scale_data(
        kwargs["input_scaler"], X_train, X_test, kwargs["output"], "input"
    )
    # print("[XGB] DEBUG: Check X and mask have been created properly")
    # breakpoint()
    X_train_scaled.to_pickle(os.path.join(kwargs["output"], "X_train_scaled.pkl"))
    X_test_scaled.to_pickle(os.path.join(kwargs["output"], "X_test_scaled.pkl"))

    # split mask
    mask_train = mask.set_index("SMILES").loc[smiles_train][["Zeolite", "exists"]]
    mask_train.to_pickle(os.path.join(kwargs["output"], "mask_train.pkl"))
    mask_test = mask.set_index("SMILES").loc[smiles_test][["Zeolite", "exists"]]
    mask_test.to_pickle(os.path.join(kwargs["output"], "mask_test.pkl"))

    # hyperparameter tuning
    if kwargs["tune"]:
        print("[XGB] Tuning hyperparameters")
        model, search = get_tuned_model(
            params=None,
            X=X_train_scaled,
            y=truth_train_scaled,
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
    # breakpoint()
    # fit and predict tuned model
    print("[XGB] Using tuned model")
    print(model)
    # model.set_params(eval_metric=) # TODO: figure this out
    # disable_default_eval_metric - do we need this?

    # TODO: save the information from here in order to plot train and test loss curves
    # eval_metric does not seem to allow custom eval_metrics from the most recent documentation, but
    # https://datascience.stackexchange.com/questions/99505/xgboost-fit-wont-recognize-my-custom-eval-metric-why

    model.set_params(early_stopping_rounds=10, eval_metric="rmse", n_estimators=1000)

    # eval set for early stopping only computes for binding pairs
    X_eval = (
        X_test_scaled.reset_index()
        .set_index("SMILES")
        .loc[mask_test[mask_test.exists == 1].index]
        .reset_index()
        .set_index(["SMILES", "Zeolite"])
    )
    truth_eval = (
        truth_test_scaled.reset_index()
        .set_index("SMILES")
        .loc[mask_test[mask_test.exists == 1].index]
        .reset_index()
        .set_index(["SMILES", "Zeolite"])
    )

    # If there’s more than one item in eval_set, the last entry will be used for early stopping.
    model.fit(
        X_train_scaled,
        truth_train_scaled,
        eval_set=[
            (
                X_train_scaled,
                truth_train_scaled,
            ),
            (X_eval, truth_eval),
        ],
        verbose=True,
    )
    # feature_weights=)
    model.save_model(os.path.join(kwargs["output"], "xgboost.json"))
    results = model.evals_result()
    np.save(kwargs["output"] + "/" + "train_loss.npy", results["validation_0"]["rmse"])
    np.save(kwargs["output"] + "/" + "test_loss.npy", results["validation_1"]["rmse"])

    y_pred_train = model.predict(X_train_scaled)
    df = pd.DataFrame(y_pred_train, columns=["Prediction"], index=truth_train.index)
    df["Binding (SiO2)"] = truth_train["Binding (SiO2)"]
    df.to_pickle(os.path.join(kwargs["output"], "train.pkl"))

    y_pred_test = model.predict(X_test_scaled)
    df = pd.DataFrame(y_pred_test, columns=["Prediction"], index=truth_test.index)
    df["Binding (SiO2)"] = truth_test["Binding (SiO2)"]
    df.to_pickle(os.path.join(kwargs["output"], "test.pkl"))

    # y_pred_all = model.predict(X) # Should have been on the SCALED. Is kay, we can use the train and test above
    # g = truth.reset_index().set_index(["SMILES", "Zeolite"])
    # df = pd.DataFrame(y_pred_all, columns=["Prediction"], index=g.index)
    # df["Binding (SiO2)"] = g["Binding (SiO2)"]
    # df.to_pickle(os.path.join(kwargs["output"], "all.pkl"))

    # performance
    print(
        "[XGB] Train score:",
        masked_rmse(
            truth_train["Binding (SiO2)"].values, y_pred_train, mask_train.exists
        ),
    )
    print(
        "[XGB] Test score:",
        masked_rmse(truth_test["Binding (SiO2)"].values, y_pred_test, mask_test.exists),
    )
    print(
        "[XBG] Unmasked train score:",
        np.sqrt(mean_squared_error(truth_train, y_pred_train)),
    )
    print(
        "[XBG] Unmasked test score:",
        np.sqrt(mean_squared_error(truth_test, y_pred_test)),
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
    kwargs = {
        "energy_type": Energy_Type.BINDING,
        "split_type": SplitType.OSDA_ISOMER_SPLITS,
        "k_folds": 5,
        # "model": None,
    }

    input = input.__dict__

    # make output directory
    if os.path.isdir(input["output"]):
        now = "_%d%d%d_%d%d%d" % (
            datetime.now().year,
            datetime.now().month,
            datetime.now().day,
            datetime.now().hour,
            datetime.now().minute,
            datetime.now().second,
        )
        input["output"] = input["output"] + now
    print("[XGB] Output folder is", input["output"])
    os.mkdir(input["output"], exist_ok=True)
    
    # transform some inputs
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

    # dump args
    yaml_args = yaml.dump(kwargs, indent=2, default_flow_style=False)
    with open(Path(kwargs["output"]) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    print("Output folder:", kwargs["output"], "\n")
    print(f"Args:\n{yaml_args}\n")

    return kwargs


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
    parser.add_argument(
        "--energy_type",
        help="Binding or templating energy",
        type=str,
        default="binding",
    )
    # parser.add_argument(
    #     "--stack_combined_priors",
    #     help="Treatment for stacking priors",
    #     type=str,
    #     required=True,
    #     default="all",
    # )
    parser.add_argument(
        "--prior_treatment",
        help="Which priors to concatenate to form the final prior. See main function for how it is used",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--search_type",
        help="Hyperparameter tuning method",
        type=str,
        # required=True,
        default="hyperopt",
    )
    parser.add_argument("--truth", help="Ground truth file", type=str, required=True)
    parser.add_argument("--mask", help="Mask file", type=str, required=True)
    parser.add_argument(
        "--sieved_file",
        help="Dataframe whose index is used to sieve for desired data points",
        type=str,
        default=None,  # TODO: generalize to substrate
    )
    parser.add_argument(
        "--osda_prior_file",
        help="OSDA prior file, only read if prior_method is CustomOSDAVector or CustomOSDAandZeoliteAsRows",
        type=str,
        default=OSDA_CONFORMER_PRIOR_FILE_CLIPPED,
    )
    parser.add_argument(
        "--other_prior_to_concat",
        help="2nd OSDA prior file to concat; this is a relic and bad practice. Currently not using it",
        type=str,
        default=None,
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
        "--truth_scaler",
        help="Scaling method for ground truth",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--input_scaler", help="Scaling method for inputs", type=str, default="standard"
    )
    parser.add_argument(
        "--split_seed", help="Data split seed", type=str, default=ISOMER_SEED
    )
    parser.add_argument(
        "--nthread",
        help="Number of threads for XGBoost",
        type=int,
        default=6,  # was 5 for runs 1 and 2
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
    parser.add_argument("--debug", action="store_true", dest="debug")
    args = parser.parse_args()
    kwargs = preprocess(args)
    main(kwargs)
