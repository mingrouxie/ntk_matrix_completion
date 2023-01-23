import pandas as pd
import numpy as np
import os
import json
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime

# from ntk_matrix_completion.utils.logger import setup_logger
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader

from ntk_matrix_completion.utils.package_matrix import Energy_Type
from ntk_matrix_completion.models.neural_tangent_kernel.ntk import (
    SplitType,
)  # TODO: this should be in utils.package_matrix
from ntk_matrix_completion.models.multitask.multitask_model import MLP
from ntk_matrix_completion.features.prior import make_prior
from ntk_matrix_completion.utils.package_matrix import (
    Energy_Type,
    get_ground_truth_energy_matrix,
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

from sklearn.model_selection import train_test_split

def train_simple(kwargs):  # simple

    # extract data

    # define data loaders

    # define model

    # create trainer (tensorboard)

    # load from checkpoint

    return


def train(kwargs):
    # TODO: working notes
    # see sam's code for structure reference
    # see xgb on how to retrieve the data zzz

#### TODO: dedup because code is the same as in xgb.py START

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
    print("[MT] prior_method used is", kwargs["prior_method"])
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
            print(f"[MT] What do you want to do with the priors??")
            breakpoint()

    else:
        print(f"[MT] prior_method {kwargs['prior_method']} not implemented")
        breakpoint()

    X = pd.DataFrame(X, index=truth.index)
    print("[MT] Final prior X shape:", X.shape)

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

#### TODO: dedup because code is the same as in xgb.py END

    

    # hyperparam
    # run sigopt
    if kwargs["tune"]:
        # tune(#truth, #prior, kwargs)
        pass

    # feature selection based on train? 

    # build model

    # train model

    # evaluation

    # remember to unscale targets

    # compute metrics

    # export results

    # save model

    return


def preprocess(args):
    kwargs = args.__dict__

    with open(kwargs['config'], "rb") as file:
        kwargs = yaml.load(file, Loader=yaml.Loader)

    if os.path.isdir(kwargs["output"]):
        now = "_%d%d%d_%d%d%d" % (
            datetime.now().year,
            datetime.now().month,
            datetime.now().day,
            datetime.now().hour,
            datetime.now().minute,
            datetime.now().second,
        )
        kwargs["output"] = kwargs["output"] + now
    print("[MT] Output folder is", kwargs["output"])
    os.mkdir(kwargs["output"], exist_ok=True)
    # setup_logger(kwargs["output"], log_name="multitask_train.log", debug=kwargs["debug"])
    # pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # transform some inputs
    kwargs["energy_type"] = Energy_Type(kwargs["energy_type"])
    kwargs["split_type"] = SplitType(kwargs["split_type"])

    if kwargs.get("split_type", None) == "naive":
        kwargs["split_type"] = SplitType.NAIVE_SPLITS
    elif kwargs.get("split_type", None) == "zeolite":
        kwargs["split_type"] = SplitType.ZEOLITE_SPLITS
    elif kwargs.get("split_type", None) == "osda":
        kwargs["split_type"] = SplitType.OSDA_ISOMER_SPLITS

    # dump args
    yaml_args = yaml.dump(kwargs, indent=2, default_flow_style=False)
    with open(Path(kwargs["output"]) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    print("Output folder:", kwargs["output"], "\n")
    print(f"Args:\n{yaml_args}\n")
    return kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="multitask")
    parser.add_argument("--config", help="Config file", required=True)



    args = parser.parse_args()
    kwargs = preprocess(args)
    train(kwargs)
