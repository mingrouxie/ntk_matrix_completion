import pandas as pd
import numpy as np
import os
import json
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime

from utils.logging import setup_logger
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from utils.package_matrix import Energy_Type
from models.neural_tangent_kernel.ntk import (
    SplitType,
)  # TODO: this should be in utils.package_matrix
from models.multitask.multitask_model import MLP
from ntk_matrix_completion.features.prior import make_prior
from ntk_matrix_completion.utils.loss import masked_rmse

def train_model_simple(kwargs):  # simple

    # extract data

    # define data loaders

    # define model

    # create trainer (tensorboard)

    # load from checkpoint

    return


def train_model(kwargs):

    # TODO: working notes
    # see sam's code for structure reference
    # see xgb on how to retrieve the data zzz

    # get labels
    truth = pd.read_csv(kwargs["truth"])
    if kwargs["sieved_file"]:
        sieved_priors_index = pd.read_pickle(kwargs["sieved_file"]).index
        sieved_priors_index.name = "SMILES"
        truth = truth[truth["SMILES"].isin(sieved_priors_index)]

    truth = truth.set_index(["SMILES", "Zeolite"])
    truth = truth[["Binding (SiO2)"]]
    mask = pd.read_csv(kwargs["mask"])

    # get features
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
    breakpoint()
    # build dataloader

    # featurize data in data loaders (?)

    # scale data

    # scale targets

    # hyperparam
    # run sigopt

    # feature selection based on train

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
    setup_logger(kwargs["output"], log_name="ffn_train.log", debug=kwargs["debug"])
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # transform some inputs
    kwargs["energy_type"] = Energy_Type(kwargs["energy_type"])
    kwargs["split_type"] = SplitType(kwargs["split_type"])

    # dump args
    yaml_args = yaml.dump(kwargs, indent=2, default_flow_style=False)
    with open(Path(kwargs["output"]) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    logging.info("Output folder:\n")
    logging.info(kwargs["output"])
    logging.info(f"Args:\n{yaml_args}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="multitask")
    parser.add_argument("--output", help="Output dir", required=True)
    parser.add_argument("--config", help="Config file", required=True)
    parser.add_argument(
        "--tune", help="Hyperparameter tuning", default="False", action="store_true"
    )
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--split_type", default=3, help="data split type")
    parser.add_argument("--energy_type", default=2, help="energy type")
    parser.add_argument(
        "--scale_ip", default=True, action="store_true", help="If true, scale features"
    )
    parser.add_argument(
        "--scale_energy",
        default=True,
        action="store_true",
        help="If true, scale energies",
    )
    parser.add_argument(
        "--scale_load",
        default=True,
        action="store_true",
        help="If true, scale loadings",
    )
    parser.add_argument(
        "--model", type=str, help="model type to choose", default=None
    )  # TODO: do not use now but use it later to generalize between XGB, maybe NTK, and variants of multi task models
    parser.add_argument(
        "--ignore-train",
        help="If true, ignore evaluation on the training set",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    kwargs = preprocess(args)
    train_model(kwargs)
