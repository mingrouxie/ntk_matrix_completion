import os, sys, glob, json, tqdm
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

sys.path.append("/home/mrx/projects/matrix_completion/")
# from ntk_matrix_completion.utils.loss import masked_rmse
from ntk_matrix_completion.utils.analysis_utilities import calculate_metrics
from ntk_matrix_completion.utils.package_matrix import Energy_Type
from sklearn.metrics import mean_squared_error
from math import sqrt

sys.path.append("/home/mrx/general/utils")
from figures import *

root = "/home/mrx/projects/matrix_completion/ntk_matrix_completion"
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.decomposition import PCA
import xgboost as xgb
from xgboost.plotting import plot_importance


def get_lig_data(kwargs):
    breakpoint()
    if kwargs["new_lig_dir"]:
        lig_files = sorted(glob.glob(kwargs["new_lig_dir"] + "/*"))
        lig_priors = [pd.read_pickle(file) for file in lig_files]
        lig_priors = pd.concat(lig_priors)
    if kwargs["new_lig_file"]:
        lig_priors = pd.read_pickle(kwargs["new_lig_file"])
    if kwargs["new_lig"]:
        lig_priors = lig_priors[lig_priors.index.isin(kwargs["new_lig"])]
    lig_priors = lig_priors[~lig_priors.index.duplicated(keep='first')]
    return lig_priors


def get_sub_data(kwargs):
    breakpoint()
    if kwargs["new_sub_dir"]:
        sub_files = sorted(glob.glob(kwargs["new_sub_dir"]+ "/*"))
        sub_priors = [pd.read_pickle(file) for file in sub_files]
        sub_priors = pd.concat(sub_priors)
    if kwargs["new_sub_file"]:
        sub_priors = pd.read_pickle(kwargs["new_sub_file"])
    if kwargs["new_sub"]:
        sub_priors = sub_priors[sub_priors.index.isin(kwargs["new_sub"])]
    sub_priors = sub_priors[~sub_priors.index.duplicated(keep='first')]
    return sub_priors


def get_energies(priors, model, idx, kwargs):
    """Predict energies and concat column to scaled features"""
    test = kwargs["lig_map"].keys()
    test_ls = list(test)
    breakpoint()
    print(priors[test_ls])
    lig_priors = priors.filter(items=kwargs["lig_map"].keys())
    sub_priors = priors.filter(items=kwargs["sub_map"].keys())
    priors_to_use = pd.concat([lig_priors, sub_priors], axis=1)
    priors_scaled = priors_to_use.mul(np.array(kwargs["ip_scale"]["scale"])).add(
        kwargs["ip_scale"]["min"]
    )

    # predict energies
    energies = model.predict(priors_scaled)
    data_scaled = pd.concat(
        [priors_scaled.reset_index(), pd.Series(energies, name="Binding (SiO2)")],
        axis=1,
    )
    data = pd.concat(
        [priors.reset_index(), pd.Series(energies, name="Binding (SiO2)")],
        axis=1,
    )

    # save data
    data.to_pickle(os.path.join(kwargs["output"]), "pred_"+str(idx)+".pkl")
    data_scaled.to_pickle(os.path.join(kwargs["output"]), "pred_scaled_"+str(idx)+".pkl")
    return data, data_scaled


def main(kwargs):
    """
    Main method for predicting energies for new ligands and/ or substrates using a pre-trained model. Outputs are saved to kwargs["output"]
    """
    if not os.path.isdir(kwargs["output"]):
        os.mkdir(kwargs["output"])

    # load model
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(kwargs["model_dir"], "xgboost.json"))
    
    # get priors
    lig_priors = get_lig_data(kwargs)
    sub_priors = get_sub_data(kwargs)
    
    # get weights of priors to be used
    with open(os.path.join(kwargs["lig_weights"]), "r") as f:
        kwargs["lig_map"] = json.load(f)
    with open(os.path.join(kwargs["sub_weights"]), "r") as f:
        kwargs["sub_map"] = json.load(f)
    
    # get input scaling of priors
    with open(os.path.join(kwargs["model_dir"], "input_scaling.json"), "r") as scalef:
        kwargs["ip_scale"] = json.load(scalef)
        print("scale for priors has keys", kwargs["ip_scale"].keys())

    # get entire prior dataframe - TODO: might need to chunk here
    pairs = np.array(list(product(lig_priors.index, sub_priors.index)))
    breakpoint()
    chunk_size = 10000
    chunk_num = pairs.shape[0] // chunk_size
    chunked_pairs = np.array_split(pairs, chunk_num)
    for idx, chunk in enumerate(tqdm(chunked_pairs)):
        breakpoint()
        l_priors = lig_priors.loc[chunk[:, 0]]
        s_priors = sub_priors.loc[chunk[:, 1]]
        priors = pd.concat([l_priors.reset_index(), s_priors.reset_index()], axis=1)
        priors = priors.rename(columns={"ligand": "SMILES", "fw": "Zeolite"})
        priors = priors.set_index(["SMILES", "Zeolite"])
        # predict
        get_energies(priors, model, idx, kwargs)
    return


def preprocess(args):
    return args.__dict__


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost")
    parser.add_argument("--output", help="Output directory", type=str, required=True)
    parser.add_argument(
        "--model_dir",
        help="Directory where model file and input_scaling is",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--new_lig_dir",
        help="Folder containing new ligand prior files. Include an asterisk, as glob.glob is used on this to grab all the files in the folder",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--new_sub_dir",
        help="Folder containing new substrate prior files. Include an asterisk, as glob.glob is used on this to grab all the files in the folder",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--new_lig_file", help="New ligand prior file", type=str, default=None
    )
    parser.add_argument(
        "--new_sub_file", help="New substrate prior file.", type=str, default=None
    )
    # parser.add_argument(
    #     "--science",
    #     help="Science paper OSDAs",
    #     type=str,
    #     default="/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/iza_all/osda_priors_20221120_6230_0.pkl",
    # )
    parser.add_argument(
        "--new_lig",
        help="Specific ligands (SMILES) to predict for",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--new_sub",
        help="Specific substrates to predict for",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--lig_weights",
        help="Weight map JSON file for ligand priors",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sub_weights",
        help="Weight map JSON file for substrate priors",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    kwargs = preprocess(args)
    main(kwargs)




### TODO:
# In case I forget, we're making it easy to predict
# sieving out the Science OSDAs or whatever can be done in a notebook
# We just need to make sure we get the predictions saved somewhere,
# Ideally both the scaled and unscaled
# And also include the features that were not used - ie. they wouldn't be scaled in the scaled data section