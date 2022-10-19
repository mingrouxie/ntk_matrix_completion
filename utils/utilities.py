import os
import pdb
import random
from math import ceil, floor

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from ntk_matrix_completion.utils.random_seeds import ISOMER_SEED, SUBSTRATE_SEED
from rdkit import Chem
from rdkit.Chem import AddHs, RemoveAllHs
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from utils.path_constants import OUTPUT_DIR

# from rdkit.Chem import RemoveAllHs


def plot_matrix(M, file_name, mask=None, vmin=16, vmax=23, to_save=True):
    fig, ax = plt.subplots()
    cmap = mpl.cm.get_cmap()
    cmap.set_bad(color="white")
    if mask is not None:

        def invert_binary_mask(m):
            return np.logical_not(m).astype(int)

        inverted_mask = invert_binary_mask(mask)
        masked_M = np.ma.masked_where(inverted_mask, M)
    else:
        masked_M = M
    im = ax.imshow(masked_M, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im)
    if to_save:
        fig.savefig(OUTPUT_DIR + "/" + file_name + ".png", dpi=150)


def plot_two_matrices(
    M1, M1_title, M2, M2_title, file_name, mask=None, vmin=15, vmax=22
):
    fig, ax = plt.subplots()
    plt.set_cmap("plasma")

    cmap = mpl.cm.get_cmap()
    cmap.set_bad(color="white")
    if mask is not None:

        def invert_binary_mask(m):
            return np.logical_not(m).astype(int)

        inverted_mask = invert_binary_mask(mask)
        masked_M1 = np.ma.masked_where(inverted_mask, M1)
        masked_M2 = np.ma.masked_where(inverted_mask, M2)
    else:
        masked_M1 = M1
        masked_M2 = M2
    im = ax.imshow(masked_M1, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(M1_title)
    ax.set_ylabel("OSDAs")
    ax2 = fig.add_subplot(111)
    im2 = ax2.imshow(
        masked_M2, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax2.set_title(M2_title)
    fig.text(0.6, 0.04, "Zeolites", ha="center", va="center")
    fig.colorbar(im)
    fig.savefig(OUTPUT_DIR + "/" + file_name + ".png", dpi=150)


def chunks(lst, n, chunk_train=False, chunk_metrics=None):
    """
    Yield successive n-sized chunks from lst. Note the absence of seeds here.
    This method is purely to yield chunks.

    Inputs:

    lst: a list of entries
    n: size of chunks
    chunk_train: True if train_chunk is returned (TODO: when is this False)
    chunk_metrics: a list

    Returns:

    train_chunk, test_chunk, metrics_chunk
    """
    for i in range(0, len(lst) - n, n):
        leftside_index = i
        rightside_index = i + n
        if len(lst) - 2 * n < i:  # for the last chunk that might be < n
            rightside_index = len(lst)
        train_chunk = (
            pd.concat([lst[:leftside_index], lst[rightside_index:]])
            if chunk_train
            else None
        )
        test_chunk = lst[leftside_index:rightside_index]
        metrics_chunk = (
            chunk_metrics[leftside_index:rightside_index]
            if chunk_metrics is not None
            else None
        )
        yield train_chunk, test_chunk, metrics_chunk


def get_isomer_chunks(all_data, metrics_mask, k_folds, random_seed=ISOMER_SEED):
    """
    Inputs:

    all_data: Dataframe with an index of SMILES strings
    metrics_mask: array of same shape as all_data?
    k_folds: number of chunks to create
    random_seed: seed for shuffling isomer clusters

    Returns:

    An iterable of tuples (train, test and metrics (typically a mask))

    """
    clustered_isomers = pd.Series(cluster_isomers(smiles=all_data.index).values())
    # Shuffle the isomer clusters
    clustered_isomers = clustered_isomers.sample(frac=1, random_state=random_seed)
    # Chunk by the isomer sets (train / test sets will not be balanced perfectly)
    nested_iterator = chunks(
        lst=clustered_isomers,
        n=floor(len(clustered_isomers) / k_folds),
        chunk_train=True,
    )
    # Flatten the iterated isomer train / test sets
    for train, test, _ in nested_iterator:
        train_osdas = list(set().union(*train))
        test_osdas = list(set().union(*test))
        if np.any(metrics_mask):
            yield all_data.loc[train_osdas], all_data.loc[test_osdas], metrics_mask.loc[
                test_osdas
            ]
        else:
            # no masking of non-binding involved, used in baseline_models for predicting binding energies
            # print("Isomer train-test split:", len(train_osdas), len(test_osdas))
            # yield train_osdas, test_osdas
            yield all_data.index.isin(train_osdas), all_data.index.isin(test_osdas)
            # yield all_data.loc[train_osdas], all_data.loc[test_osdas]


def save_matrix(matrix, file_name, overwrite=True):
    file = os.path.abspath("")
    savepath = os.path.join(file, file_name)
    if not (overwrite) and os.path.exists(savepath):
        overwrite = input(
            f"A file already exists at path {savepath}, do you want to overwrite? (Y/N): "
        )
    matrix.to_csv(savepath.replace(".pkl", ".csv"))
    matrix.to_pickle(savepath)


def get_splits_in_zeolite_type(
    allData, metrics_mask, k=10, seed=SUBSTRATE_SEED, shuffle=True
):
    fold_iterator = KFold(n_splits=k, shuffle=shuffle, random_state=seed).split(allData)
    for _fold in range(k):
        train_idx, test_idx = next(fold_iterator)
        yield allData.iloc[train_idx], allData.iloc[test_idx], metrics_mask.iloc[
            test_idx
        ]


class IsomerKFold:
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        X = X.reset_index("Zeolite")
        breakpoint()
        iterator = get_isomer_chunks(X, metrics_mask=None, k_folds=self.n_splits)
        return iterator

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


def cluster_isomers(smiles):
    """
    Take the SMILES of our OSDAs and cluster all isomers for train / test / eval split
    using their iupac_name which should be stereo-isomer invariant.
    Test me with 'python tests/cluster_isomers_test.py'
    """
    nonisomeric_smiles_lookup = {}
    for smile in smiles:
        m = Chem.MolFromSmiles(smile)
        m = RemoveAllHs(m)
        # Remove isomeric information
        relaxed_smiles = Chem.rdmolfiles.MolToSmiles(m, isomericSmiles=False)
        smiles_set = nonisomeric_smiles_lookup.get(relaxed_smiles, set())
        smiles_set.add(smile)
        nonisomeric_smiles_lookup[relaxed_smiles] = smiles_set
    # bins = np.linspace(0, 60, 60)
    # plt.hist([len(c) for c in nonisomeric_smiles_lookup], bins=bins, alpha=0.5, label="isomer set sizes")
    # plt.title("OSDA Isomer Set Sizes")
    return nonisomeric_smiles_lookup


def report_best_scores(search, n_top=3, search_type="hyperopt"):
    """
    Function for reporting hyperparameter optimization results from output of
    sklearn's RandomizedSearchCV
    """
    if search_type == "random":
        results = search.cv_results_
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results["rank_test_score"] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print(
                    "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        results["mean_test_score"][candidate],
                        results["std_test_score"][candidate],
                    )
                )
                print("Parameters: {0}".format(results["params"][candidate]))
                print("")
    elif search_type == "hyperopt":
        print("Best parameters:", search.best_params_)
        print("")


def get_scaler(scaler_type):
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "quantile_normal": QuantileTransformer(output_distribution="normal"),
    }
    return scalers[scaler_type]


def scale_data(scaler_type: str, train: pd.DataFrame, test: pd.DataFrame, output_folder: str, data_type="truth"):
    scaler = get_scaler(scaler_type)
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train.values), index=train.index, columns=train.columns
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test.values), index=test.index, columns=test.columns
    )
    if output_folder:
        if scaler_type == "standard":
            gts_dict = {
                "mean": scaler.mean_.tolist(),
                "var": scaler.var_.tolist(),
            }
        elif scaler_type == "minmax":
            gts_dict = {"scale": scaler.scale_, "min": scaler.min_}
        filename = os.path.join(output_folder, data_type+"_scaling.json")
        with open(filename, "w") as gts:
            json.dump(gts_dict, gts)
    return train_scaled, test_scaled
