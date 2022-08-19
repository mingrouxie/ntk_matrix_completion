import numpy as np
import pandas as pd
import os
import pdb
from sklearn import metrics
from math import ceil, floor

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib as mpl
from rdkit import Chem
from rdkit.Chem import RemoveAllHs, AddHs

# from rdkit.Chem import RemoveAllHs
import random

from random_seeds import ISOMER_SEED, SUBSTRATE_SEED


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
        fig.savefig("data/output/" + file_name + ".png", dpi=150)


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
    fig.savefig("../data/output/" + file_name + ".png", dpi=150)


def chunks(lst, n, chunk_train=False, chunk_metrics=None):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst) - n, n):
        leftside_index = i
        rightside_index = i + n
        if len(lst) - 2 * n < i: # for the last chunk that might be < n
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
    # random.seed(random_seed) # TODO: commented out because sample already uses a random_state
    clustered_isomers = pd.Series(cluster_isomers(all_data.index).values())
    clustered_isomers = clustered_isomers.sample(frac=1, random_state=random_seed)
    # Chunk by the isomer sets (train / test sets will not be balanced perfectly)
    nested_iterator = chunks(
        lst=clustered_isomers,
        n=floor(len(clustered_isomers) / k_folds),
        chunk_train=True,
    )
    # Now flatten the iterated isomer train / test sets
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


def plot_binding_energies(datas):
    print(f"plot_binding_energies not coded yet")
    return


def plot_spheres(datas):
    print("plot_spheres not coded yet")
    return


from sklearn.model_selection import StratifiedKFold, KFold


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


def report_best_scores(search, n_top=3, search_type='hyperopt'):
    """
    Function for reporting hyperparameter optimization results from output of
    sklearn's RandomizedSearchCV
    """
    if search_type == 'random':
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
    elif search_type == 'hyperopt':
        breakpoint()
