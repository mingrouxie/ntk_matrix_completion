import numpy as np
import pandas as pd
import os
import pdb

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib as mpl
from rdkit import Chem
import random


def plot_matrix(M, file_name, mask=None, vmin=16, vmax=23):
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
        if i == len(lst) - n:
            # Throw all the crumbs (i.e., last < 2n datapoints) into the final split
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


def get_isomer_chunks(all_data, metrics_mask, k_folds, random_seed=5):
    random.seed(random_seed)
    clustered_isomers = pd.Series(cluster_isomers(all_data.index).values())
    clustered_isomers = clustered_isomers.sample(frac=1, random_state=random_seed)
    # Chunk by the isomer sets (train / test sets will not be balanced perfectly)
    nested_iterator = chunks(
        lst=clustered_isomers,
        n=int(len(clustered_isomers) / k_folds),
        chunk_train=True,
    )
    # Now flatten the iterated isomer train / test sets
    for train, test, _ in nested_iterator:
        train_osdas = list(set().union(*train))
        test_osdas = list(set().union(*test))
        yield all_data.loc[train_osdas], all_data.loc[test_osdas], metrics_mask.loc[
            test_osdas
        ]

def save_matrix(matrix, file_name, overwrite=True):
    file = os.path.abspath("")
    savepath = os.path.join(file, file_name)
    if not (overwrite) and os.path.exists(savepath):
        overwrite = input(
            f"A file already exists at path {savepath}, do you want to overwrite? (Y/N): "
        )
    matrix.to_csv(savepath.replace(".pkl", ".csv"))
    matrix.to_pickle(savepath)


def get_splits_in_zeolite_type(allData, metrics_mask, k=10, seed=5, shuffle=True):
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
    
def cluster_isomers(smiles):
    """
    Take the SMILES of our OSDAs and cluster all isomers for train / test / eval split
    using their iupac_name which should be stereo-isomer invariant.
    Test me with 'python tests/cluster_isomers_test.py'
    """
    nonisomeric_smiles_lookup = {}
    for smile in smiles:
        m = Chem.MolFromSmiles(smile)
        m = Chem.RemoveAllHs(m)
        # Remove isomeric information
        relaxed_smiles = Chem.rdmolfiles.MolToSmiles(m, isomericSmiles=False)
        smiles_set = nonisomeric_smiles_lookup.get(relaxed_smiles, set())
        smiles_set.add(smile)
        nonisomeric_smiles_lookup[relaxed_smiles] = smiles_set
    # bins = np.linspace(0, 60, 60)
    # plt.hist([len(c) for c in nonisomeric_smiles_lookup], bins=bins, alpha=0.5, label="isomer set sizes")
    # plt.title("OSDA Isomer Set Sizes")
    return nonisomeric_smiles_lookup
