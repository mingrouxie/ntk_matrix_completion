import numpy as np
import pandas as pd
import pdb
import os
import pathlib

from sklearn.model_selection import KFold
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_top_k_curves(top_accuracies):
    plt.plot(top_accuracies)
    plt.title("Top K Accuracy for Zeolites per OSDA")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(0, len(top_accuracies) + 1, step=5))
    plt.show()
    plt.draw()
    plt.savefig("top_k_accuracies.png", dpi=100)

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
    fig.savefig(file_name + ".png", dpi=150)


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
    fig.savefig(file_name + ".png", dpi=150)

def train_test_w_controls(allData, drugs_in_train, seed):
    np.random.seed(seed)
    allCells = allData.index.get_level_values('unit').drop_duplicates()

    train = None
    test = None

    for cell_type in allCells:
        cell_subset = allData[allData.index.get_level_values('unit') == cell_type]
        drug_set = cell_subset.index.get_level_values('intervention').drop_duplicates()
        train_drugs = np.random.choice(drug_set, size=drugs_in_train, replace=False)

        if train is None:
            train = cell_subset[cell_subset.index.get_level_values('intervention').isin(train_drugs)]
        else:
            train = pd.concat([train, cell_subset[cell_subset.index.get_level_values('intervention').isin(train_drugs)]])

        if test is None:
            test = cell_subset[~cell_subset.index.get_level_values('intervention').isin(train_drugs)]
        else:
            test = pd.concat([test, cell_subset[~cell_subset.index.get_level_values('intervention').isin(train_drugs)]])

    return train, test


def get_cosims(true, pred, bypass_epsilon_check = False, filter_value = None):
    cosims = []

    for row_id in range(true.shape[0]):
        i = true[row_id]
        j = pred[row_id]
        if filter_value:
            j = j[i != filter_value]
            i = i[i != filter_value]
        if bypass_epsilon_check or (not(np.abs(np.sum(i)) <= 1e-8 or np.abs(np.sum(j)) <= 1e-8)):
            cosims.append(1 - cosine(i, j))

    return cosims

def save_matrix(matrix, file_name, overwrite=True):
        file = os.path.abspath("")
        dir_main = pathlib.Path(file).parent.absolute()
        savepath = os.path.join(dir_main, file_name)
        if not(overwrite) and os.path.exists(savepath):
            overwrite = input(f"A file already exists at path {savepath}, do you want to overwrite? (Y/N): ")
        matrix.to_pickle(savepath)

def get_splits_in_zeolite_type(allData, metrics_mask, k=10, seed=5):
    fold_iterator = KFold(n_splits=k, shuffle=True, random_state=seed).split(allData)
    for _fold in range(k):
        train_idx, test_idx = next(fold_iterator)
        yield allData.iloc[train_idx], allData.iloc[test_idx], metrics_mask.iloc[test_idx]


def get_splits_in_cell_type(allData, k=10, seed=5):
    cell_types = list(allData.index.get_level_values("unit").drop_duplicates())

    fold_iterators = [KFold(n_splits=k, shuffle=True, random_state=seed) for cell_type in cell_types]

    for idx, cell_type in enumerate(cell_types):
        dataSlice = allData[allData.index.get_level_values("unit") == cell_type]
        fold_iterators[idx] = fold_iterators[idx].split(dataSlice)

    for fold in range(k):
        train = None
        test = None

        for idx, cell_type in enumerate(cell_types):
            train_idx, test_idx = next(fold_iterators[idx])
            dataSlice = allData[allData.index.get_level_values("unit") == cell_type]

            if train is None:
                train = dataSlice.iloc[train_idx]
                test = dataSlice.iloc[test_idx]
            else:
                train = train.append(dataSlice.iloc[train_idx])
                test = test.append(dataSlice.iloc[test_idx])

        yield train, test
    
# TODO: I don't love this built in assumption that all 2D matrices must be zeolite
# and that all 3D matrices must be CMAP... We can do better...
def get_splits(allData, k=10, seed=5):
    if len(allData.shape) == 3:
        return get_splits_in_cell_type(allData, k=10, seed =5)
    elif len(allData.shape) == 2:
        return get_splits_in_zeolite_type(allData, k=10, seed =5)
    else:
        raise ValueError(
            "Only matrices of 2 or 3 dimensions are currently supported, "+
            "but further data shapes can easily be implemented."
        )