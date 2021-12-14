import numpy as np
import pandas as pd
import pdb

from sklearn.model_selection import KFold
from scipy.spatial.distance import cosine


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


def get_cosims(true, pred, bypass_epsilon_check = False):
    cosims = []

    for row_id in range(true.shape[0]):
        i = true[row_id]
        j = pred[row_id]

        if bypass_epsilon_check or (not(np.abs(np.sum(i)) <= 1e-8 or np.abs(np.sum(j)) <= 1e-8)):
            cosims.append(1 - cosine(i, j))

    return cosims

def get_splits_in_zeolite_type(allData, k=10, seed=5):
    fold_iterator = KFold(n_splits=k, shuffle=True, random_state=seed).split(allData)
    for _fold in range(k):
        train_idx, test_idx = next(fold_iterator)
        yield allData.iloc[train_idx], allData.iloc[test_idx]


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