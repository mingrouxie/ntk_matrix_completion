import os
import pdb
import random
import torch
from tqdm import tqdm
from math import ceil, floor
from enum import Enum, IntEnum
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
from ntk_matrix_completion.utils.path_constants import OUTPUT_DIR
from torch.utils.data import TensorDataset, DataLoader, Dataset
from typing import Callable, Optional, List, Tuple, Dict, Union

# from rdkit.Chem import RemoveAllHs

class SplitType(IntEnum):
    NAIVE_SPLITS = 1
    ZEOLITE_SPLITS = 2
    OSDA_ISOMER_SPLITS = 3

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

        If metrics_mask is provided, returns an iterable of tuples (train, test and metrics_mask), where each item in the tuple is a DataFrame with the same index of SMILES strings. If metrics_mask is not provided, an iterable of tuples (train, test) is returned
    """
    clustered_isomers = pd.Series(cluster_isomers(smiles=all_data.index).values())
    # Shuffle the isomer clusters
    clustered_isomers = clustered_isomers.sample(frac=1, random_state=random_seed)
    # Chunk by the isomer sets
    print("[utils/get_isomer_chunks] Note that train/test split is not perfect due to differing size of each isomer group")
    nested_iterator = chunks(
        lst=clustered_isomers,
        n=floor(len(clustered_isomers) / k_folds),
        chunk_train=True,
    )
    # Flatten the iterated isomer train / test sets
    for train, test, _ in nested_iterator:
        train_osdas = list(set().union(*train))
        test_osdas = list(set().union(*test))
        print("[utils/get_isomer_chunks] train/test length:", len(train_osdas), len(test_osdas))
        if np.any(metrics_mask):
            yield all_data.index.isin(train_osdas), all_data.index.isin(train_osdas), metrics_mask.index.isin(train_osdas)
        else:
            # no masking of non-binding involved, used in baseline_models for predicting binding energies
            yield all_data.index.isin(train_osdas), all_data.index.isin(test_osdas)


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
                "scaler_type": "standard",
                "mean": scaler.mean_.tolist(), # json cannot save array
                "var": scaler.var_.tolist(),
            }
        elif scaler_type == "minmax":
            gts_dict = {
                "scaler_type": "minmax", 
                "scale": scaler.scale_.tolist(), 
                "min": scaler.min_.tolist(),
                "data_min": scaler.data_min_.tolist(),
                "data_max": scaler.data_max_.tolist(),
                }
        else:
            print("[utilities/scale_data] Scaler type not known")
        filename = os.path.join(output_folder, data_type+"_scaling.json")
        with open(filename, "w") as gts:
            json.dump(gts_dict, gts)
    return train_scaled, test_scaled, gts_dict

def unscale_data(scaler_dict, data):
    # I think there's an internal method if you can set up the scaler object lol
    if scaler_dict["scaler_type"] == 'minmax':
        data -= scaler_dict["min"]
        data /= scaler_dict["scale"]
    elif scaler_dict["scaler_type"] == 'standard': 
        data = data * np.sqrt(scaler_dict["var"]) + scaler_dict["mean"]
    return data


def create_iterator(split_type, all_data, metrics_mask, k_folds, seed): 
    """
    Inputs:

        split_type: method of constructing data splits
        all_data: Dataframe where the index is used to create the iterator based on the split_type
        E.g. for NTK this is a DataFrame of binding energies. For XGB this is a DataFrame of priors.
        For both examples the index of the DataFrame is SMILES
        metrics_mask: array with 1 for binding and 0 for non-binding entries
        k_folds: number of folds to create
        seed: seed for splits in zeolite_types

    Returns:
        
        An iterator that returns the following in each iteration:
        - train: portion of all_data for training
        - test: portion of all_data for testing
        - test_mask_chunk: binding/non-binding mask for test
    """
    if split_type == SplitType.NAIVE_SPLITS:
        # The iterator shuffles the data which is why we need to pass in metrics_mask together.
        iterator = tqdm(
            get_splits_in_zeolite_type(all_data, metrics_mask, k=k_folds, seed=seed),
            total=k_folds,
        )
    elif split_type == SplitType.ZEOLITE_SPLITS:
        # This branch is only for skinny matrix which require that
        # we chunk by folds to be sure we don't spill zeolites/OSDA rows
        # between training & test sets
        assert len(all_data) % k_folds == 0, (
            "[create_iterator] skinny_matrices need to be perfectly modulo by k_folds in order to avoid leaking training/testing data"
        )
        iterator = tqdm(
            chunks(
                lst=all_data,
                n=int(len(all_data) / k_folds),
                chunk_train=True,
                chunk_metrics=metrics_mask,
            ),
            total=k_folds,
        )
    elif split_type == SplitType.OSDA_ISOMER_SPLITS:
        # split OSDAs by isomers. The number of OSDAs in each split might not be equal 
        # due to different number of OSDAs in each isomer cluster
        assert (
            all_data.index.name == "SMILES"
        ), "[create_iterator] The OSDA isomer split is currently only implemented for OSDAs as rows"
        iterator = tqdm(
            get_isomer_chunks(
                all_data,
                metrics_mask,
                k_folds,
                random_seed=seed
            )
        )
    else:
        raise Exception("[create_iterator] Need to provide a SplitType for run_ntk(), xgb hyperopt,")
    return iterator


class MultiTaskTensorDataset(Dataset):
    """Dataset wrapping tensors and molecule-zeolite pair indices (list)

    Each sample will be retrieved by indexing tensors along the first dimension (the list is indexed as is).

    Args:
        *iterables (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *iterables) -> None:
        tensors = iterables[:-1] # X, y, mask, _ 
        pair_indices = iterables[-1] # smiles-zeolite pairs
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        assert len(pair_indices) == tensors[0].size(0), "Size mismatch between tensor and pair indices"
        self.tensors = tensors
        self.pair_indices = pair_indices

    def __getitem__(self, index):
        item = [tensor[index] for tensor in self.tensors]
        item.extend([self.pair_indices[index]])
        return item

    def __len__(self):
        return self.tensors[0].size(0)

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

def get_optimizer(opt_type, **kwargs): 
    # TODO: SGD has momentum and some others, Adam also has tunable parameters
    return OPTIMIZERS[opt_type](params=kwargs['params'], lr=kwargs['lr'])


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Copied from https://github.com/samgoldman97/enz-pred/, 
    which states that the code is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    Noam learning rate scheduler with piecewise linear increase and exponential decay.
    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). 
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: List[Union[float, int]],
        total_epochs: List[int],
        steps_per_epoch: int,
        init_lr: List[float],
        max_lr: List[float],
        final_lr: List[float],
        **kwargs,
    ):
        """
        Initializes the learning rate scheduler.
        Args:
            optimizer: A PyTorch optimizer.
            warmup_epochs (List[Union[float, int]]): The number of epochs during which to linearly increase the learning rate.
            total_epochs (List[int]): The total number of epochs.
            steps_per_epoch (int): The number of steps (batches) per epoch.
            init_lr (List[float]): The initial learning rate.
            max_lr (List[float]): The maximum learning rate (achieved after warmup_epochs).
            final_lr (List[float]): The final learning rate (achieved after total_epochs).
            kwargs :
        """
        assert (
            len(optimizer.param_groups)
            == len(warmup_epochs)
            == len(total_epochs)
            == len(init_lr)
            == len(max_lr)
            == len(final_lr)
        )

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (
            1 / (self.total_steps - self.warmup_steps)
        )

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.
        Args:
            current_step (int): Optionally specify what step to set the learning rate to.
                If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = (
                    self.init_lr[i] + self.current_step * self.linear_increment[i]
                )
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (
                    self.exponential_gamma[i]
                    ** (self.current_step - self.warmup_steps[i])
                )
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]["lr"] = self.lr[i]


SCHEDULERS = {
    'step': torch.optim.lr_scheduler.StepLR,
    'lambda': torch.optim.lr_scheduler.LambdaLR,
    'multistep': torch.optim.lr_scheduler.MultiStepLR,
    'exponential': torch.optim.lr_scheduler.ExponentialLR,
    'cosineannealing': torch.optim.lr_scheduler.CosineAnnealingLR,
    'cyclic': torch.optim.lr_scheduler.CyclicLR,
    'noam': NoamLR,
}

def get_scheduler(sched_type, kwargs):
    return SCHEDULERS[sched_type](**kwargs)