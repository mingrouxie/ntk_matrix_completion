import sys
import pathlib
import os
from typing_extensions import TypedDict

from prior import make_prior
import matplotlib.pyplot as plt
import pdb
import numpy as np
import pandas as pd
from auto_tqdm import tqdm
from precompute_osda_priors import smile_to_property
from scipy.sparse import csc_matrix
import scipy as sp
import time
from utilities import plot_matrix

from prior import zeolite_prior
from analysis_utilities import calculate_top_k_accuracy
from ooc_matrix_multiplication import ooc_dot
from eigenpro.eigenpro import FKR_EigenPro
import torch



sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))

from package_matrix import (
    Energy_Type,
    get_ground_truth_energy_matrix,
    make_skinny,
    unmake_skinny,
)
from utilities import (
    get_splits_in_zeolite_type,
    save_matrix,
    chunks,
)
from analysis_utilities import calculate_metrics
from path_constants import (
    HYPOTHETICAL_OSDA_ENERGIES,
    HYPOTHETICAL_OSDA_BOXES,
    OSDA_HYPOTHETICAL_PRIOR_FILE,
    OSDA_HYPOTHETICAL_PREDICTED_ENERGIES,
    ZEOLITE_HYPOTHETICAL_PREDICTED_ENERGIES,
    TEN_FOLD_CROSS_VALIDATION_ENERGIES,
    ZEO_1_PRIOR,
)


sys.path.insert(
    1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical")
)

SEED = 5
NORM_FACTOR = 0.001
PI = np.pi

def kappa(x):
    return (x * (PI - np.arccos(x)) + np.sqrt(1 - np.square(x))) / PI + (
        x * (PI - np.arccos(x))
    ) / PI

def kappa_with_clip(X_0, X_1):
    # torch.float32 TODO: make this torch.float32 like gaussian does....
    return np.float32(kappa(np.clip(X_0 @ X_1.T, -1, 1)).float())

def predict(all_data, mask, num_test_rows, X, reduce_footprint=False):
    """
    Run the NTK Matrix Completion Algorithm
    https://arxiv.org/abs/2108.00131
    """
    if reduce_footprint:
        # For whatever reason this throws everything off...
        X = X.astype(np.float32)
        all_data = all_data.astype(np.float32)
        mask = mask.astype(np.float32)
    all_data = all_data.T 
    mask = mask.T 
    num_observed = int(np.sum(mask[0:1, :])) 
    num_missing = mask[0:1, :].shape[-1] - num_observed

    K_matrix = np.zeros((num_observed, num_observed))
    k_matrix = np.zeros((num_observed, num_missing))
    observed_data = all_data[:, :num_observed]
    observed_data = observed_data.astype('float64')
    X_cross_terms = kappa(np.clip(X @ X.T, -1, 1))
    K_matrix[:, :] = X_cross_terms[:num_observed, :num_observed]
    k_matrix[:, :] = X_cross_terms[
        :num_observed, num_observed : num_observed + num_missing
    ]
    # plot_matrix(X_cross_terms, 'X_cross_terms', vmin=0, vmax=2)
    # plot_matrix(k_matrix, 'little_k', vmin=0, vmax=2)
    # plot_matrix(K_matrix, 'big_K', vmin=0, vmax=2)
    # plot_matrix(X, 'X', vmin=0, vmax=1)
    # breakpoint()
    results = np.linalg.solve(K_matrix, observed_data.T).T @ k_matrix
    assert results.shape == (all_data.shape[0], num_test_rows), "Results malformed"
    # breakpoint()
    assert (np.any(np.isnan(results)) == False)
    return results.T


def run_ntk(
    all_data,
    prior,
    metrics_mask,
    shuffled_iterator=True,
    k_folds=10,
    SEED=SEED,
    prior_map=None,
    norm_factor=NORM_FACTOR,
    use_eigenpro=False,
):
    # breakpoint()
    if shuffled_iterator:
        # The iterator shuffles the data which is why we need to pass in metrics_mask together.
        iterator = tqdm(
            get_splits_in_zeolite_type(all_data, metrics_mask, k=k_folds, seed=SEED),
            total=k_folds,
        )
    else:
        # This branch is only for skinny matrix which require that
        # we chunk by folds to be sure we don't spill zeolites/OSDA rows
        # between training & test sets
        assert len(all_data) % k_folds == 0, (
            "A bit silly, but we do require skinny_matrices to be perfectly modulo"
            + " by k_folds in order to avoid leaking training/testing data"
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
    aggregate_pred = None  # Predictions for all fold(s)
    aggregate_true = None  # Ground truths for all fold(s)
    aggregate_mask = None  # Non-binding masks for all fold(s) 

    # Iterate over all predictions and populate metrics matrices
    for train, test, test_mask_chunk in iterator:
        X = make_prior(
            train,
            test,
            prior,
            normalization_factor=norm_factor,
            prior_map=prior_map,
        )
        all_data = pd.concat([train, test]).to_numpy()
        ##### SAFETY
        all_data[train.shape[0] :, :] = 0
        ##### SAFETY
        mask = np.ones_like(all_data)
        mask[len(train) :, :] = 0
        # The bottom 1/10 of mask and all_data are just all zeros.
        # breakpoint()
        assert(np.any(np.isnan(X)==False))
        X = np.nan_to_num(X, copy=True, nan=0.0) 
        # TODO (Mingrou): Check with Yitong why the code worked before without this line
        results_ntk = predict(all_data, mask, len(test), X=X)
        prediction_ntk = pd.DataFrame(
            data=results_ntk, index=test.index, columns=test.columns
        )
        aggregate_pred = (
            prediction_ntk
            if aggregate_pred is None
            else pd.concat([aggregate_pred, prediction_ntk])
        )
        aggregate_true = (
            pd.concat([aggregate_true, test]) if aggregate_true is not None else test
        )
        aggregate_mask = (
            pd.concat([aggregate_mask, test_mask_chunk])
            if aggregate_mask is not None
            else test_mask_chunk
        )
    # We return aggregate_pred, aggregate_true, aggregate_mask.
    # Aggregate_mask is necessary to keep track of which cells in the matrix are 
    # non-binding (for spearman & rmse calculation) when shuffled_iterator=True
    # breakpoint()
    return aggregate_pred, aggregate_true, aggregate_mask


# TODO: This is in development and pretty god awful.
# TODO: make sample_size bigger, maybe 10K?
def skinny_ntk_sampled_not_sliced(
    sample_size=100, num_splits=10, seed=SEED, use_eigenpro=True
):
    """
    This method runs 10-fold cross validation on the 1194x209 Ground Truth matrix made SKINNY
    But sampled not sliced from the top left corner (100, 30) style
    """
    ground_truth, binary_data = get_ground_truth_energy_matrix(desired_shape=(100, 30))
    iterator = tqdm(
        get_splits_in_zeolite_type(ground_truth, binary_data, k=num_splits, seed=SEED),
        total=num_splits,
    )
    aggregate_pred = None  # Predictions for all fold(s)
    aggregate_true = None  # Ground truths for all fold(s)
    aggregate_mask = None  # Masks for all fold(s)
    for train, test, test_mask_chunk in iterator:
        # Here's where sampled not sliced comes in...
        sampled_skinny_train = make_skinny(train).sample(
            n=sample_size, random_state=seed
        )
        skinny_test = make_skinny(test)
        X = make_prior(
            sampled_skinny_train,
            skinny_test,
            method="CustomOSDAandZeoliteAsRows",
            normalization_factor=NORM_FACTOR,
        )
        all_data = pd.concat([sampled_skinny_train, skinny_test]).to_numpy()
        ##### SAFETY
        all_data[sampled_skinny_train.shape[0] :, :] = 0
        ##### SAFETY
        mask = np.ones_like(all_data)
        mask[len(sampled_skinny_train) :, :] = 0
        # The bottom 1/10 of mask and all_data are just all zeros.
        if use_eigenpro:
            # TODO: add a flag to return the prior with no identity concatted.
            # take the X with no eye
            # reduced_X = X[:, : X.shape[1] - X.shape[0]]
            # reduced_X = reduced_X / (max(reduced_X, key=sum).sum())
            x_train = X[: sampled_skinny_train.shape[0]].astype('float32')
            y_train = sampled_skinny_train.to_numpy().astype('float32')
            x_test = X[sampled_skinny_train.shape[0] :].astype('float32')
            y_test = skinny_test.to_numpy().astype('float32')
            # We'll probably need to chop X into test (# samples by dimensions), and train (# samples by dimensions)
            device = torch.device("cpu")
            kernel_fn = lambda x,y: gaussian(x, y, bandwidth=5)
            # model = FKR_EigenPro(kappa_with_clip, x_train, 289, device=device)
            model = FKR_EigenPro(kernel_fn, x_train, 289, device=device)
            _ = model.fit(x_train, y_train, x_test, y_test, epochs=[1, 2, 5], mem_gb=10)

        results_ntk = predict(all_data, mask, len(skinny_test), X=X)

        skinny_prediction_ntk = pd.DataFrame(
            data=results_ntk, index=skinny_test.index, columns=skinny_test.columns
        )
        prediction_ntk = unmake_skinny(skinny_prediction_ntk)
        aggregate_pred = (
            prediction_ntk
            if aggregate_pred is None
            else pd.concat([aggregate_pred, prediction_ntk])
        )
        aggregate_true = (
            pd.concat([aggregate_true, test]) if aggregate_true is not None else test
        )
        aggregate_mask = (
            pd.concat([aggregate_mask, test_mask_chunk])
            if aggregate_mask is not None
            else test_mask_chunk
        )
    metrics = calculate_metrics(
        aggregate_pred.to_numpy(), ground_truth.to_numpy(), aggregate_mask.to_numpy()
    )
    print(metrics)

    pdb.set_trace()
    print("blah")
