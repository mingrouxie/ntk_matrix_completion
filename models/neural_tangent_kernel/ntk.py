import sys
import pathlib
import os
import torch

from features.prior import make_prior
import pdb
import numpy as np
import pandas as pd
from auto_tqdm import tqdm
from enum import Enum, IntEnum

from ntk_matrix_completion.features.precompute_osda_priors import smile_to_property
from ntk_matrix_completion.features.prior import zeolite_prior
from ntk_matrix_completion.tests.ooc_matrix_multiplication import ooc_dot
from ntk_matrix_completion.models.neural_tangent_kernel.eigenpro.eigenpro import (
    FKR_EigenPro,
)
from ntk_matrix_completion.utils.package_matrix import (
    Energy_Type,
    get_ground_truth_energy_matrix,
    make_skinny,
    unmake_skinny,
)
from ntk_matrix_completion.utils.utilities import (
    get_splits_in_zeolite_type,
    save_matrix,
    chunks,
    get_isomer_chunks,
)
from ntk_matrix_completion.utils.analysis_utilities import (
    calculate_metrics,
    calculate_top_k_accuracy,
)
from ntk_matrix_completion.utils.path_constants import (
    HYPOTHETICAL_OSDA_ENERGIES,
    HYPOTHETICAL_OSDA_BOXES,
    OSDA_HYPOTHETICAL_PRIOR_FILE,
    OSDA_HYPOTHETICAL_PREDICTED_ENERGIES,
    ZEOLITE_HYPOTHETICAL_PREDICTED_ENERGIES,
    TEN_FOLD_CROSS_VALIDATION_ENERGIES,
    ZEO_1_PRIOR,
    OSDA_PRIOR_FILE,
)
from ntk_matrix_completion.utils.random_seeds import MODEL_SEED

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(
    1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical")
)

NORM_FACTOR = 0.001
PI = np.pi


class SplitType(IntEnum):
    NAIVE_SPLITS = 1
    ZEOLITE_SPLITS = 2
    OSDA_ISOMER_SPLITS = 3


def kappa(x):
    return (x * (PI - np.arccos(x)) + np.sqrt(1 - np.square(x))) / PI + (
        x * (PI - np.arccos(x))
    ) / PI


def kappa_with_clip(X_0, X_1, device):
    # torch.float32 TODO: make this torch.float32 like gaussian does....
    return torch.tensor(kappa(np.clip(X_0 @ X_1.T, -1, 1)), device=device)


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
    all_data = all_data.T  # columns are being filled in
    mask = mask.T
    num_observed = int(np.sum(mask[0:1, :]))
    num_missing = mask[0:1, :].shape[-1] - num_observed

    K_matrix = np.zeros((num_observed, num_observed))
    k_matrix = np.zeros((num_observed, num_missing))
    observed_data = all_data[:, :num_observed]
    observed_data = observed_data.astype("float64")
    X_cross_terms = kappa(np.clip(X @ X.T, -1, 1))  # kernel
    K_matrix[:, :] = X_cross_terms[:num_observed, :num_observed]
    k_matrix[:, :] = X_cross_terms[
        :num_observed, num_observed : num_observed + num_missing
    ]
    # plot_matrix(X_cross_terms, 'X_cross_terms', vmin=0, vmax=2)
    # plot_matrix(k_matrix, 'little_k', vmin=0, vmax=2)
    # plot_matrix(K_matrix, 'big_K', vmin=0, vmax=2)
    # plot_matrix(X, 'X', vmin=0, vmax=1)

    results = np.linalg.solve(K_matrix, observed_data.T).T @ k_matrix  # pg21 in paper
    assert results.shape == (all_data.shape[0], num_test_rows), "Results malformed"
    # breakpoint()

    assert np.any(np.isnan(results)) == False
    return results.T


def create_iterator(split_type, all_data, metrics_mask, k_folds, seed):
    """
    Inputs:

        split_type: method of constructing data splits
        all_data: for NTK this is a DataFrame of binding energies. For XGB this is a DataFrame of priors.
        For both the index of the DataFrame is SMILES
        metrics_mask: array with 1 for binding and 0 for non-binding entries
        k_folds: number of folds to create
        seed: seed for splits in zeolite_types

    Returns:

        train: portion of all_data for training
        test: portion of all_data for testing
        test_mask_chunk: binding/non-binding mask for test
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
            "[create_iterator] A bit silly, but we do require skinny_matrices to be perfectly modulo"
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
            )
        )
    else:
        raise Exception("[create_iterator] Need to provide a SplitType for run_ntk()")
    return iterator


def run_ntk(
    all_data,
    prior,
    metrics_mask,
    split_type=SplitType.NAIVE_SPLITS,
    k_folds=10,
    seed=MODEL_SEED,
    prior_map=None,
    norm_factor=NORM_FACTOR,
    use_eigenpro=False,
    osda_prior_file=OSDA_PRIOR_FILE,
):
    """
    From a given full matrix of binding energies, produces an identically-shaped matrix of 
    binding energies predicted with matrix completion. This matrix of predicted energies
    is created by splitting the matrix into filled and unfilled matrices, in a fashion similar
    to cross validation, completing the unfilled matrix, and aggregating all of the completed
    matrices into the full-sized matrix. 

    Note that the priors are created within the loop that iterates over different filled-unfilled
    splits. 

    Inputs:

    all_data: full matrix of binding energies
    prior (str): desired prior_type
    metrics_mask: array of same shape as all_data, containing 1 for binding and 0 for non-binding
    split_type: SplitType.<split_type> to indicate how the data is split
    k_folds: number of splits to make
    seed: integer for model reproducibility
    prior_map: dictionary containing weights for each prior (TODO: please confirm)
    norm_factor: 0.001 for identity weight (TODO: please confirm purpose)
    use_eigenpro: for future use when data size gets large
    osda_prior_file: OSDA prior file to read from. Only gets used when certain `prior` names are specified
    
    Returns:

    Three arrays or DataFrames of the same shape: Predictions, truth and mask (in this order)
    """
    iterator = create_iterator(split_type, all_data, metrics_mask, k_folds, seed)
    aggregate_pred = None  # Predictions for all fold(s)
    aggregate_true = None  # Ground truths for all fold(s)
    aggregate_mask = None  # Non-binding masks for all fold(s)
    # Iterate over all predictions and populate metrics matrices
    for train, test, test_mask_chunk in iterator:
        X = make_prior(
            train,
            test,
            method=prior,
            normalization_factor=norm_factor,
            prior_map=prior_map,
            osda_prior_file=osda_prior_file,
        )
        all_data = pd.concat([train, test]).to_numpy()
        ##### SAFETY
        all_data[train.shape[0] :, :] = 0
        ##### SAFETY
        mask = np.ones_like(all_data)
        mask[len(train) :, :] = 0
        # The bottom 1/10 of mask and all_data are just all zeros.

        assert np.any(np.isnan(X) == False)
        X = np.nan_to_num(X, copy=True, nan=0.0)
        # breakpoint()
        # TODO (Mingrou): Check with Yitong why the code worked before without this line
        # TODO: another fillna? Oh dear. Please extract all data preprocessing into one script
        # And, also, just have cleaner data files in general. Use percentile?
        # DEBUG 1092 and 1096. These look suspiciously familiar
        # {'C[C@H]1C[N+]2(CCCCC[N+]3(C)CCCCC3)CCC1CC2',
        # 'C[C@@H]1C[N+]2(CCCCC[N+]3(C)CCCCC3)CCC1CC2',
        # 'C1CN2CCC1CC2',
        # 'C[N+]1(CC2CCCCC2)CCC(CCCC2CC[N+](C)(CC3CCCCC3)CC2)CC1'}
        print(
            "[run_ntk] Predicting for data of shape",
            all_data.shape,
            mask.shape,
            X.shape,
        )
        print("[run_ntk] Test set of shape", test.shape)
        results_ntk = predict(all_data, mask, len(test), X=X)
        print("[run_ntk] results_ntk of shape", results_ntk.shape)
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
    print(
        "Returning data of shape",
        aggregate_pred.shape,
        aggregate_true.shape,
        aggregate_mask.shape,
    )
    return aggregate_pred, aggregate_true, aggregate_mask


def euclidean_distances(samples, centers, squared=True):
    """Calculate the pointwise distance.
    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        squared: boolean.
    Returns:
        pointwise distances (n_sample, n_center).
    """
    samples_norm = torch.sum(samples**2, dim=1, keepdim=True)
    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = torch.sum(centers**2, dim=1, keepdim=True)
    centers_norm = torch.reshape(centers_norm, (1, -1))

    distances = samples.mm(torch.t(centers))
    distances.mul_(-2)
    distances.add_(samples_norm)
    distances.add_(centers_norm)
    if not squared:
        distances.clamp_(min=0)
        distances.sqrt_()

    return distances


def gaussian(samples, centers, bandwidth):
    """Gaussian kernel.
    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.
    Returns:
        kernel matrix of shape (n_sample, n_center).
    """
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers)
    kernel_mat.clamp_(min=0)
    gamma = 1.0 / (2 * bandwidth**2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat


# TODO: Please please delete me before releasing this code.
# TODO: This is in development and pretty god awful.
# TODO: make sample_size bigger, maybe 10K?
def skinny_ntk_sampled_not_sliced(
    sample_size=1000, num_splits=10, seed=MODEL_SEED, use_eigenpro=True
):
    """
    This method runs 10-fold cross validation on the 1194x209 Ground Truth matrix made SKINNY
    But sampled not sliced from the top left corner (100, 30) style
    """
    ground_truth, binary_data = get_ground_truth_energy_matrix(desired_shape=(100, 30))
    iterator = tqdm(
        get_splits_in_zeolite_type(ground_truth, binary_data, k=num_splits, seed=seed),
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
            # We'll probably need to chop X into test (# samples by dimensions), and train (# samples by dimensions)
            device = torch.device("cpu")

            x_train = torch.tensor(
                X[: sampled_skinny_train.shape[0]].astype("float32"), device=device
            )
            y_train = torch.tensor(
                sampled_skinny_train.to_numpy().astype("float32"), device=device
            )
            x_test = torch.tensor(
                X[sampled_skinny_train.shape[0] :].astype("float32"), device=device
            )
            y_test = torch.tensor(
                skinny_test.to_numpy().astype("float32"), device=device
            )
            pdb.set_trace()
            kernel_fn = lambda x, y: kappa_with_clip(x, y, device)
            # model = FKR_EigenPro(kappa_with_clip, x_train, 1, device=device)
            # TODO: What's up with 289 here? looks like 'y_dim' = 298? that's very very arbitrary.
            model = FKR_EigenPro(kernel_fn, x_train, 1, device=device)
            _ = model.fit(x_train, y_train, x_test, y_test, epochs=[1, 2, 5], mem_gb=10)
            pdb.set_trace()
            print("he;o")

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
