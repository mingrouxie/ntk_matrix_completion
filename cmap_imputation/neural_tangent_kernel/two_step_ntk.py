import sys
import pathlib
import os
from prior import make_prior
from cli import validate_zeolite_inputs

import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb
import numpy as np
import pandas as pd
from auto_tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))
from utilities import train_test_w_controls, get_cosims, get_splits

sys.path.insert(
    1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical")
)
from singular_graphics import plot_graphics

NORM_FACTOR = 0.1 # IS this better than 0.1?
PI = np.pi

# is this kappa something to be tuned?
# can't this be reduced????
def kappa(x):
    return (x * (PI - np.arccos(x)) + np.sqrt(1 - np.square(x))) / PI + (
        x * (PI - np.arccos(x))
    ) / PI


def predict_space_opt_CMAP_data(all_data, mask, num_test_rows, X):
    """
    Space optimized version for CMap Data
    """
    all_data = all_data.T
    mask = mask.T

    # OKay weird... num_observed is rotated.... so this is actually holding out OSDAs, not zeolites...
    num_observed = int(np.sum(mask[0:1, :]))
    num_missing = mask[0:1, :].shape[-1] - num_observed

    K_matrix = np.zeros((num_observed, num_observed))
    k_matrix = np.zeros((num_observed, num_missing))

    observed_data = all_data[:, :num_observed]
    # this might not be kosher...
    # TODO: this might break the whole ntk assurance
    # X_squared = (X @ X.T)
    # X_squared = X_squared / X_squared.max() if normalize_intermediate else X_squared
    X_cross_terms = kappa(np.clip(X @ X.T, -1, 1))
    K_matrix[:, :] = X_cross_terms[:num_observed, :num_observed]
    k_matrix[:, :] = X_cross_terms[
        :num_observed, num_observed : num_observed + num_missing
    ]
    # plot_matrix(X_cross_terms, 'X_cross_terms', vmin=0, vmax=2)
    # plot_matrix(X[0:100,0:100], 'close_up_prior_with_zeolite_diameter', vmin=0, vmax=1)
    results = np.linalg.solve(K_matrix, observed_data.T).T @ k_matrix
    pdb.set_trace()
    assert results.shape == (all_data.shape[0], num_test_rows), "Results malformed"
    # pdb.set_trace()
    return results.T


# TODO: collapse run_ntk & run_ntk_binary_classification together...
def run_ntk(
    allData,
    only_train,
    method,
    SEED,
    path_prefix,
    plot,
    prior,
):
    path_prefix += f"{prior}Prior"
    if method[0] == "kfold":
        iterator = tqdm(get_splits(allData, k=method[1], seed=SEED), total=method[1])
    elif method[0] == "sparse":
        iterator = tqdm(
            [train_test_w_controls(allData, drugs_in_train=method[1], seed=SEED)],
            total=1,
        )
    else:
        raise AssertionError("Unknown method")

    all_true = None  # Ground truths for all fold(s)
    all_metrics_ntk = None  # Metrics (R^2, Cosine Similarity, Pearson R) for each entry
    ntk_predictions = None  # Predictions for all fold(s)
    splits = None  # Per-fold metrics
    # Iterate over all predictions and populate metrics matrices
    for train, test in iterator:
        X = make_prior(train, only_train, test, prior, normalization_factor=NORM_FACTOR)
        pdb.set_trace()
        all_data = pd.concat([train, test]).to_numpy()
        ##### SAFETY
        all_data[train.shape[0] :, :] = 0
        ##### SAFETY
        mask = np.ones_like(all_data)
        mask[len(train) :, :] = 0

        # okay so the bottom 1/10 of mask and all_data are just all zeros.
        results_ntk = predict_space_opt_CMAP_data(all_data, mask, len(test), X=X)

        prediction_ntk = pd.DataFrame(
            data=results_ntk, index=test.index, columns=test.columns
        )
        ntk_predictions = (
            prediction_ntk
            if ntk_predictions is None
            else pd.concat([ntk_predictions, prediction_ntk])
        )

        true = test.to_numpy()
        # temp_df_ntk is the r2 score in the first column and cosine similarity in the second...
        temp_df_ntk = pd.DataFrame(
            data=np.column_stack(
                [
                    r2_score(true.T, results_ntk.T, multioutput="raw_values"),
                    get_cosims(true, results_ntk),
                ]
            ),
            index=test.index,
        )
        temp_df_ntk.columns = ["r_squared", "cosine_similarity"]
        print(temp_df_ntk)
        print(train.shape, test.shape)
        new_splits = pd.DataFrame(
            data=np.column_stack(
                [
                    temp_df_ntk["r_squared"].mean(),
                    temp_df_ntk["cosine_similarity"].mean(),
                    len(train),
                    len(test),
                ]
            )
        )
        new_splits.columns = [
            "r_squared",
            "cosine_similarity",
            "train_size",
            "test_size",
        ]

        splits = (
            new_splits
            if splits is None
            else pd.concat([splits, new_splits], ignore_index=True)
        )
        all_metrics_ntk = (
            temp_df_ntk
            if all_metrics_ntk is None
            else pd.concat([all_metrics_ntk, temp_df_ntk])
        )
        all_true = pd.concat([all_true, test]) if all_true is not None else test

    r_ntk, p_ntk = pearsonr(
        all_true.to_numpy().ravel(), ntk_predictions.to_numpy().ravel()
    )
    all_metrics_ntk["pearson_r"] = r_ntk
    all_metrics_ntk["pearson_r_p_value"] = p_ntk

    pd.to_pickle(all_true, path_prefix + "GroundTruth.pkl")
    pd.to_pickle(ntk_predictions, path_prefix + "Predictions.pkl")
    pd.to_pickle(all_metrics_ntk, path_prefix + "AllMetrics.pkl")
    pd.to_pickle(splits, path_prefix + "SplitMetrics.pkl")

    if plot:
        plot_graphics(ntk_predictions, all_true, all_metrics_ntk, path_prefix, True)


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


def num_correct(true, results):
    true - results


def run_ntk_binary_classification(
    allData,
    only_train,
    method,
    SEED,
    path_prefix,
    plot,
    prior,
):
    path_prefix += f"{prior}Prior"
    if method[0] == "kfold":
        iterator = tqdm(get_splits(allData, k=method[1], seed=SEED), total=method[1])
    elif method[0] == "sparse":
        iterator = tqdm(
            [train_test_w_controls(allData, drugs_in_train=method[1], seed=SEED)],
            total=1,
        )
    else:
        raise AssertionError("Unknown method")

    all_true = None  # Ground truths for all fold(s)
    all_metrics_ntk = None  # Metrics (R^2, Cosine Similarity, Pearson R) for each entry
    ntk_predictions = None  # Predictions for all fold(s)
    splits = None  # Per-fold metrics
    # Iterate over all predictions and populate metrics matrices
    for train, test in iterator:
        # TODO: take this method declaration somewhere else.
        X = make_prior(train, only_train, test, method="CustomZeolite", normalization_factor=NORM_FACTOR)

        all_data = pd.concat([train, test]).to_numpy()
        ##### SAFETY
        all_data[train.shape[0] :, :] = 0
        ##### SAFETY
        mask = np.ones_like(all_data)
        mask[len(train) :, :] = 0

        # okay so the bottom 1/10 of mask and all_data are just all zeros.
        # TODO: take out this normalize intermediate
        results_ntk = predict_space_opt_CMAP_data(all_data, mask, len(test), X=X)
        # TADA! binary classification in full swing :^)
        results_ntk = results_ntk.round()

        prediction_ntk = pd.DataFrame(
            data=results_ntk, index=test.index, columns=test.columns
        )
        # okay ntk_predictions is actuallly useful. I admit it.
        ntk_predictions = (
            prediction_ntk
            if ntk_predictions is None
            else pd.concat([ntk_predictions, prediction_ntk])
        )

        true = test.to_numpy()

        correct = results_ntk[results_ntk == true]
        incorrect = results_ntk[results_ntk != true]
        new_splits = pd.DataFrame(
            data=np.column_stack(
                [
                    len(correct[correct == 1]),
                    len(correct[correct == 0]),
                    # the prediction was 1 but true was 0
                    len(incorrect[incorrect == 1]),
                    # the prediction was 0 but true was 1
                    len(incorrect[incorrect == 0]),
                    len(train),
                    len(test),
                ]
            )
        )
        new_splits.columns = [
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
            "train_size",
            "test_size",
        ]
        splits = (
            new_splits
            if splits is None
            else pd.concat([splits, new_splits], ignore_index=True)
        )
        all_true = pd.concat([all_true, test]) if all_true is not None else test
    total_accuracy = (1.0 * sum(splits.true_positive) + sum(splits.true_negative)) / (
        sum(splits.false_positive)
        + sum(splits.false_negative)
        + sum(splits.true_positive)
        + sum(splits.true_negative)
    )
    precision = (1.0 * sum(splits.true_positive)) / (
        sum(splits.false_positive) + sum(splits.true_positive)
    )
    recall = (1.0 * sum(splits.true_positive)) / (
        sum(splits.true_positive) + sum(splits.false_negative)
    )
    print(
        "total accuracy: ",
        total_accuracy,
        " total precision: ",
        precision,
        " total recall: ",
        recall,
    )
    plot_matrix(all_true, "binary_classification_truth", vmin=0, vmax=1)
    plot_matrix(ntk_predictions, "binary_classification_prediction", vmin=0, vmax=1)


if __name__ == "__main__":
    print(
        "Modify make_prior in prior.py to add a custom prior! There are a few choices to start."
    )
    # TODO: change this... make this tkae an argument from the input for col_names.
    # TODO: add a mask? then

    (
        allData,
        secondaryData,
        only_train,
        method,
        SEED,
        path_prefix,
        plot,
        prior,
    ) = validate_zeolite_inputs(col_name="SMILES")
    # pdb.set_trace()
    allData = allData.T
    secondaryData = secondaryData.T
    run_ntk_binary_classification(
        secondaryData,
        only_train,
        method,
        SEED,
        path_prefix,
        plot,
        prior,
    )
    # run_ntk(
    #     allData,
    #     only_train,
    #     method,
    #     SEED,
    #     path_prefix,
    #     plot,
    #     prior,
    #     binary_classification=False,
    # )
