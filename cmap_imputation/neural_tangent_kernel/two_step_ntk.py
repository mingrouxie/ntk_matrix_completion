from statistics import mode
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
from sklearn.metrics import top_k_accuracy_score


sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))
from utilities import train_test_w_controls, get_cosims, get_splits

sys.path.insert(
    1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical")
)
from singular_graphics import plot_graphics

NORM_FACTOR = 0.01  # IS this better than 0.1?
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
    # THIS IS CURSED. FEEL VERY ASHAMED
    # X_squared = (X @ X.T)
    # X_squared = X_squared / X_squared.max() #if normalize_intermediate else X_squared
    # X_cross_terms = kappa(np.clip(X_squared, -1, 1))
    X_cross_terms = kappa(np.clip(X @ X.T, -1, 1))
    K_matrix[:, :] = X_cross_terms[:num_observed, :num_observed]
    k_matrix[:, :] = X_cross_terms[
        :num_observed, num_observed : num_observed + num_missing
    ]
    # plot_matrix(X_cross_terms, 'X_cross_terms', vmin=0, vmax=2)
    # plot_matrix(k_matrix, 'little_k', vmin=0, vmax=2)
    # plot_matrix(K_matrix, 'big_K', vmin=0, vmax=2)
    # plot_matrix(X, 'X', vmin=0, vmax=1)
    # plot_matrix(X[0:100,0:100], 'close_up_X', vmin=0, vmax=1)
    results = np.linalg.solve(K_matrix, observed_data.T).T @ k_matrix
    assert results.shape == (all_data.shape[0], num_test_rows), "Results malformed"
    return results.T


def filter_res(true, pred, filter_value, average_by_rows):
    cosims = []
    r2_scores = []
    for row_id in range(true.shape[0]):
        i = true[row_id]
        j = pred[row_id]
        if average_by_rows:
            # If we want to cheat we can also use mode()
            filter_value = i.mean()
        j = j[i != filter_value]
        i = i[i != filter_value]
        if len(i) == 0:
            # TODO: this is wrong. come back and fix me.
            cosims.append([1.0])
            r2_scores.append([1.0])
            continue
        cosims.append(get_cosims(np.array([i]), np.array([j])))
        r2_scores.append(
            r2_score(np.array([i]).T, np.array([j]).T, multioutput="raw_values")
        )
    return cosims, r2_scores


# TODO: collapse run_ntk & run_ntk_binary_classification together...
def run_ntk(
    allData,
    only_train,
    method,
    SEED,
    path_prefix,
    plot,
    prior,
    fill_value,
    average_by_rows=False,
):
    # Filter not by 30 but by the average mean...
    if average_by_rows:
        allData[allData == fill_value] = None
        allData = allData.apply(lambda row: row.fillna(row.mean()), axis=1)
        # Let's filter out all the rows with all empty values...
        # e.g., these four: C1COCCOCCNCCOCCOCCN1, C1COCCOCCOCCN1, Nc1ccccc1, OCC(CO)(CO)CO
        allData = allData.dropna(thresh=1)
        allData = allData[allData.max(axis=1) != allData.min(axis=1)]
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
        X = make_prior(
            train, only_train, test, "CustomZeolite", normalization_factor=NORM_FACTOR #, feature = {'accessible_volume':1.0}
        )
        all_data = pd.concat([train, test]).to_numpy()
        # plot_matrix(all_data, "all_data_regression")
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
        # TODO: this filter is pretty brutal... do we want to continue applying it?
        cosims, r2_scores = filter_res(true, results_ntk, fill_value, average_by_rows)
        temp_df_ntk = pd.DataFrame(
            data=np.column_stack(
                [
                    r2_scores,
                    cosims
                    # r2_score(true.T, results_ntk.T, multioutput="raw_values"),
                    # get_cosims(true, results_ntk),
                ]
            ),
            index=test.index,
        )
        temp_df_ntk.columns = ["r_squared", "cosine_similarity"]
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

    # We want the lowest by zeolite... probably right?
    # UGH it makes no sense to test the lowest by zeolite. We want the lowest by OSDA...
    # Let's just do bof.
    top_1 = calculate_top_k_accuracy(all_true, ntk_predictions, 1)
    top_3 = calculate_top_k_accuracy(all_true, ntk_predictions, 3)
    top_5 = calculate_top_k_accuracy(all_true, ntk_predictions, 5)
    print(
        splits.mean(),
        "\ntop_1_accuracy: ",
        top_1.round(4),
        "\ntop_3_accuracy: ",
        top_3.round(4),
        "\ntop_5_accuracy: ",
        top_5.round(4),
    )
    # plot_matrix(ntk_predictions, "zeolites_justaccessible_volume")
    pdb.set_trace()
    if plot:
        plot_graphics(ntk_predictions, all_true, all_metrics_ntk, path_prefix, True)


def calculate_top_k_accuracy(all_true, ntk_predictions, k, by_row=True):
    if by_row:
        lowest_mask = (all_true.T == all_true.min(axis=1)).T
        _col_nums, top_indices = np.where(lowest_mask)
        pred = -ntk_predictions.to_numpy()
    else:
        lowest_mask = all_true == all_true.min(axis=0)
        _col_nums, top_indices = np.where(lowest_mask)
        pred = -ntk_predictions.to_numpy().T
    return top_k_accuracy_score(top_indices, pred, k=k, labels=range(pred.shape[1]))


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
    prior=None,
    # TODO: get rid of this argument and just use the fed in method argument.
    feature=None,  # "CustomOSDA",
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
        all_data = pd.concat([train, test]).to_numpy()
        ##### SAFETY
        all_data[train.shape[0] :, :] = 0
        ##### SAFETY
        mask = np.ones_like(all_data)
        mask[len(train) :, :] = 0

        # TODO: take this method declaration somewhere else.
        X = make_prior(
            train,
            only_train,
            test,
            method="CustomOSDAandZeoliteAsRows",  # "CustomOSDA",
            normalization_factor=NORM_FACTOR,
            feature=feature,
        )
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
    total = (
        sum(splits.false_positive)
        + sum(splits.false_negative)
        + sum(splits.true_positive)
        + sum(splits.true_negative)
    )
    total_accuracy = (
        1.0 * sum(splits.true_positive) + sum(splits.true_negative)
    ) / total
    if sum(splits.false_positive) + sum(splits.true_positive) == 0:
        precision = 0
    else:
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
    # plot_matrix(all_true, "binary_classification_truth", vmin=0, vmax=1)
    # plot_matrix(ntk_predictions, "binary_classification_prediction", vmin=0, vmax=1)
    # return ntk_predictions

    # plot_matrix(all_true, "binary_classification_truth_" + feature, vmin=0, vmax=1)
    # plot_matrix(
    #     ntk_predictions,
    #     "binary_classification_prediction_" + feature,
    #     vmin=0,
    #     vmax=1,
    # )
    return (
        "total accuracy: ",
        round(total_accuracy, 3),
        " total precision: ",
        round(precision, 3),
        " total recall: ",
        round(recall, 3)
    )


def test_a_bunch_of_features(
    binaryData,
    only_train,
    method,
    SEED,
    path_prefix,
    plot,
):
    features_to_test = [
        "a",
        "b",
        "c",
        "alpha",
        "betta",
        "gamma",
        "volume",
        "rdls",
        "framework_density",
        "td_10",
        "td",
        "included_sphere_diameter",
        "diffused_sphere_diameter_a",
        "diffused_sphere_diameter_b",
        "diffused_sphere_diameter_c",
        "accessible_volume",
    ]
    results = {}
    for feature in features_to_test:
        results[feature] = run_ntk_binary_classification(
            binaryData,
            only_train,
            method,
            SEED,
            path_prefix,
            plot,
            feature={feature: 1.0},
        )
    print(results)


if __name__ == "__main__":
    print(
        "Modify make_prior in prior.py to add a custom prior! There are a few choices to start."
    )
    # TODO: change this... make this tkae an argument from the input for col_names.
    # TODO: add a mask? then
    # 112426

    (
        allData,
        binaryData,
        only_train,
        method,
        SEED,
        path_prefix,
        plot,
        prior,
    ) = validate_zeolite_inputs(col_name="SMILES")
    # allData = allData.T
    # binaryData = binaryData.T
    # run_ntk_binary_classification_from_two_orientations
    # TODO: Be very careful with this prior...
    # don't let the legacy code add it in some weird way..
    # binaryData = binaryData.iloc[:100, :30]
    # test_a_bunch_of_features(
    #     binaryData,
    #     only_train,
    #     method,
    #     SEED,
    #     path_prefix,
    #     plot,
    # )
    run_ntk_binary_classification(
        binaryData,
        only_train,
        method,
        SEED,
        path_prefix,
        plot,
    )
    # Let's take out rows that have just no templating energies at all...
    # not even sure how they got into the dataset... Worth investigating...
    # e.g., these four: C1COCCOCCNCCOCCOCCN1, C1COCCOCCOCCN1, Nc1ccccc1, OCC(CO)(CO)CO
    allData = allData[allData.max(axis=1) != allData.min(axis=1)]
    run_ntk(
        allData,
        only_train,
        method,
        SEED,
        path_prefix,
        plot,
        prior="CustomOSDA",
        fill_value=30,
        average_by_rows=True,
    )
    # when we turn it sideways these two zeolites have all the same lowest energies...
    # SMILES   C(CC1CCNCC1)CC1CCNCC1  ...   c1ccncc1
    # Zeolite                         ...
    # PAU                   9.708801  ...   9.708801
    # LOV                  11.821684  ...  11.821684

    # python neural_tangent_kernel/two_step_ntk.py -i /Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/zeoliteOSDAIndexedMatrix.pkl -i2 /Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/smaller_skinny_matrix.pkl -r -o
