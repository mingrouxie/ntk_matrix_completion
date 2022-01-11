from statistics import mode
import sys
import pathlib
import os
from prior import make_prior
from cli import validate_zeolite_inputs

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb
import numpy as np
import pandas as pd
from auto_tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import top_k_accuracy_score
import time


sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))
from utilities import train_test_w_controls, get_cosims, get_splits

sys.path.insert(
    1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical")
)
from singular_graphics import plot_graphics

NORM_FACTOR = 0.001  # IS this better than 0.1?
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
    # plot_matrix(X[0:50,0:50], 'close_up_X', vmin=0, vmax=0.05)
    results = np.linalg.solve(K_matrix, observed_data.T).T @ k_matrix
    assert results.shape == (all_data.shape[0], num_test_rows), "Results malformed"
    return results.T


# (30, 30, 12, 30, 30)
# (29.9, 29.9, 11, 29.9, 29.9,)

# (12, )
# (11,)
# Filter out all the non-binding results when calculating cosim & r2
def filter_res(true, pred, filter_value, average_by_rows):
    cosims = []
    r2_scores = []
    rmse_scores = []
    spearman_scores = []
    for row_id in range(true.shape[0]):
        i = true[row_id]
        j = pred[row_id]
        if average_by_rows:
            # If we want to cheat we can also use mode()
            filter_value = i.mean()
        j = j[i != filter_value]
        i = i[i != filter_value]
        if len(i) == 0:
            raise ValueError(
                "There exists complete rows in your dataset which are completely unbinding."
            )
        cosims.append(get_cosims(np.array([i]), np.array([j])))
        r2_scores.append(r2_score(i, j, multioutput="raw_values"))
        rmse_scores.append(
            [math.sqrt(mean_squared_error(i, j, multioutput="raw_values"))]
        )
        spearman_scores.append([spearmanr(i, j).correlation])
    return cosims, r2_scores, rmse_scores, spearman_scores


# TODO: Are these the more accurate results to report?
def res_without_filter(true, pred):
    cosims = []
    r2_scores = []
    rmse_scores = []
    spearman_scores = []
    for row_id in range(true.shape[0]):
        i = true[row_id]
        j = pred[row_id]
        cosims.append(get_cosims(np.array([i]), np.array([j])))
        r2_scores.append(r2_score(i, j, multioutput="raw_values"))
        rmse_scores.append(
            [math.sqrt(mean_squared_error(i, j, multioutput="raw_values"))]
        )
        spearman_scores.append([spearmanr(i, j).correlation])
    return cosims, r2_scores, rmse_scores, spearman_scores


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
    skinny=False,
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
            train,
            only_train,
            test,
            prior,
            normalization_factor=NORM_FACTOR,  # , feature = {'accessible_volume':1.0}
        )
        all_data = pd.concat([train, test]).to_numpy()
        ##### SAFETY
        all_data[train.shape[0] :, :] = 0
        ##### SAFETY
        mask = np.ones_like(all_data)
        mask[len(train) :, :] = 0

        # okay so the bottom 1/10 of mask and all_data are just all zeros.
        results_ntk = predict_space_opt_CMAP_data(all_data, mask, len(test), X=X)

        if skinny:
            results_ntk, true = unmake_skinny(results_ntk, test)
        else:
            true = test.to_numpy()

        prediction_ntk = pd.DataFrame(
            data=results_ntk, index=test.index, columns=test.columns
        )
        ntk_predictions = (
            prediction_ntk
            if ntk_predictions is None
            else pd.concat([ntk_predictions, prediction_ntk])
        )

        # TODO: this filter is pretty brutal... do we want to continue applying it?
        cosims, r2_scores, rmse_scores, spearman_scores = filter_res(
            true, results_ntk, fill_value, average_by_rows
        )
        temp_df_ntk = pd.DataFrame(
            data=np.column_stack([r2_scores, cosims, rmse_scores, spearman_scores]),
            index=test.index,
        )
        temp_df_ntk.columns = [
            "r_squared",
            "cosine_similarity",
            "rmse",
            "spearman_correlation",
        ]
        new_splits = pd.DataFrame(
            data=np.column_stack(
                [
                    temp_df_ntk["r_squared"].mean(),
                    temp_df_ntk["cosine_similarity"].mean(),
                    temp_df_ntk["rmse"].mean(),
                    temp_df_ntk["spearman_correlation"].mean(),
                    len(train),
                    len(test),
                ]
            )
        )
        new_splits.columns = [
            "r_squared",
            "cosine_similarity",
            "rmse",
            "spearman_correlation",
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

    # NOTE: this is top_k_accuracy as split by ROWS
    # Rows in the original data are OSDAs, if the data is transposed then the rows will be Zeolites.
    # Probably the top_k that makes the most sense is calculating it by zeolites (not by OSDAs)
    top_1 = calculate_top_k_accuracy(all_true, ntk_predictions, 1)
    top_3 = calculate_top_k_accuracy(all_true, ntk_predictions, 3)
    top_5 = calculate_top_k_accuracy(all_true, ntk_predictions, 5)
    top_20_accuracies = [
        calculate_top_k_accuracy(all_true, ntk_predictions, k) for k in range(0, 21)
    ]
    plot_top_k_curves(top_20_accuracies)
    pdb.set_trace()
    print(
        splits.mean(),
        "\ntop_1_accuracy: ",
        top_1.round(4),
        "\ntop_3_accuracy: ",
        top_3.round(4),
        "\ntop_5_accuracy: ",
        top_5.round(4),
    )
    plot_matrix(all_true, "regression_truth")
    plot_matrix(ntk_predictions, "regression_prediction")
    return ntk_predictions


def plot_top_k_curves(top_accuracies):
    plt.plot(top_accuracies)
    plt.title("Top K Accuracy for Zeolites per OSDA")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(0, len(top_accuracies) + 1, step=5))
    plt.show()
    plt.draw()
    plt.savefig("top_k_accuracies.png", dpi=100)


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

        X = make_prior(
            train,
            only_train,
            test,
            method=prior,
            normalization_factor=NORM_FACTOR,
            feature=feature,
        )
        # okay so the bottom 1/10 of mask and all_data are just all zeros.
        results_ntk = predict_space_opt_CMAP_data(all_data, mask, len(test), X=X)
        # TADAA! binary classification in full swing :^)
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
    plot_matrix(all_true, "binary_classification_truth", vmin=0, vmax=1)
    plot_matrix(ntk_predictions, "binary_classification_prediction", vmin=0, vmax=1)
    return ntk_predictions


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


def make_skinny(allData, col_1="variable", col_2="SMILES"):
    allData = allData.reset_index()
    melted_matrix = pd.melt(
        allData, id_vars=col_2, value_vars=list(allData.columns[1:])
    )
    return melted_matrix.set_index([col_2, col_1])


def unmake_skinny(skinnyPrediction, skinnyTrue):
    true_zeolites_per_osda = {}
    predicted_zeolites_per_osda = {}
    for index in range(len(skinnyTrue)):
        true_value = skinnyTrue.iloc[index][0]
        osda, zeolite = skinnyTrue.iloc[index].name
        pred_value = skinnyPrediction[index][0]

        if osda not in true_zeolites_per_osda:
            true_zeolites_per_osda[osda] = {}
        else:
            pdb.set_trace()
        true_zeolites_per_osda[osda][zeolite] = true_value

        if osda not in predicted_zeolites_per_osda:
            predicted_zeolites_per_osda[osda] = {}
        predicted_zeolites_per_osda[osda][zeolite] = pred_value
    predicted_zeolites_per_osda
    return skinnyPrediction, skinnyTrue


def skinny_ntk():
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
    allData = allData[allData.max(axis=1) != allData.min(axis=1)]
    allData = allData.iloc[:100, :30]
    allData = make_skinny(allData, col_1="Zeolite")

    binaryData = binaryData[binaryData.max(axis=1) != binaryData.min(axis=1)]
    binaryData = binaryData.iloc[:200, :100]
    binaryData = make_skinny(binaryData, col_1="Zeolite")
    # TODO: for regression on skinny matrix... see how to average_by_rows...
    # run_ntk(
    #     allData,
    #     only_train,
    #     method,
    #     SEED,
    #     path_prefix,
    #     plot,
    #     prior="CustomOSDAandZeoliteAsRows",
    #     fill_value=30,
    #     average_by_rows=False,
    #     skinny=True,
    # )
    run_ntk_binary_classification(
        binaryData,
        only_train,
        method,
        SEED,
        path_prefix,
        plot,
        prior="CustomOSDAandZeoliteAsRows",
    )


def save_matrix(matrix, file_name):
    file = os.path.abspath("")
    dir_main = pathlib.Path(file).parent.absolute()
    savepath = os.path.join(dir_main, file_name)
    matrix.to_pickle(savepath)


def test_all_osdas():
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
    precomputed_energies = pd.read_pickle(
        "/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/data_from_daniels_ml_models/precomputed_energies_78616by196.pkl"
    )
    precomputed_priors = pd.read_pickle(
        "/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/data_from_daniels_ml_models/precomputed_energies_78616by196WithWhims.pkl"
    )
    precomputed_energies = precomputed_energies.reindex(precomputed_priors.index)
    # Set non-bindings to the row mean...
    fill_value = 30
    allData[allData == fill_value] = None
    allData = allData.apply(lambda row: row.fillna(row.mean()), axis=1)
    # Let's filter out all the rows with all empty values...
    # e.g., these four: C1COCCOCCNCCOCCOCCN1, C1COCCOCCOCCN1, Nc1ccccc1, OCC(CO)(CO)CO
    allData = allData.dropna(thresh=1)
    allData = allData[allData.max(axis=1) != allData.min(axis=1)]
    # 1190 rows x 209 columns

    truth = precomputed_energies.to_numpy()

    allData = pd.concat([allData, precomputed_energies])
    # We need to dedup indices so we're not testing on training samples. (only 2 overlap??? crazy)
    allData = allData[~allData.index.duplicated(keep="first")]
    # This 10K is a limitation we've got to get around with matrix multiplication and out of memory...
    allData = allData.iloc[: (1190 + 10000), :]

    # 79536 rows x 209 columns
    # So now the question becomes, do we do the binary sweep first? My guess is yes.
    # and after the binary sweep we can zero everything to some middle value...

    X = make_prior(
        None,
        None,
        None,
        method="CustomOSDA",
        normalization_factor=NORM_FACTOR,
        all_data=allData,
    )

    allData = allData.to_numpy()
    num_test_rows = 10000  # 78346
    mask = np.ones_like(allData)
    mask[len(allData) - num_test_rows :, :] = 0

    start = time.time()
    results = predict_space_opt_CMAP_data(allData, mask, num_test_rows, X)
    end = time.time()
    print("total time: ", end - start)
    pdb.set_trace()
    # TODO: test this shit...
    plot_matrix(results, "results", vmin=0, vmax=1)
    plot_matrix(truth, "truth", vmin=0, vmax=1)


# TODO: try some stuff.
def calculate_results_with_masks(true, pred):
    cosims = []
    r2_scores = []
    rmse_scores = []
    spearman_scores = []
    for row_id in range(true.shape[0]):
        i = true[row_id]
        j = pred[row_id]

        # I still don't know if this is koshure, to fill in all the nan results with the mean like this...
        i = np.nan_to_num(i, nan=np.mean(i[~np.isnan(i)]))
        j = np.nan_to_num(j, nan=np.mean(j[~np.isnan(j)]))

        cosims.append(get_cosims(np.array([i]), np.array([j])))
        r2_scores.append(r2_score(i, j, multioutput="raw_values"))
        rmse_scores.append(
            [math.sqrt(mean_squared_error(i, j, multioutput="raw_values"))]
        )
        spearman_scores.append([spearmanr(i, j).correlation])
    return cosims, r2_scores, rmse_scores, spearman_scores


def crazy_stuff():
    allData = pd.read_pickle(
        "/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/templating_truth.pkl"
    )
    mask_truth = pd.read_pickle(
        "/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/mask_truth.pkl"
    )
    templating_predictions = pd.read_pickle(
        "/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/templating_pred.pkl"
    )
    mask_predictions = pd.read_pickle(
        "/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/mask_pred.pkl"
    )

    # # I can kinda already tell this binary prediction task is not going to help... unless we weigh the actual training differently...

    # cosims, r2_scores, rmse_scores, spearman_scores = calculate_results_with_masks(
    #     allData.to_numpy(),
    #     templating_predictions.mask(np.logical_not(mask_predictions)).to_numpy(),
    # )

    # combined_predictions = np.nan_to_num(
    #     templating_predictions.mask(np.logical_not(mask_predictions)).to_numpy(),
    #     nan=100,
    # )
    # results = np.nan_to_num(
    #     allData.to_numpy(),
    #     nan=100,
    # )
    # pdb.set_trace()
    # top_1 = calculate_top_k_accuracy(results, templating_predictions.to_numpy(), 1)
    # top_3 = calculate_top_k_accuracy(results, templating_predictions.to_numpy(), 3)
    # top_5 = calculate_top_k_accuracy(results, templating_predictions.to_numpy(), 5)

    # pdb.set_trace()
    # print("howdy yitong")
    # print(
    #     "\ntop_1_accuracy: ",
    #     top_1.round(4),
    #     "\ntop_3_accuracy: ",
    #     top_3.round(4),
    #     "\ntop_5_accuracy: ",
    #     top_5.round(4),
    # )

    # # def calculate_top_k_accuracy(all_true, ntk_predictions, k, by_row=True):
    # #     if by_row:
    # #         lowest_mask = (allData.T == allData.min(axis=1)).T
    # #         _col_nums, top_indices = np.where(lowest_mask)
    # #         pred = -templating_predictions.to_numpy()
    # #     else:
    # #         lowest_mask = all_true == all_true.min(axis=0)
    # #         _col_nums, top_indices = np.where(lowest_mask)
    # #         pred = -ntk_predictions.to_numpy().T
    # #     return top_k_accuracy_score(top_indices, pred, k=k, labels=range(pred.shape[1]))

    # # allData = pd.read_pickle('/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/templating_truth.pkl')
    # # mask_truth = pd.read_pickle('/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/mask_truth.pkl')
    # # templating_predictions = pd.read_pickle('/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/templating_pred.pkl')

    plot_two_matrices(
        allData.to_numpy(),
        "Ground Truth",
        templating_predictions.to_numpy(),
        "Predicted",
        "combined_figure_2",
        mask_truth,
    )



def buisness_as_normal():
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
    # TODO(Mingrou): For the new zeolite you'll want to take the transpose of allData & binaryData
    # TODO(Mingrou): You'll also want to set prior="CustomZeolite"
    # allData = allData.T
    # binaryData = binaryData.T
    binaryData = binaryData[binaryData.max(axis=1) != binaryData.min(axis=1)]
    # binary_predictions = run_ntk_binary_classification(
    #     binaryData,
    #     only_train,
    #     method,
    #     SEED,
    #     path_prefix,
    #     plot,
    #     prior="CustomOSDAVector",
    # )
    # Let's take out rows that have just no templating energies at all...
    # not even sure how they got into the dataset... Worth investigating...
    # e.g., these four: C1COCCOCCNCCOCCOCCN1, C1COCCOCCOCCN1, Nc1ccccc1, OCC(CO)(CO)CO
    allData = allData[allData.max(axis=1) != allData.min(axis=1)]
    templating_predictions = run_ntk(
        allData,
        only_train,
        method,
        SEED,
        path_prefix,
        plot,
        prior="CustomOSDAVector",
        fill_value=30,
        average_by_rows=True,
    )

    plot_matrix(
        templating_predictions.to_numpy(),
        "templating_predictions",
        binaryData.to_numpy(),
    )
    plot_matrix(allData.to_numpy(), "combined_truth", binaryData.to_numpy())

    pdb.set_trace()

    save_matrix(templating_predictions, "templating_pred.pkl")
    # save_matrix(binary_predictions, "mask_pred.pkl")
    save_matrix(allData, "templating_truth.pkl")
    save_matrix(binaryData, "mask_truth.pkl")

    print("hello yitong! howzit look all combined?")
    
if __name__ == "__main__":
    buisness_as_normal()
