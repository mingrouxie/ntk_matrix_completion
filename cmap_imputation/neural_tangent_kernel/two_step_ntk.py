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
from precompute_prior import smile_to_property


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


# TODO: Move the processing of the data (replacing non-binding with row mean) outside of run_ntk()
# TODO: perhaps move the analysis (spearman, etc) after the function as well.
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
    plot_matrix(all_true, "regression_truth", vmin=-30, vmax=5)
    plot_matrix(ntk_predictions, "regression_prediction", vmin=-30, vmax=5)
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
    # TODO: average this by rows...

    binaryData = make_skinny(binaryData, col_1="Zeolite")
    # TODO: for regression on skinny matrix... see how to average_by_rows...
    # TODO: rather than make things weird. just pass in a mask for what to use in calculations.
    run_ntk(
        allData,
        only_train,
        method,
        SEED,
        path_prefix,
        plot,
        prior="CustomOSDAandZeoliteAsRows",
        fill_value=30,
        average_by_rows=False,
        skinny=True,
    )


def save_matrix(matrix, file_name):
    file = os.path.abspath("")
    dir_main = pathlib.Path(file).parent.absolute()
    savepath = os.path.join(dir_main, file_name)
    matrix.to_pickle(savepath)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def test_all_osdas():
    # (
    #     training_data,
    #     binaryData,
    #     only_train,
    #     method,
    #     SEED,
    #     path_prefix,
    #     plot,
    #     prior,
    # ) = validate_zeolite_inputs(col_name="SMILES")
    # First thing's first, let's process the training data
    # Set non-bindings to the row mean...
    # fill_value = 30
    # training_data[training_data == fill_value] = None
    # training_data = training_data.apply(lambda row: row.fillna(row.mean()), axis=1)
    # # Let's filter out all the rows with all empty values...
    # # e.g., these four: C1COCCOCCNCCOCCOCCN1, C1COCCOCCOCCN1, Nc1ccccc1, OCC(CO)(CO)CO
    # training_data = training_data.dropna(thresh=1)
    # training_data = training_data[
    #     training_data.max(axis=1) != training_data.min(axis=1)
    # ]

    binding_data = pd.read_pickle("data/BindingSiO2.pkl")
    binding_data = binding_data.apply(lambda row: row.fillna(row.mean()), axis=1)
    # Let's filter out all the rows with all empty values...
    # e.g., these four: C1COCCOCCNCCOCCOCCN1, C1COCCOCCOCCN1, Nc1ccccc1, OCC(CO)(CO)CO
    binding_data = binding_data.dropna(thresh=1)
    training_data = binding_data[binding_data.max(axis=1) != binding_data.min(axis=1)]
    pdb.set_trace()
    # training_data ends up as 1190 rows x 209 columns

    # These are predictions according to Daniel's per Zeolite ML model for 78K OSDAs
    daniel_energies = pd.read_pickle(
        "/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/data_from_daniels_ml_models/precomputed_energies_78616by196.pkl"
    )
    # These are precomputed priors for those 78K OSDAs
    precomputed_priors = pd.read_pickle(
        "/Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/data_from_daniels_ml_models/precomputed_energies_78616by196WithWhims.pkl"
    )
    daniel_energies = daniel_energies.reindex(precomputed_priors.index)
    truth = daniel_energies.to_numpy()
    # We need to dedup indices so we're not testing on training samples. (only 2 overlap??? crazy)
    daniel_energies = daniel_energies.drop(
        set.intersection(set(training_data.index), set(daniel_energies.index))
    )

    # Chunking by 10K to get around how bad matrix multiplication scales with memory...
    # This won't actually make our results any less accurate since we're only training with the 1190x209 samples in training_data
    # and just extending the results to the 10K next samples.
    chunk_size = 10000
    iterator = tqdm(chunks(range(len(daniel_energies.index)), chunk_size))
    predicted_energies = pd.DataFrame()
    for chunk in iterator:
        daniel_energies_chunk = daniel_energies.iloc[chunk, :]
        allData = pd.concat([training_data, daniel_energies_chunk])
        X = make_prior(
            None,
            None,
            None,
            method="CustomOSDA",
            normalization_factor=NORM_FACTOR,
            all_data=allData,
        )

        # So now the question becomes, do we do the binary sweep first? My guess is yes.
        # and after the binary sweep we can zero everything to some middle value...
        # bleh binary sweep doesn't work well :(
        # Remember that we set all non-binding energies to the row mean so predicted templating energies
        # that are close to row means might be way off.
        # Only the really low values & really high values are to be trusted.
        # That's the problem with setting non-binding to row means.

        allData = allData.to_numpy()
        mask = np.ones_like(allData)
        mask[len(allData) - len(chunk) :, :] = 0
        results = predict_space_opt_CMAP_data(allData, mask, len(chunk), X)
        predicted_energies = predicted_energies.append(
            pd.DataFrame(
                results,
                index=daniel_energies_chunk.index,
                columns=training_data.columns,
            )
        )
    save_matrix(predicted_energies, "predicted_binding_energies_for_78K_OSDAs.pkl")


def make_skinny(allData, col_1="variable", col_2="SMILES"):
    allData = allData.reset_index()
    melted_matrix = pd.melt(
        allData, id_vars=col_2, value_vars=list(allData.columns[1:])
    )
    return melted_matrix.set_index([col_2, col_1])


def lets_look_at_predicted_energies():
    (
        training_data,
        binaryData,
        only_train,
        method,
        SEED,
        path_prefix,
        plot,
        prior,
    ) = validate_zeolite_inputs(col_name="SMILES")
    # fill_value = 30
    # training_data[training_data == fill_value] = None
    # training_data = training_data.apply(lambda row: row.fillna(row.mean()), axis=1)
    # # Let's filter out all the rows with all empty values...
    # # e.g., these four: C1COCCOCCNCCOCCOCCN1, C1COCCOCCOCCN1, Nc1ccccc1, OCC(CO)(CO)CO
    # training_data = training_data.dropna(thresh=1)
    # training_data = training_data[
    #     training_data.max(axis=1) != training_data.min(axis=1)
    # ]
    # skinny_training_data = make_skinny(training_data, col_1="Zeolite", col_2="SMILES")
    # skinny_training_data.plot.hist(bins=30)
    # plt.savefig("training_data_histogram.png", dpi=100)

    daniel_energies = pd.read_pickle(
        "data/data_from_daniels_ml_models/precomputed_energies_78616by196.pkl"
    )
    predicted_energies = pd.read_pickle(
        "data/predicted_templating_energies_for_78K_OSDAs.pkl"
    )
    # predicted_energies = pd.read_pickle(
    #     "data/predicted_binding_energies_for_78K_OSDAs.pkl"
    # )
    # all_energies = pd.concat([predicted_energies, training_data])
    sorted_training_data_by_column = pd.DataFrame()
    for col in predicted_energies:
        sorted_training_data_by_column[col] = training_data[col].sort_values(
            ignore_index=True
        )

    sorted_energies_by_column = pd.DataFrame()
    for col in predicted_energies:
        sorted_energies_by_column[col] = predicted_energies[col].sort_values(
            ignore_index=True
        )
    
    bag_of_differences = []
    bag_where_we_beat_existing_OSDAs_with_labels = []
    for col_index in range(len(sorted_energies_by_column.columns)):
        # if sorted_energies_by_column.iloc[0, col_index] > sorted_training_data_by_column.iloc[0, col_index]:
        #     continue
        difference = sorted_training_data_by_column.iloc[0, col_index] - sorted_energies_by_column.iloc[0, col_index]
        bag_of_differences.append(difference) 
        bag_where_we_beat_existing_OSDAs_with_labels.append((difference, col_index))
    plt.hist(bag_of_differences, bins = 100)
    plt.show()
    plt.savefig("histogram_where_were_lower_than_literature.png", dpi=30)

    bag_where_we_beat_existing_OSDAs_with_labels.sort(key = lambda x: x[0]) 

    lowest_value = sorted_energies_by_column.iloc[0,154]
    column_name = sorted_energies_by_column.columns[154]
    row = predicted_energies.loc[predicted_energies[column_name] == lowest_value]
    smile_to_property('CC(C)[P+](C(C)C)(C(C)C)C(C)C', debug=True)

    lowest_value = sorted_training_data_by_column.iloc[0,154]
    column_name = sorted_training_data_by_column.columns[154]
    training_row = training_data.loc[training_data[column_name] == lowest_value]
    # 112
    # sorted_energies_by_column.iloc[0, 112]


    # difference & the corresponding lowest templating energy.
    differences_between_last_two = [
        (
            sorted_energies_by_column.iloc[1, col_index]
            - sorted_energies_by_column.iloc[0, col_index],
            col_index,
        )
        for col_index in range(len(sorted_energies_by_column.columns))
    ]
    differences_between_last_two.sort(key = lambda x: x[0]) 
    # sorted_energies_by_column[]
    lowest_value = sorted_energies_by_column.iloc[0,147]
    column_name = sorted_energies_by_column.columns[147]
    row = predicted_energies.loc[predicted_energies[column_name] == lowest_value]
    smile_to_property('CCC[N+](CCC)(CCC)CCC', debug=True)


    # differences_between_last_two = [
    #     sorted_energies_by_column.iloc[1, col_index] - sorted_energies_by_column.iloc[0, col_index]
    #     for col_index in range(len(sorted_energies_by_column.columns))
    # ]
    # differences_between_last_two.sort() 


    # plt.hist(differences_between_last_two, bins = 30)
    # plt.show()
    # plt.savefig("predicted_energy_difference_histogram.png", dpi=100)
    # pdb.set_trace()

    # # predicted_energies.mean(axis=1).max()
    # skinny_energies = make_skinny(predicted_energies, col_1="Zeolite", col_2="index")
    # skinny_energies.plot.hist(bins=30)
    # plt.savefig("predicted_energy_histogram.png", dpi=100)

    # # Sorted each energy
    # skinny_energies = skinny_energies.sort_values(by=['value'])
    # smile_to_property('C[C@H]1CCCC[N@@+]12CCC[C@@H]2C', debug=True)

    # For comparison make sure they all have the same columns & rows
    daniel_energies = daniel_energies.loc[
        set.intersection(set(predicted_energies.index), set(daniel_energies.index))
    ]
    predicted_energies = predicted_energies.loc[
        set.intersection(set(predicted_energies.index), set(daniel_energies.index))
    ]
    predicted_energies = predicted_energies.filter(daniel_energies.columns)
    predicted_energies = predicted_energies.reindex(daniel_energies.index)
    daniel_energies = daniel_energies.reindex(predicted_energies.index)
    top_20_accuracies = [
        calculate_top_k_accuracy(daniel_energies, predicted_energies, k)
        for k in range(0, 21)
    ]
    plot_top_k_curves(top_20_accuracies)
    cosims, r2_scores, rmse_scores, spearman_scores = res_without_filter(
        daniel_energies.to_numpy(), predicted_energies.to_numpy()
    )
    print(
        "cosims:\n",
        np.mean(cosims),
        "r2_scores:\n",
        np.mean(r2_scores),
        "rmse_scores:\n",
        np.mean(rmse_scores),
        "spearman_scores:\n",
        np.mean(spearman_scores),
        "top_20_accuracies: ",
        top_20_accuracies,
    )

    print("hello")


# TODO: Are these the more accurate results to report?
def res_without_filter(true, pred):
    cosims = []
    r2_scores = []
    rmse_scores = []
    spearman_scores = []
    for row_id in tqdm(range(true.shape[0])):
        i = true[row_id]
        j = pred[row_id]
        cosims.append(get_cosims(np.array([i]), np.array([j])))
        r2_scores.append(r2_score(i, j, multioutput="raw_values"))
        rmse_scores.append(
            [math.sqrt(mean_squared_error(i, j, multioutput="raw_values"))]
        )
        spearman_scores.append([spearmanr(i, j).correlation])
    return cosims, r2_scores, rmse_scores, spearman_scores


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
    binding_data = pd.read_pickle("data/BindingSiO2.pkl")
    binding_data = binding_data.apply(lambda row: row.fillna(30), axis=1)
    # Let's filter out all the rows with all empty values...
    # e.g., these four: C1COCCOCCNCCOCCOCCN1, C1COCCOCCOCCN1, Nc1ccccc1, OCC(CO)(CO)CO
    binding_data = binding_data.dropna(thresh=1)
    binding_data = binding_data[binding_data.max(axis=1) != binding_data.min(axis=1)]
    # Let's take out rows that have just no templating energies at all...
    # not even sure how they got into the dataset... Worth investigating...
    # e.g., these four: C1COCCOCCNCCOCCOCCN1, C1COCCOCCOCCN1, Nc1ccccc1, OCC(CO)(CO)CO
    allData = allData[allData.max(axis=1) != allData.min(axis=1)]
    templating_predictions = run_ntk(
        binding_data,
        only_train,
        method,
        SEED,
        path_prefix,
        plot,
        prior="CustomOSDAVector",
        fill_value=30,
        average_by_rows=True,
    )
    pdb.set_trace()
    # plot_matrix(
    #     binding_predictions.to_numpy(),
    #     "binding_predictions",
    #     binaryData.to_numpy(),
    # )

    pdb.set_trace()

    # save_matrix(binding_predictions, "templating_pred.pkl")
    # save_matrix(binary_predictions, "mask_pred.pkl")

    print("hello yitong! howzit look all combined?")


if __name__ == "__main__":
    # test_all_osdas()
    # buisness_as_normal()
    # lets_look_at_predicted_energies()
    skinny_ntk()
