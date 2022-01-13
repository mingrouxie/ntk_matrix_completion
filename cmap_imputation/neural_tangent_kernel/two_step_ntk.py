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
from precompute_osda_priors import smile_to_property
from enum import Enum


sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))
from utilities import (
    get_cosims,
    get_splits_in_zeolite_type,
    plot_matrix,
    plot_two_matrices,
    plot_top_k_curves,
    save_matrix,
)

sys.path.insert(
    1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical")
)

SEED = 5
NORM_FACTOR = 0.001
PI = np.pi


class Energy_Type(Enum):
    TEMPLATING = 1
    BINDING = 2


def kappa(x):
    return (x * (PI - np.arccos(x)) + np.sqrt(1 - np.square(x))) / PI + (
        x * (PI - np.arccos(x))
    ) / PI


def predict(all_data, mask, num_test_rows, X):
    """
    Space optimized version for CMap Data
    """
    all_data = all_data.T
    mask = mask.T
    num_observed = int(np.sum(mask[0:1, :]))
    num_missing = mask[0:1, :].shape[-1] - num_observed

    K_matrix = np.zeros((num_observed, num_observed))
    k_matrix = np.zeros((num_observed, num_missing))
    observed_data = all_data[:, :num_observed]

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


# If metrics_mask is given then filter out all the non-binding results according to mask
def calculate_metrics(true, pred, metrics_mask=None):
    cosims = []
    r2_scores = []
    rmse_scores = []
    spearman_scores = []
    for row_id in range(true.shape[0]):
        i = true[row_id]
        j = pred[row_id]
        if metrics_mask is not None:
            m = metrics_mask.iloc[row_id].to_numpy()
            j = j[m == 1.0]
            i = i[m == 1.0]
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


def run_ntk(
    allData,
    prior,
    metrics_mask,
    skinny=False,
    k_folds=10,
    SEED=SEED,
):
    # The iterator shuffles the data which is why we need to pass in metrics_mask together.
    iterator = tqdm(
        get_splits_in_zeolite_type(allData, metrics_mask, k=k_folds, seed=SEED),
        total=k_folds,
    )
    all_true = None  # Ground truths for all fold(s)
    all_metrics_ntk = (
        None  # Metrics (R^2, Cosine Similarity, Spearman, etc.) for each entry
    )
    ntk_predictions = None  # Predictions for all fold(s)
    splits = None  # Per-fold metrics
    # Iterate over all predictions and populate metrics matrices
    for train, test, test_metrics_mask in iterator:
        X = make_prior(
            train,
            test,
            prior,
            normalization_factor=NORM_FACTOR,
        )
        all_data = pd.concat([train, test]).to_numpy()
        ##### SAFETY
        all_data[train.shape[0] :, :] = 0
        ##### SAFETY
        mask = np.ones_like(all_data)
        mask[len(train) :, :] = 0

        # The bottom 1/10 of mask and all_data are just all zeros.
        results_ntk = predict(all_data, mask, len(test), X=X)

        if skinny:
            results_ntk, true = unmake_skinny(results_ntk, test)
        else:
            true = test.to_numpy()

        prediction_ntk = pd.DataFrame(
            data=results_ntk, index=test.index, columns=test.columns
        )
        # ntk_predictions is just the running collection of all the prediction_ntk
        ntk_predictions = (
            prediction_ntk
            if ntk_predictions is None
            else pd.concat([ntk_predictions, prediction_ntk])
        )
        cosims, r2_scores, rmse_scores, spearman_scores = calculate_metrics(
            true, results_ntk, test_metrics_mask
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

    # NOTE: this is top_k_accuracy as split by ROWS in the matrix
    top_1 = calculate_top_k_accuracy(all_true, ntk_predictions, 1)
    top_3 = calculate_top_k_accuracy(all_true, ntk_predictions, 3)
    top_5 = calculate_top_k_accuracy(all_true, ntk_predictions, 5)
    top_20_accuracies = [
        calculate_top_k_accuracy(all_true, ntk_predictions, k) for k in range(0, 21)
    ]
    print(
        splits.mean(),
        "\ntop_1_accuracy: ",
        top_1.round(4),
        "\ntop_3_accuracy: ",
        top_3.round(4),
        "\ntop_5_accuracy: ",
        top_5.round(4),
    )
    plot_top_k_curves(top_20_accuracies)
    # TODO: should probably add that as an argument
    # For binding energies versus templating energies you'll want to set vmin=-30, vmax=5
    plot_matrix(all_true, "regression_truth")
    plot_matrix(ntk_predictions, "regression_prediction")
    return ntk_predictions


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


def get_ground_truth_energy_matrix(energy_type=Energy_Type.TEMPLATING):
    if energy_type == Energy_Type.TEMPLATING:
        ground_truth = pd.read_pickle("data/TemplatingGroundTruth.pkl")
        templating_energy = pd.read_pickle("data/TemplatingGroundTruth.pkl")
    elif energy_type == Energy_Type.BINDING:
        ground_truth = pd.read_pickle("data/BindingSiO2GroundTruth.pkl")
    else:
        # REFERENCE format_ground_truth_pkl()
        raise ValueError(
            "Sorry, but if you want to use a different ground truth for the energy then create the matrix first."
        )
    # Set all empty spots in the matrix to be the row mean
    ground_truth = ground_truth.apply(lambda row: row.fillna(row.mean()), axis=1)
    ground_truth = ground_truth.dropna(thresh=1)
    # Let's take out rows that have just no  energies at all...
    # not even sure how they got into the dataset... Worth investigating...
    # e.g., these four: C1COCCOCCNCCOCCOCCN1, C1COCCOCCOCCN1, Nc1ccccc1, OCC(CO)(CO)CO
    ground_truth = ground_truth[ground_truth.max(axis=1) != ground_truth.min(axis=1)]

    # This is the Ground Truth for the sparsity of the matrix
    # 1s for whether pairings physically make sense 0s if they do not.
    # We could also generate it straight from ground_truth, meh.
    binary_data = pd.read_pickle("data/BinaryGroundTruth.pkl")
    binary_data = binary_data.reindex(ground_truth.index)
    return ground_truth, binary_data


def skinny_ntk():
    # TODO: Make sure we don't leak information by taking the row mean
    # from test to train ... is that possible?
    # uh oh, potentially yes since we're splitting not by column or rows anymore.
    ground_truth, binary_data = get_ground_truth_energy_matrix()
    # TODO(Mingrou): You'll also want to set prior="CustomZeolite"

    allData = ground_truth.iloc[:100, :30]
    allData = make_skinny(allData, col_1="Zeolite")

    binary_data = binary_data.iloc[:100, :30]
    binary_data = make_skinny(binary_data, col_1="Zeolite")

    # TODO: for regression on skinny matrix... see how to average_by_rows...
    # TODO: rather than make things weird. just pass in a mask for what to use in calculations.
    run_ntk(
        allData,
        prior="CustomOSDAandZeoliteAsRows",
        metrics_mask=binary_data,
        skinny=True,
    )


# This method calculates binding energies for the 78K new OSDAs
# from the Ground truth set of 1.19K x 200 matrix
def calculate_energies_for_78K_osdas():
    # training_data ends up as 1190 rows x 209 columns
    training_data, _binary_data = get_ground_truth_energy_matrix(Energy_Type.BINDING)

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
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    chunk_size = 10000
    iterator = tqdm(chunks(range(len(daniel_energies.index)), chunk_size))
    predicted_energies = pd.DataFrame()
    for chunk in iterator:
        daniel_energies_chunk = daniel_energies.iloc[chunk, :]
        allData = pd.concat([training_data, daniel_energies_chunk])
        X = make_prior(
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
        results = predict(allData, mask, len(chunk), X)
        predicted_energies = predicted_energies.append(
            pd.DataFrame(
                results,
                index=daniel_energies_chunk.index,
                columns=training_data.columns,
            )
        )
    save_matrix(predicted_energies, "predicted_binding_energies_for_78K_OSDAs.pkl")


# This method is pure mess. but it's just examining the predicted energies for daniel's 78K new OSDAs
def lets_look_at_predicted_energies():
    daniel_energies = pd.read_pickle(
        "data/data_from_daniels_ml_models/precomputed_energies_78616by196.pkl"
    )
    predicted_energies = pd.read_pickle(
        "data/predicted_templating_energies_for_78K_OSDAs.pkl"
    )
    # predicted_energies = pd.read_pickle(
    #     "data/predicted_binding_energies_for_78K_OSDAs.pkl"
    # )
    training_data, _binary_data = get_ground_truth_energy_matrix()
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
        difference = (
            sorted_training_data_by_column.iloc[0, col_index]
            - sorted_energies_by_column.iloc[0, col_index]
        )
        bag_of_differences.append(difference)
        bag_where_we_beat_existing_OSDAs_with_labels.append((difference, col_index))
    plt.hist(bag_of_differences, bins=100)
    plt.show()
    plt.savefig("histogram_where_were_lower_than_literature.png", dpi=30)

    bag_where_we_beat_existing_OSDAs_with_labels.sort(key=lambda x: x[0])

    lowest_value = sorted_energies_by_column.iloc[0, 154]
    column_name = sorted_energies_by_column.columns[154]
    row = predicted_energies.loc[predicted_energies[column_name] == lowest_value]
    smile_to_property("CC(C)[P+](C(C)C)(C(C)C)C(C)C", debug=True)

    lowest_value = sorted_training_data_by_column.iloc[0, 154]
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
    differences_between_last_two.sort(key=lambda x: x[0])
    # sorted_energies_by_column[]
    lowest_value = sorted_energies_by_column.iloc[0, 147]
    column_name = sorted_energies_by_column.columns[147]
    row = predicted_energies.loc[predicted_energies[column_name] == lowest_value]
    smile_to_property("CCC[N+](CCC)(CCC)CCC", debug=True)

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
    cosims, r2_scores, rmse_scores, spearman_scores = calculate_metrics(
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


# This command runs 10-fold cross validation on the 1194x209 Ground Truth matrix.
def buisness_as_normal():
    # TODO(Mingrou): For the new zeolite you'll want to take the transpose of ground_truth & binary_data
    ground_truth, binary_data = get_ground_truth_energy_matrix()
    # TODO(Mingrou): You'll also want to set prior="CustomZeolite"
    templating_predictions = run_ntk(
        ground_truth, prior="CustomOSDAVector", metrics_mask=binary_data
    )
    save_matrix(templating_predictions, "templating_pred.pkl")


def format_ground_truth_pkl():
    ground_truth_df = pd.read_csv("data/binding.csv", index_col=0)
    binding_matrix = ground_truth_df.pivot(
        index="SMILES", columns="Zeolite", values="Binding (SiO2)"
    )
    print(
        "The binding matrix has these many values: ",
        binding_matrix.notna().sum().sum(),
        " out of these many total cells",
        binding_matrix.isna().sum().sum() + binding_matrix.notna().sum().sum(),
    )
    save_matrix(binding_matrix, "BindingSiO2GroundTruth.pkl")

    templating_matrix = ground_truth_df.pivot(
        index="SMILES", columns="Zeolite", values="Templating"
    )
    print(
        "The templating matrix has these many values: ",
        templating_matrix.notna().sum().sum(),
        " out of these many total cells",
        templating_matrix.isna().sum().sum() + templating_matrix.notna().sum().sum(),
    )
    save_matrix(templating_matrix, "TemplatingGroundTruth.pkl")


if __name__ == "__main__":
    # format_ground_truth_pkl()
    calculate_energies_for_78K_osdas()
    # lets_look_at_predicted_energies()
    # buisness_as_normal()
