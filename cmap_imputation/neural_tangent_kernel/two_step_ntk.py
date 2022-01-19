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

from prior import zeolite_prior
from analysis_utilities import calculate_top_k_accuracy

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))

from package_matrix import (
    Energy_Type,
    get_ground_truth_energy_matrix,
    make_skinny,
    unmake_skinny,
)
from utilities import (
    get_splits_in_zeolite_type,
    plot_two_matrices,
    save_matrix,
    chunks,
)
from analysis_utilities import calculate_metrics

sys.path.insert(
    1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical")
)

SEED = 5
NORM_FACTOR = 0.1
PI = np.pi


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


def run_ntk(
    all_data,
    prior,
    metrics_mask,
    shuffled_iterator=True,
    k_folds=10,
    SEED=SEED,
):
    if shuffled_iterator:
        # The iterator shuffles the data which is why we need to pass in metrics_mask together.
        iterator = tqdm(
            get_splits_in_zeolite_type(all_data, metrics_mask, k=k_folds, seed=SEED),
            total=k_folds,
        )
    else:
        # For skinny matrix we need to chunk to make sure we don't expose
        # test zeolites/OSDA columns/rows to our training
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
    aggregate_mask = None  # Masks for all fold(s)
    # Iterate over all predictions and populate metrics matrices
    for train, test, test_mask_chunk in iterator:
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
    # We return aggregate_pred, aggregate_true, aggregate_mask for the case when shuffled_iterator=True
    # And the results are shuffled from the original all_data & metrics_mask
    return aggregate_pred, aggregate_true, aggregate_mask

# This method calculates binding energies for the 78K new OSDAs
# from the Ground truth set of 1.19K x 200 matrix
def calculate_energies_for_78K_osdas():
    # training_data ends up as 1190 rows x 209 columns
    training_data, _binary_data = get_ground_truth_energy_matrix(
        energy_type=Energy_Type.BINDING  # TEMPLATING or BINDING
    )

    # These are predictions according to Daniel's per Zeolite ML model for 78K OSDAs
    daniel_energies = pd.read_pickle(
        "data/daniels_data/precomputed_energies_78616by196.pkl"
    )
    # These are precomputed priors for those 78K OSDAs
    precomputed_priors = pd.read_pickle(
        "data/daniels_data/precomputed_energies_78616by196WithWhims.pkl"
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
    for _train_chunk, chunk, _none in iterator:
        daniel_energies_chunk = daniel_energies.iloc[chunk, :]
        all_data = pd.concat([training_data, daniel_energies_chunk])
        X = make_prior(
            None,
            None,
            method="CustomOSDA",
            normalization_factor=NORM_FACTOR,
            all_data=all_data,
        )

        # So now the question becomes, do we do the binary sweep first? My guess is yes.
        # and after the binary sweep we can zero everything to some middle value...
        # bleh binary sweep doesn't work well :(
        # Remember that we set all non-binding energies to the row mean so predicted templating energies
        # that are close to row means might be way off.
        # Only the really low values & really high values are to be trusted.
        # That's the problem with setting non-binding to row means.

        all_data = all_data.to_numpy()
        mask = np.ones_like(all_data)
        mask[len(all_data) - len(chunk) :, :] = 0
        results = predict(all_data, mask, len(chunk), X)
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

    differences_between_last_two = [
        sorted_energies_by_column.iloc[1, col_index]
        - sorted_energies_by_column.iloc[0, col_index]
        for col_index in range(len(sorted_energies_by_column.columns))
    ]
    differences_between_last_two.sort()

    plt.hist(differences_between_last_two, bins=30)
    plt.show()
    plt.savefig("predicted_energy_difference_histogram.png", dpi=100)
    pdb.set_trace()

    # predicted_energies.mean(axis=1).max()
    skinny_energies = make_skinny(predicted_energies, col_1="Zeolite", col_2="index")
    skinny_energies.plot.hist(bins=30)
    plt.savefig("predicted_energy_histogram.png", dpi=100)

    # Sorted each energy
    skinny_energies = skinny_energies.sort_values(by=["value"])
    smile_to_property("C[C@H]1CCCC[N@@+]12CCC[C@@H]2C", debug=True)

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
    calculate_metrics(predicted_energies.to_numpy(), daniel_energies.to_numpy())

    print("hello")


def skinny_ntk():
    """
    This method runs 10-fold cross validation on the 1194x209 Ground Truth matrix made SKINNY
    """
    # Make sure the row # for desired_shape is modulo 10 (aka our k_folds size)
    ground_truth, binary_data = get_ground_truth_energy_matrix(desired_shape=(100, 30))
    skinny_ground_truth = make_skinny(ground_truth)

    skinny_pred, _t, _m = run_ntk(
        skinny_ground_truth,
        prior="CustomOSDAandZeoliteAsRows",
        metrics_mask=binary_data,
        shuffled_iterator=False,
    )
    pred = unmake_skinny(skinny_pred)
    calculate_metrics(pred.to_numpy(), ground_truth.to_numpy(), binary_data.to_numpy())


def buisness_as_normal():
    """
    This method runs 10-fold cross validation on the 1194x209 Ground Truth matrix.
    """
    ground_truth, binary_data = get_ground_truth_energy_matrix(
        energy_type=Energy_Type.BINDING, transpose=True
    )

    # change prior=CustomZeolite etc.
    pred, true, mask = run_ntk(
        ground_truth, prior="CustomZeolite", metrics_mask=binary_data
    )

    # Which top k accuracy to look at
    K_ACCURACY = 100

    def get_top_k_for_all_osdas(pred, true):
        # pred.shape[1]
        return [calculate_top_k_accuracy(true, pred, k) for k in range(0, K_ACCURACY)]

    # get_top_k_for_all_osdas(np.array([pred.iloc[0].to_numpy()]), np.array([true.iloc[0].to_numpy()]))
    metrics = [
        (
            column,
            get_top_k_for_all_osdas(
                np.array([pred.loc[column].to_numpy()]),
                np.array([true.loc[column].to_numpy()]),
            ),
        )
        for column in pred.index
    ]

    accurate_zeolites = [
        (h[0], h[1].index(1.0) if 1.0 in h[1] else K_ACCURACY) for h in metrics
    ]
    sorted_accurate_zeolites = sorted(accurate_zeolites, key=lambda x: x[1])
    zeolite_priors = zeolite_prior(ground_truth, None, normalize=False)

    # plot k-accuracy against volume (spoiler: no trends there)
    volume_accuracy_pairs = [
        (zeolite_priors.loc[h[0]]["volume"], h[1]) for h in sorted_accurate_zeolites
    ]
    x_val = [x[0] for x in volume_accuracy_pairs]
    y_val = [x[1] for x in volume_accuracy_pairs]
    # plt.scatter(x_val,y_val)

    breakpoint()
    calculate_metrics(pred.to_numpy(), true.to_numpy(), mask.to_numpy())
    save_matrix(pred, "data/templating_pred.pkl")


def calculate_energies_for_new_zeolite(name, parameter_file):
    # training_data ends up as 1190 rows x 209 columns
    training_data, _binary_data = get_ground_truth_energy_matrix(
        energy_type=Energy_Type.BINDING
    )

    # number of new zeolites
    num_new_zeolites = 1

    # add new zeolite to training data
    td = training_data.T
    new = pd.Series([])
    new.name = name
    td = td.append(new)

    X = make_prior(
        None,
        None,
        method="CustomZeolite",
        normalization_factor=NORM_FACTOR,
        all_data=td,
        file_name=parameter_file,
    )
    all_data = td.to_numpy()
    mask = np.ones_like(all_data)
    mask[len(all_data) - num_new_zeolites :, :] = 0
    results = predict(all_data, mask, num_new_zeolites, X)
    save_matrix(results, "predicted_binding_energies_for_new_zeolite.pkl")


if __name__ == "__main__":
    buisness_as_normal()
    calculate_energies_for_new_zeolite(name="ZEO1", parameter_file="data/zeo_1.pkl")
    # skinny_ntk()
    # calculate_energies_for_78K_osdas()
    # lets_look_at_predicted_energies()
