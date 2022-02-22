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
from analysis_utilities import examine_osda_feature_causes, plot_volume


sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))

from ntk import predict
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
from path_constants import (
    HYPOTHETICAL_OSDA_ENERGIES,
    HYPOTHETICAL_OSDA_BOXES,
    OSDA_HYPOTHETICAL_PRIOR_FILE,
    OSDA_HYPOTHETICAL_PREDICTED_ENERGIES,
    ZEOLITE_HYPOTHETICAL_PREDICTED_ENERGIES,
    TEN_FOLD_CROSS_VALIDATION_ENERGIES,
    ZEO_1_PRIOR,
)

NORM_FACTOR = 0.001


sys.path.insert(
    1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical")
)

# This method calculates binding energies for the 78K new OSDAs
# using the Ground truth set of 1.19K x 200 matrix
def calculate_energies_for_78K_osdas():
    # training_data ends up as 1190 rows x 209 columns
    training_data, _binary_data = get_ground_truth_energy_matrix(
        energy_type=Energy_Type.BINDING  # TEMPLATING or BINDING
    )
    daniel_energies = pd.read_csv(HYPOTHETICAL_OSDA_ENERGIES)
    precomputed_priors = pd.read_csv(OSDA_HYPOTHETICAL_PRIOR_FILE)
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
        all_data = all_data.to_numpy()
        mask = np.ones_like(all_data)
        mask[len(all_data) - len(chunk) :, :] = 0
        results = predict(all_data, mask, len(chunk), X)
        # predict energies for the OSDAs of chunk_size & append predictions to our growing list
        predicted_energies = predicted_energies.append( 
            pd.DataFrame(
                results,
                index=daniel_energies_chunk.index,
                columns=training_data.columns,
            )
        )
    save_matrix(predicted_energies, OSDA_HYPOTHETICAL_PREDICTED_ENERGIES)


# This method is a pure mess. but it's just examining the predicted energies for daniel's 78K new OSDAs.
def lets_look_at_predicted_energies():
    daniel_energies = pd.read_csv(HYPOTHETICAL_OSDA_ENERGIES)
    predicted_energies = pd.read_pickle(OSDA_HYPOTHETICAL_PREDICTED_ENERGIES)
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


def calculate_energies_for_new_zeolite():
    """
    Calculate energies for Zeo-1 using our ground truth matrix.
    """
    ground_truth, binary_data = get_ground_truth_energy_matrix(
        energy_type=Energy_Type.TEMPLATING, transpose=True
    )
    def predict_for_new_zeolite(ground_truth, new_zeolites_df, zeolite):
        num_new_zeolites = len(new_zeolites_df)
        # add new zeolites to training data
        all_data = ground_truth.append(
            new_zeolites_df.drop(new_zeolites_df.columns, axis=1)
        )
        X = make_prior(
            None,
            None,
            method="CustomZeolite",
            normalization_factor=NORM_FACTOR,
            all_data=all_data,
        )
        all_data = all_data.to_numpy()
        mask = np.ones_like(all_data)
        mask[len(all_data) - num_new_zeolites :, :] = 0
        results = predict(all_data, mask, num_new_zeolites, X)
        # plot_double_histogram(make_skinny(ground_truth), results[0], zeolite)
        return results

    new_zeolites_df = pd.read_pickle(ZEO_1_PRIOR)
    results = predict_for_new_zeolite(ground_truth, new_zeolites_df, "zeo-1")
    series = pd.Series(results[0])
    series.index = ground_truth.columns

    prediction_ntk = pd.DataFrame(series)
    prediction_ntk = prediction_ntk.T
    prediction_ntk.to_csv("predicted_zeo1_osdas.csv")
    prediction_ntk.to_pickle("predicted_zeo1_osdas.pkl")

    metrics_per_heldout_zeolite = []
    for zeolite in ground_truth.index:
        all_but_one = ground_truth.loc[ground_truth.index != zeolite]
        held_out = pd.DataFrame(index=[zeolite])
        pred = predict_for_new_zeolite(all_but_one, held_out, zeolite)
        true = ground_truth.loc[ground_truth.index == zeolite].to_numpy()
        metrics = calculate_metrics(pred, true, verbose=False)
        metrics_per_heldout_zeolite.append((metrics, zeolite))

    def calc_rmse_error_by_feature(zeolite_priors, feature):
        volume_rmse_pairs = [
            (zeolite_priors.loc[h[1]][feature], h[0]["rmse_scores"])
            for h in metrics_per_heldout_zeolite
        ]
        x_val = [x[0] for x in volume_rmse_pairs]
        y_val = [x[1] for x in volume_rmse_pairs]
        plt.scatter(x_val, y_val)
        plt.savefig(feature + "_vs_volume.png", dpi=150)
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x_val, y_val)
        pdb.set_trace()

    zeolite_priors = zeolite_prior(ground_truth, None, normalize=False)
    examine_osda_feature_causes(prediction_ntk)
    plot_volume(zeolite_priors)

    calc_rmse_error_by_feature(zeolite_priors, "included_sphere_diameter")
    calc_rmse_error_by_feature(zeolite_priors, "volume")
    breakpoint()


def calculate_energies_for_new_zeolite_skinny_style():
    ground_truth, binary_data = get_ground_truth_energy_matrix(
        energy_type=Energy_Type.TEMPLATING, transpose=True
    )

    def predict_for_new_zeolite(ground_truth, new_zeolites_df, zeolite):
        """
        Breaking the intention of this method a bit... new_zeolites_df can only have one index of one new zeolite...
        """
        skinny_ground_truth = make_skinny(ground_truth)
        # TODO: fix this... allow this method to accept more than just one new zeolite at a time.
        single_new_zeolite_name = new_zeolites_df.index[0]
        predicted_index = [(i, single_new_zeolite_name) for i in ground_truth.columns]
        new_zeolites_df = pd.DataFrame(index=predicted_index, data=[])
        num_new_zeolites = len(new_zeolites_df)

        # Sample more intelligently from skinny_ground_truth
        skinny_ground_truth = skinny_ground_truth.sample(5000)

        # add new zeolites to training data
        all_data = skinny_ground_truth.append(
            new_zeolites_df.drop(new_zeolites_df.columns, axis=1)
        )
        X = make_prior(
            None,
            None,
            method="CustomOSDAandZeoliteAsRows",
            normalization_factor=NORM_FACTOR,
            all_data=all_data,
        )
        all_data = all_data.to_numpy()
        mask = np.ones_like(all_data)
        mask[len(all_data) - num_new_zeolites :, :] = 0
        results = predict(all_data, mask, num_new_zeolites, X)
        # plot_double_histogram(skinny_ground_truth.values, results, zeolite)
        return results

    # new_zeolites_df = pd.read_pickle(ZEO_1_PRIOR)
    # predict_for_new_zeolite(ground_truth, new_zeolites_df, "zeo-1")
    # # pdb.set_trace()

    metrics_per_heldout_zeolite = []
    for zeolite in tqdm(ground_truth.index):
        all_but_one = ground_truth.loc[ground_truth.index != zeolite]
        held_out = pd.DataFrame(index=[zeolite])
        pred = predict_for_new_zeolite(all_but_one, held_out, zeolite)
        true = ground_truth.loc[ground_truth.index == zeolite].to_numpy()
        metrics = calculate_metrics(pred, true, verbose=False)
        metrics_per_heldout_zeolite.append((metrics, zeolite))

    def calc_rmse_error_by_feature(zeolite_priors, feature):
        volume_rmse_pairs = [
            (zeolite_priors.loc[h[1]][feature], h[0]["rmse_scores"])
            for h in metrics_per_heldout_zeolite
        ]
        x_val = [x[0] for x in volume_rmse_pairs]
        y_val = [x[1] for x in volume_rmse_pairs]
        plt.scatter(x_val, y_val)
        plt.savefig(feature + "_vs_volume.png", dpi=150)
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x_val, y_val)
        print(
            feature + "_vs_volume ",
            "slope ",
            slope,
            "intercept ",
            intercept,
            "r_value ",
            r_value,
            "p_value ",
            p_value,
            "std_err ",
            std_err,
        )

    zeolite_priors = zeolite_prior(ground_truth, None, normalize=False)
    calc_rmse_error_by_feature(zeolite_priors, "included_sphere_diameter")
    calc_rmse_error_by_feature(zeolite_priors, "volume")
    breakpoint()


if __name__ == "__main__":
    calculate_energies_for_78K_osdas()
    # lets_look_at_predicted_energies()
    # calculate_energies_for_new_zeolite()
    # calculate_energies_for_new_zeolite_skinny_style()
