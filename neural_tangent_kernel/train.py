import sys
import pathlib
import pandas as pd
import io
import os
from sklearn.model_selection import train_test_split

import pdb
from utilities import plot_matrix
from analysis_utilities import plot_top_k_curves
from weights import ZEOLITE_PRIOR_LOOKUP, OSDA_PRIOR_LOOKUP
from ntk import run_ntk, skinny_ntk_sampled_not_sliced

import numpy as np
import time

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))

from package_matrix import (
    Energy_Type,
    get_ground_truth_energy_matrix,
    make_skinny,
    unmake_skinny,
)
from utilities import (
    save_matrix,
)
from analysis_utilities import calculate_metrics
from path_constants import (
    TEN_FOLD_CROSS_VALIDATION_ENERGIES,
    ZEOLITE_PRIOR_SELECTION_FILE,
)

TEST_SEED = 424956


def buisness_as_normal():
    """
    This method runs 10-fold cross validation on the 1194x209 Ground Truth matrix.
    With OSDAs as rows (aka using OSDA priors to predict energies for new OSDAs)
    """
    ground_truth, binary_data = get_ground_truth_energy_matrix()
    pred, true, mask = run_ntk(
        ground_truth,
        prior="CustomOSDAVector",
        metrics_mask=binary_data,
        use_eigenpro=False,
    )
    calculate_metrics(
        pred.to_numpy(),
        true.to_numpy(),
        mask.to_numpy(),
        verbose=True,
        method="top_k_in_top_k",
    )
    plot_matrix(pred, "pred")
    save_matrix(pred, TEN_FOLD_CROSS_VALIDATION_ENERGIES)


def buisness_as_normal_transposed(
    energy_type, prior, method="top_k", verbose=True, to_write=True, to_plot=False
):
    """
    This method runs 10-fold cross validation on the 1194x209 Ground Truth matrix.
    With Zeolites as rows (aka using Zeolite priors to predict energies for new Zeolites)
    """
    ground_truth, binary_data = get_ground_truth_energy_matrix(
        transpose=True, energy_type=energy_type
    )
    # pred, true, mask = run_ntk(
    #     ground_truth, prior="CustomZeoliteEmbeddings", metrics_mask=binary_data
    # )
    pred, true, mask = run_ntk(ground_truth, prior=prior, metrics_mask=binary_data)
    # breakpoint()
    results = calculate_metrics(
        pred.to_numpy(),
        true.to_numpy(),
        mask.to_numpy(),
        energy_type=Energy_Type.BINDING,
        verbose=verbose,
        method=method,
        to_write=to_write,
        to_plot=to_plot,
    )
    pdb.set_trace()

    print("finished! ")


def buisness_as_normal_skinny():
    """
    This method runs 10-fold cross validation on the 1194x209 Ground Truth matrix made SKINNY.
    """
    # Make sure the row # for desired_shape is modulo 10 (aka our k_folds size)

    # TODO: Notice that this just takes the first (100, 30) chunk of the energy matrix as a way to
    # downsample the matrix. This is almost certainly a bad idea for an accurate test.
    # skinny_ntk_sampled_not_sliced() in ntk.py addresses this but is messy be warned
    ground_truth, binary_data = get_ground_truth_energy_matrix(desired_shape=(100, 30))
    skinny_ground_truth = make_skinny(ground_truth)
    skinny_pred, _t, _m = run_ntk(
        skinny_ground_truth,
        prior="CustomOSDAandZeoliteAsRows",
        metrics_mask=binary_data,
        shuffled_iterator=False,
    )
    pred = unmake_skinny(skinny_pred)
    calculate_metrics(
        pred.to_numpy(),
        ground_truth.to_numpy(),
        binary_data.to_numpy(),
        method="top_k_in_top_k",
    )


def buisness_as_normal_transposed_over_many_priors(transpose=False, to_write=False):
    """
    This method is an example of how to test a set of priors to see their individual performance.
    """
    ground_truth, binary_data = get_ground_truth_energy_matrix(transpose=transpose)
    if not transpose:  # osda
        prior = "CustomOSDA"
        priors_to_test = [
            "mol_weights",
            "volume",
            "normalized_num_rotatable_bonds",
            "formal_charge",
            "asphericity",
            "inertial_shape_factor",
            "spherocity",
            "eccentricity",
            "gyration_radius",
            "pmi1",
            "pmi2",
            "pmi3",
            "npr1",
            "npr2",
            "free_sas",
            "bertz_ct",
        ]
    else:
        print(
            "Some temporary stuff going on here please check you are calling the right thing"
        )
        breakpoint()
        prior = "CustomZeolite"  # "CustomZeoliteEmbeddings"
        priors_to_test = ZEOLITE_PRIOR_LOOKUP.keys()
        priors_to_test = [  # TODO: Mingrou temporary, just to test the 0D
            "cell_birth-of-top-1-0D-feature",
            "cell_death-of-top-1-0D-feature",
            "cell_persistence-of-top-1-0D-feature",
            "cell_birth-of-top-2-0D-feature",
            "cell_death-of-top-2-0D-feature",
            "cell_persistence-of-top-2-0D-feature",
            "cell_birth-of-top-3-0D-feature",
            "cell_death-of-top-3-0D-feature",
            "cell_persistence-of-top-3-0D-feature",
            "supercell_birth-of-top-1-0D-feature",
            "supercell_death-of-top-1-0D-feature",
            "supercell_persistence-of-top-1-0D-feature",
            "supercell_birth-of-top-2-0D-feature",
            "supercell_death-of-top-2-0D-feature",
            "supercell_persistence-of-top-2-0D-feature",
            "supercell_birth-of-top-3-0D-feature",
            "supercell_death-of-top-3-0D-feature",
            "supercell_persistence-of-top-3-0D-feature",
        ]
    results = []
    # Let's do 10-fold cross validation & gather metrics for each prior by itself.
    for prior_to_test in priors_to_test:
        pred, true, mask = run_ntk(
            ground_truth,
            prior=prior,
            metrics_mask=binary_data,
            prior_map={prior_to_test: 1.0},
        )

        metrics = calculate_metrics(
            pred.to_numpy(),
            true.to_numpy(),
            mask.to_numpy(),
            verbose=False,
            meta=prior_to_test,
            method="top_k_in_top_k",
        )
        print(prior_to_test, metrics["top_5_accuracy"])
        results.append((prior_to_test, metrics))
    # Let's get metrics for the baseline: identity
    identity_pred, identity_true, mask = run_ntk(
        ground_truth,
        prior="identity",
        metrics_mask=binary_data,
    )
    identity_metrics = calculate_metrics(
        identity_pred.to_numpy(),
        identity_true.to_numpy(),
        mask.to_numpy(),
        verbose=False,
        method="top_k_in_top_k",
    )

    results.append(("identity", identity_metrics))
    results.sort(key=lambda x: x[1]["rmse_scores"])
    results.sort(key=lambda x: -x[1]["top_1_accuracy"])

    results_print_out = [
        r[0]
        + "\t"
        + str(r[1]["rmse_scores"])
        + "\t"
        + str(r[1]["spearman_scores"])
        + "\t"
        + str(r[1]["top_1_accuracy"])
        + "\t"
        + str(r[1]["top_3_accuracy"])
        + "\t"
        + str(r[1]["top_5_accuracy"])
        for r in results
    ]
    print(
        "here are the results sorted by rmse & top_1_accuracy "
    )  # , results_print_out)

    ## Test all the priors together...
    full_pred, full_true, mask = run_ntk(
        ground_truth, prior="identity", metrics_mask=binary_data, norm_factor=0.1
    )
    full_metrics = calculate_metrics(
        full_pred.to_numpy(),
        full_true.to_numpy(),
        mask.to_numpy(),
        verbose=False,
        method="top_k_in_top_k",
    )
    plot_matrix(full_pred, "full_pred")  # , vmin=0, vmax=2)
    plot_matrix(full_true, "full_true")  # , vmin=0, vmax=2)
    plot_top_k_curves(full_metrics["top_20_accuracies"])
    print("full metrics run: ", full_metrics)

    if to_write:
        df = pd.read_csv(io.StringIO("\n".join(results_print_out)), sep="\t")
        df.to_csv("./buisness_as_normal_transposed_over_many_priors.csv")

    return results_print_out


def select_zeolite_priors(
    energy_type,
    prior,
    metrics_method,
    n_features_to_select=10,
    direction="forward",
    best_feature="top_20_accuracy",
):
    """
    This method recursively chooses zeolites features to form a set of features that gives the best
    prediction. Code is adapted from sklearn's Sequential Feature Selector (SFS).
    Only works for forward selection because the number of features is unknown at this point in the code.
    (Only retrieved inside the run_ntk function)
    This method runs 10-fold cross validation on the 1194x209 Ground Truth matrix.
    With Zeolites as rows (aka using Zeolite priors to predict energies for new Zeolites)
    """
    prior_map = np.array(list(ZEOLITE_PRIOR_LOOKUP.keys()))
    with open(ZEOLITE_PRIOR_SELECTION_FILE, "a") as csv_file:
        np.savetxt(csv_file, np.expand_dims(prior_map, axis=0), delimiter=",", fmt="%s")
    # breakpoint()

    ground_truth, binary_data = get_ground_truth_energy_matrix(
        transpose=True, energy_type=energy_type
    )
    row_indices = np.arange(ground_truth.shape[0])
    ###### shuffle data and put aside a test set
    (
        ground_truth_train,
        ground_truth_test,
        binary_data_train,
        binary_data_test,
        row_indices_train,
        row_indices_test,
    ) = train_test_split(
        ground_truth, binary_data, row_indices, test_size=0.1, random_state=TEST_SEED
    )
    ######

    current_mask = np.zeros(shape=prior_map.shape[0], dtype=bool)
    scores = pd.DataFrame(
        index=np.arange(n_features_to_select), columns=["feature", "score"]
    )

    for i in range(n_features_to_select):
        new_feature_idx, new_feature, score = get_best_new_feature(
            ground_truth=ground_truth_train,  # note it is train set here only
            binary_data=binary_data_train,
            prior=prior,
            prior_map=prior_map,
            current_mask=current_mask,
            metrics_method=metrics_method,
            direction=direction,
            best_feature=best_feature,
        )
        current_mask[new_feature_idx] = True
        scores[i, 0] = new_feature
        scores[i, 1] = score
    
    print(f"chosen priors based on {best_feature} are {prior_map[current_mask]}")
    print("final performance is")

    chosen_prior_map = dict(
        zip(prior_map[current_mask], np.ones(prior_map[current_mask].shape))
    )
    pred, true, mask = run_ntk(
        ground_truth_train,
        prior=prior,
        metrics_mask=binary_data_train,
        prior_map=chosen_prior_map,
    )
    results = calculate_metrics(
        pred.to_numpy(),
        true.to_numpy(),
        mask.to_numpy(),
        verbose=True,
        method=metrics_method,
    )

    print(f"Test set indices are {row_indices_test}")
    # breakpoint()
    return prior_map[current_mask]


def get_best_new_feature(
    ground_truth,
    binary_data,
    prior,
    prior_map,
    current_mask,
    metrics_method,
    direction,
    best_feature,
    to_write=True,
):
    candidate_feature_indices = np.flatnonzero(~current_mask)
    scores = {}

    for feature_idx in candidate_feature_indices:
        candidate_mask = current_mask.copy()
        candidate_mask[feature_idx] = True
        if direction == "backward":
            candidate_mask = ~candidate_mask
        candidate_prior_map = prior_map.copy()
        candidate_prior_map = candidate_prior_map[candidate_mask]
        candidate_prior_map = dict(
            zip(candidate_prior_map, np.ones(candidate_prior_map.shape))
        )
        pred, true, mask = run_ntk(
            ground_truth,
            prior=prior,
            metrics_mask=binary_data,
            prior_map=candidate_prior_map,
        )
        results = calculate_metrics(
            pred.to_numpy(),
            true.to_numpy(),
            mask.to_numpy(),
            verbose=False,
            method=metrics_method,
        )
        scores[feature_idx] = results[best_feature]
    
    if best_feature in ["rmse_scores"]:
        best_feature_idx = min(scores, key=lambda feature_idx: scores[feature_idx])
    else:
        print('here')
        best_feature_idx = max(scores, key=lambda feature_idx: scores[feature_idx])
    print(
        f"New chosen feature is {prior_map[best_feature_idx]} with score {scores[best_feature_idx]}"
    )

    if to_write:
        with open(ZEOLITE_PRIOR_SELECTION_FILE, "a") as csv_file:
            score_values = np.zeros(current_mask.shape)
            score_values[list(scores.keys())] = np.array(list(scores.values()))
            score_values[current_mask] = np.nan
            score_values = np.expand_dims(score_values, axis=0)
            np.savetxt(csv_file, score_values, delimiter=",")

    return best_feature_idx, prior_map[best_feature_idx], scores[best_feature_idx]


if __name__ == "__main__":
    start = time.time()
    for best_feature in [
        # "rmse_scores", 
        "spearman_scores",
        # "top_1_accuracy",
        # "top_3_accuracy",
        # "top_5_accuracy",
        # "top_20_accuracy",
    ]:
        selected_priors = select_zeolite_priors(
            energy_type=Energy_Type.BINDING,
            prior="CustomZeolite",
            metrics_method="top_k_in_top_k",
            n_features_to_select=10,
            direction="forward",
            best_feature=best_feature,
        )
    # breakpoint()
    # buisness_as_normal()
    # buisness_as_normal_transposed()
    # buisness_as_normal_transposed(
    #     energy_type=Energy_Type.BINDING,
    #     prior="CustomZeolite",
    #     verbose=True,
    #     to_write=False,
    #     to_plot=True
    # )
    # skinny_ntk_sampled_not_sliced()
    # buisness_as_normal_skinny()
    # buisness_as_normal()
    # buisness_as_normal_transposed()
    # results_print_out = buisness_as_normal_transposed_over_many_priors(
    #     transpose=True, to_write=True
    # )
    # print(len(results_print_out))
    print(f"{(time.time() - start)/60} minutes taken")


