from utils.path_constants import (
    OSDA_CONFORMER_PRIOR_FILE_SIEVED,
    OSDA_PRIOR_FILE,
    TEN_FOLD_CROSS_VALIDATION_ENERGIES,
    ZEOLITE_PRIOR_SELECTION_FILE,
    OUTPUT_DIR,
    OSDA_CONFORMER_PRIOR_FILE,
    OSDA_CONFORMER_PRIOR_FILE_CLIPPED,
    BINDING_GROUND_TRUTH,
)
from utils.analysis_utilities import calculate_metrics
from utils.utilities import (
    save_matrix,
)
from utils.package_matrix import (
    Energy_Type,
    get_ground_truth_energy_matrix,
    make_skinny,
    unmake_skinny,
)
import sys
import pathlib
import pandas as pd
import io
import os
from sklearn.model_selection import train_test_split

import pdb
from utils.utilities import plot_matrix
from utils.analysis_utilities import plot_top_k_curves
from configs.weights import ZEOLITE_PRIOR_LOOKUP, OSDA_PRIOR_LOOKUP
from ntk import run_ntk, skinny_ntk_sampled_not_sliced, SplitType
from features.precompute_osda_priors import smile_to_property
from utils.random_seeds import SUBSTRATE_PRIOR_SELECTION_SEED
import numpy as np
import time

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))

# TODO: change zeolite to substrate...

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
    # shuffle data and put aside a test set
    (
        ground_truth_train,
        ground_truth_test,
        binary_data_train,
        binary_data_test,
        row_indices_train,
        row_indices_test,
    ) = train_test_split(
        ground_truth,
        binary_data,
        row_indices,
        test_size=0.1,
        random_state=SUBSTRATE_PRIOR_SELECTION_SEED,
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
        print("here")
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
    for best_feature in [
        # "rmse_scores",
        # "spearman_scores",
        "top_1_accuracy",
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

        breakpoint()
    # TODO: not implemented yet - show what the test accuracy is T.T
    # TODO: what about 50% RMSE and 50% top-k in top-k accuracy?
    # TODO: Genetic algorithm? Worth it?
