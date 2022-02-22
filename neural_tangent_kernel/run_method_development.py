import sys
import pathlib

import pdb
from utilities import plot_matrix
from analysis_utilities import plot_top_k_curves

from ntk import run_ntk


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
)


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
    calculate_metrics(pred.to_numpy(), true.to_numpy(), mask.to_numpy(), verbose=True)
    pdb.set_trace()
    plot_matrix(pred, "pred")
    save_matrix(pred, TEN_FOLD_CROSS_VALIDATION_ENERGIES)


def buisness_as_normal_transposed():
    """
    This method runs 10-fold cross validation on the 1194x209 Ground Truth matrix.
    With Zeolites as rows (aka using Zeolite priors to predict energies for new Zeolites)
    """
    ground_truth, binary_data = get_ground_truth_energy_matrix(transpose=True)
    pred, true, mask = run_ntk(
        ground_truth, prior="CustomZeolite", metrics_mask=binary_data
    )
    calculate_metrics(pred.to_numpy(), true.to_numpy(), mask.to_numpy(), verbose=True)
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
    calculate_metrics(pred.to_numpy(), ground_truth.to_numpy(), binary_data.to_numpy())


def buisness_as_normal_transposed_over_many_priors():
    """
    This method is an example of how to test a set of priors to see their individual performance.
    """
    ground_truth, binary_data = get_ground_truth_energy_matrix(transpose=False)
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
    results = []
    # Let's do 10-fold cross validation & gather metrics for each prior by itself.
    for prior_to_test in priors_to_test:
        pred, true, mask = run_ntk(
            ground_truth,
            prior="CustomOSDA",
            metrics_mask=binary_data,
            prior_map={prior_to_test: 1.0},
        )
        metrics = calculate_metrics(
            pred.to_numpy(),
            true.to_numpy(),
            mask.to_numpy(),
            verbose=False,
            meta=prior_to_test,
        )
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
    print("here are the results sorted by rmse & top_1_accuracy ", results_print_out)

    ## Test all the priors together...
    full_pred, full_true, mask = run_ntk(
        ground_truth, prior="identity", metrics_mask=binary_data, norm_factor=0.1
    )
    full_metrics = calculate_metrics(
        full_pred.to_numpy(),
        full_true.to_numpy(),
        mask.to_numpy(),
        verbose=False,
    )
    plot_matrix(full_pred, "full_pred")  # , vmin=0, vmax=2)
    plot_matrix(full_true, "full_true")  # , vmin=0, vmax=2)
    plot_top_k_curves(full_metrics["top_20_accuracies"])
    print("full metrics run: ", full_metrics)


if __name__ == "__main__":
    buisness_as_normal()
    # buisness_as_normal_transposed()
    # buisness_as_normal_skinny()
    # buisness_as_normal_transposed_over_many_priors()