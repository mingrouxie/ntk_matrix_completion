import io
import os
import pathlib
import pdb
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.analysis_utilities import calculate_metrics, plot_top_k_curves
from ntk import SplitType, run_ntk, skinny_ntk_sampled_not_sliced
from utils.package_matrix import (Energy_Type, get_ground_truth_energy_matrix,
                            make_skinny, unmake_skinny)
from utils.path_constants import (BINDING_GROUND_TRUTH, OSDA_CONFORMER_PRIOR_FILE,
                            OSDA_CONFORMER_PRIOR_FILE_CLIPPED,
                            OSDA_CONFORMER_PRIOR_FILE_SIEVED, OSDA_PRIOR_FILE,
                            OUTPUT_DIR, TEN_FOLD_CROSS_VALIDATION_ENERGIES,
                            ZEOLITE_PRIOR_SELECTION_FILE)
from features.precompute_osda_priors import smile_to_property
from utils.utilities import plot_matrix, save_matrix
from configs.weights import OSDA_PRIOR_LOOKUP, ZEOLITE_PRIOR_LOOKUP

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))


def ntk_cv(
    split_type=SplitType.NAIVE_SPLITS,
    debug=False,
    prune_index=None,
    osda_prior_file=OSDA_PRIOR_FILE,
    prior_type="CustomOSDAVector",  # "CustomConformerOSDA"
    energy_type=Energy_Type.BINDING,
    metric_method="top_k_in_top_k",
    verbose=True,
    use_eigenpro=False,
    to_write=True,
    to_plot=False,
):
    """
    This method runs 10-fold cross validation on the 1194x209 Ground Truth matrix.
    With OSDAs as rows (aka using OSDA priors to predict energies for new OSDAs).

    split_type: How to split data for the 10-fold
    debug: If True, prints the SMILES of the "top-performing" OSDAs by squared error
    prune_index: if specified, sieves the ground truth matrix for specified rows
    osda_prior_file: source file for OSDA priors
    prior_type: method of creating the prior
    energy_type: Type of ground truth (default is binding energy)
    use_eigenpro: If True, uses eigenpro. Not yet implemented
    to_write: If True, writes analysis results to file
    to_plot: If True, pltos analysis results
    """
    ground_truth, binary_data = get_ground_truth_energy_matrix(
        prune_index=prune_index, energy_type=energy_type
    )
    pred, true, mask = run_ntk(
        ground_truth,
        prior=prior_type,
        metrics_mask=binary_data,
        use_eigenpro=use_eigenpro,
        split_type=split_type,
        osda_prior_file=osda_prior_file,
    )
    print(
        "[NTK_CV] Returned NTK predictions are of shape",
        pred.shape,
        true.shape,
        mask.shape,
    )
    # TODO: implement option to compute RMSE and topk in topk for each fold
    calculate_metrics(
        pred.to_numpy(),
        true.to_numpy(),
        mask.to_numpy(),
        verbose=verbose,
        method=metric_method,
        to_write=to_write,
        to_plot=to_plot,
    )
    plot_matrix(pred, "pred", vmin=-30, vmax=5)
    if debug:
        # Quick investigation of top 10 OSDAs
        top_osdas = ((true - pred) ** 2).T.sum().sort_values()[:10]
        print(top_osdas.index)
    save_matrix(pred, TEN_FOLD_CROSS_VALIDATION_ENERGIES)
    print("ntk_cv finished")


def ntk_cv_transposed(
    energy_type=Energy_Type.BINDING,
    prior_type="CustomZeolite",  # "CustomZeoliteEmbeddings"
    metric_method="top_k_in_top_k",
    verbose=True,
    to_write=True,
    to_plot=False,
):
    """
    This method runs 10-fold cross validation on the 1194x209 Ground Truth matrix.
    With Zeolites as rows (aka using Zeolite priors to predict energies for new Zeolites).

    energy_type: Type of ground truth (default is binding energy)
    prior_type: method of creating the prior
    """
    ground_truth, binary_data = get_ground_truth_energy_matrix(
        transpose=True, energy_type=energy_type
    )
    pred, true, mask = run_ntk(ground_truth, prior=prior_type, metrics_mask=binary_data)
    results = calculate_metrics(
        pred.to_numpy(),
        true.to_numpy(),
        mask.to_numpy(),
        energy_type=Energy_Type.BINDING,
        verbose=verbose,
        method=metric_method,
        to_write=to_write,
        to_plot=to_plot,
    )
    pdb.set_trace()

    print("ntk_cv_transpose finished")


def ntk_cv_skinny():
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


def ntk_cv_compare_priors(
    priors_to_test,
    prior_type,
    transpose=False,
    to_write=False,
    to_plot=True,
    metric_method="top_k_in_top_k",
    verbose=False,
):
    """
    This method tests a set of priors to see their individual performance.
    The NTK is run with 10-fold CV with each individual prior, and then once more with all the priors together

    transpose: If True, matrix is transposed
    priors_to_test: iterable of priors to test
    to_write: If True, results are written to file
    to_plot: If True, results are plotted
    verbose: If True, prints analysis results

    Returns:
    CSV-readable iterable of results, which includes sorted per-prior and identity prior results, and the full results as well
    """
    ground_truth, binary_data = get_ground_truth_energy_matrix(transpose=transpose)
    prior = prior_type
    priors_to_test = priors_to_test
    results = []

    # 10-fold cross validation & metrics for each prior by itself.
    for prior_to_test in priors_to_test:
        pred, true, mask = run_ntk(
            ground_truth,
            prior=prior_type,
            metrics_mask=binary_data,
            prior_map={prior_to_test: 1.0},
        )

        metrics = calculate_metrics(
            pred.to_numpy(),
            true.to_numpy(),
            mask.to_numpy(),
            verbose=False,
            meta=prior_to_test,
            method=metric_method,
        )
        print(prior_to_test, metrics["top_5_accuracy"])
        results.append((prior_to_test, metrics))

    # Metrics for the baseline: identity
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
        method=metric_method,
    )
    results.append(("identity", identity_metrics))

    # sort results
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
    # print(
    #     "[NTK_COMPARE_PRIORS] Results sorted by rmse & top_1_accuracy ", results_print_out)

    # Test all the priors together
    prior_map_full = dict([(p, 1.0) for p in priors_to_test])
    full_pred, full_true, mask = run_ntk(
        ground_truth, 
        prior=prior_type,
        metrics_mask=binary_data,
        prior_map=prior_map_full,
        metrics_mask=binary_data, 
        norm_factor=0.1
    )
    full_metrics = calculate_metrics(
        full_pred.to_numpy(),
        full_true.to_numpy(),
        mask.to_numpy(),
        verbose=verbose,
        method=metric_method,
    )
    results.append(("full", full_metrics))

    if to_plot:
        plot_matrix(full_pred, "full_pred")  # , vmin=0, vmax=2)
        plot_matrix(full_true, "full_true")  # , vmin=0, vmax=2)
        plot_top_k_curves(full_metrics["top_20_accuracies"])

    if to_write:
        df = pd.read_csv(io.StringIO("\n".join(results_print_out)), sep="\t")
        df.to_csv("./ntk_cv_compare_priors.csv")

    return results_print_out


if __name__ == "__main__":
    start = time.time()
    sieved_priors_index = pd.read_pickle(OSDA_CONFORMER_PRIOR_FILE_CLIPPED).index

    # ground_truth_index = pd.read_pickle(BINDING_GROUND_TRUTH).index
    # ground_truth_index= ground_truth_index.drop('CC[N+]12C[N@]3C[N@@](C1)C[N@](C2)C3')
    # precomputed_prior = pd.read_pickle(OSDA_PRIOR_FILE)
    # def vector_explode(x): return pd.Series(x['getaway'])
    # exploded_precomputed_prior = precomputed_prior.apply(vector_explode, axis=1)
    # double_sieved_priors_index = exploded_precomputed_prior.reindex(sieved_priors_index).dropna().index
    sieved_priors_index.name = "SMILES"
    # pdb.set_trace()
    ntk_cv(
        split_type=SplitType.OSDA_ISOMER_SPLITS,
        debug=True,
        prune_index=sieved_priors_index,
        osda_prior_file=OSDA_CONFORMER_PRIOR_FILE_CLIPPED,
    )
    pdb.set_trace()

    # ntk_cv()
    # ntk_cv_transposed()
    # ntk_cv_transposed(
    #     energy_type=Energy_Type.BINDING,
    #     prior="CustomZeolite",
    #     verbose=True,
    #     to_write=False,
    #     to_plot=True
    # )
    # skinny_ntk_sampled_not_sliced()
    # ntk_cv_skinny()
    # ntk_cv()
    # ntk_cv_transposed()

    # osda_priors_to_test = [
    #     "mol_weights",
    #     "volume",
    #     "normalized_num_rotatable_bonds",
    #     "formal_charge",
    #     "asphericity",
    #     "inertial_shape_factor",
    #     "spherocity",
    #     "eccentricity",
    #     "gyration_radius",
    #     "pmi1",
    #     "pmi2",
    #     "pmi3",
    #     "npr1",
    #     "npr2",
    #     "free_sas",
    #     "bertz_ct",
    # ]
    # results_print_out = ntk_cv_compare_priors(
    #     priors_to_test=zeolite_priors_to_test, prior_type="CustomOSDA", transpose=True, to_write=True
    # )

    # zeolite_priors_to_test = [  # TODO: Mingrou temporary, just to test the 0D
    #     "cell_birth-of-top-1-0D-feature",
    #     "cell_death-of-top-1-0D-feature",
    #     "cell_persistence-of-top-1-0D-feature",
    #     "cell_birth-of-top-2-0D-feature",
    #     "cell_death-of-top-2-0D-feature",
    #     "cell_persistence-of-top-2-0D-feature",
    #     "cell_birth-of-top-3-0D-feature",
    #     "cell_death-of-top-3-0D-feature",
    #     "cell_persistence-of-top-3-0D-feature",
    #     "supercell_birth-of-top-1-0D-feature",
    #     "supercell_death-of-top-1-0D-feature",
    #     "supercell_persistence-of-top-1-0D-feature",
    #     "supercell_birth-of-top-2-0D-feature",
    #     "supercell_death-of-top-2-0D-feature",
    #     "supercell_persistence-of-top-2-0D-feature",
    #     "supercell_birth-of-top-3-0D-feature",
    #     "supercell_death-of-top-3-0D-feature",
    #     "supercell_persistence-of-top-3-0D-feature",
    # ]
    # OR
    # zeolite_priors_to_test = ZEOLITE_PRIOR_LOOKUP.keys()
    # results_print_out = ntk_cv_compare_priors(
    #     priors_to_test=zeolite_priors_to_test,
    #     prior_type="CustomZeolite",  # "CustomZeoliteEmbeddings"
    #     transpose=True,
    #     to_write=True,
    # )

    # print(len(results_print_out))
    print(f"{(time.time() - start)/60} minutes taken")
