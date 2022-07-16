from statistics import mode
import sys
import pathlib
import os
from tabnanny import verbose
import scipy as sp
import pandas as pd
import pdb
from pandas.core.frame import DataFrame

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.metrics import top_k_accuracy_score

from package_matrix import Energy_Type
from path_constants import PERFORMANCE_METRICS

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))
from utilities import (
    plot_matrix,
)

sys.path.insert(
    1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical")
)


def plot_top_k_curves(top_accuracies, method):
    plt.plot(top_accuracies)
    plt.title(f"{method} Accuracy")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(0, len(top_accuracies) + 1, step=5))
    plt.show()
    plt.draw()
    plt.savefig(f"data/output/{method}_accuracies.png", dpi=100)


def get_cosims(true, pred, bypass_epsilon_check=False, filter_value=None):
    cosims = []

    for row_id in range(true.shape[0]):
        i = true[row_id]
        j = pred[row_id]
        if filter_value:
            j = j[i != filter_value]
            i = i[i != filter_value]
        if bypass_epsilon_check or (
            not (np.abs(np.sum(i)) <= 1e-8 or np.abs(np.sum(j)) <= 1e-8)
        ):
            cosims.append(1 - cosine(i, j))

    return cosims


# If metrics_mask is given then filter out all the non-binding results according to mask
def calculate_row_metrics(true, pred, metrics_mask=None):
    cosims = []
    r2_scores = []
    rmse_scores = []
    spearman_scores = []
    min_row_len = true.shape[1]
    for row_id in range(true.shape[0]):
        i = true[row_id]
        j = pred[row_id]
        if metrics_mask is not None:
            m = metrics_mask[row_id]
            j = j[m == 1.0]
            i = i[m == 1.0]
        min_row_len = min(min_row_len, len(i))
        if len(i) <= 1:
            raise ValueError(
                "There exists complete rows in your dataset which are completely unbinding"
                + " or only have one value (which is a no go for R^2 and Spearman..."
            )
        r2_scores.append(r2_score(i, j, multioutput="raw_values"))
        spearman_scores.append([spearmanr(i, j).correlation])
        cosims.append(get_cosims(np.array([i]), np.array([j])))
        rmse_scores.append(
            [math.sqrt(mean_squared_error(i, j, multioutput="raw_values"))]
        )
    if verbose:
        print(
            "Hey, just a heads up the min row_length is "
            + str(min_row_len)
            + ".\n Keep that in mind, how valid is r2_score & spearman_scores? ",
        )
    return cosims, r2_scores, rmse_scores, spearman_scores


def calculate_metrics(
    pred,
    true,
    mask=None,
    energy_type=Energy_Type.BINDING,
    verbose=True,
    meta=None,
    method="top_k_in_top_k",
    to_write=True,
    to_plot=True,
):
    """
    All metrics as calculated by ROW
    """
    print("analysis_utilities/calculate_metrics: truth, pred and mask are of shape", true.shape, pred.shape, mask.shape)
    try:
        cosims, r2_scores, rmse_scores, spearman_scores = calculate_row_metrics(
            true, pred, mask
        )
    except:
        breakpoint()
    if method == "top_n_in_top_k":
        top_1 = []
        top_3 = []
        top_5 = []
        top_20_accuracies = []
        for n in range(1, 1195):
            top_1.append(calculate_top_n_in_top_k_accuracy(true, pred, k=1, n=n))
            top_3.append(calculate_top_n_in_top_k_accuracy(true, pred, k=3, n=n))
            top_5.append(calculate_top_n_in_top_k_accuracy(true, pred, k=5, n=n))
            top_20.append(calculate_top_n_in_top_k_accuracy(true, pred, k=20, n=n))
            if to_plot:
                top_20_accuracies.append(
                    [
                        calculate_top_n_in_top_k_accuracy(true, pred, k=k, n=n)
                        for k in range(1, 21)  # cannot do 0 because of division by k
                    ]
                )
    elif method == "top_k_in_top_k":
        top_1 = calculate_top_n_in_top_k_accuracy(true, pred, k=1)
        top_3 = calculate_top_n_in_top_k_accuracy(true, pred, k=3)
        top_5 = calculate_top_n_in_top_k_accuracy(true, pred, k=5)
        top_20 = calculate_top_n_in_top_k_accuracy(true, pred, k=20)
        if to_plot:
            top_20_accuracies = [
                calculate_top_n_in_top_k_accuracy(true, pred, k=k) for k in range(1, 209)
            ]
    elif method == "top_k":
        top_1 = calculate_top_k_accuracy(true, pred, 1)
        top_3 = calculate_top_k_accuracy(true, pred, 3)
        top_5 = calculate_top_k_accuracy(true, pred, 5)
        top_20 = calculate_top_k_accuracy(true, pred, 20)
        if to_plot:
            top_20_accuracies = [
                calculate_top_k_accuracy(true, pred, k) for k in range(0, 21)
            ]
    results = {
        "cosim": np.mean(np.mean(cosims).round(4)),
        "r2_scores": np.mean(np.mean(r2_scores).round(4)),
        "rmse_scores": np.mean(np.mean(rmse_scores).round(4)),
        "spearman_scores": np.mean(np.mean(spearman_scores).round(4)),
        "top_1_accuracy": np.array(top_1).round(4),
        "top_3_accuracy": np.array(top_3).round(4),
        "top_5_accuracy": np.array(top_5).round(4),
        "top_20_accuracy": np.array(top_20).round(4),
        "top_20_accuracies": top_20_accuracies,
    }
    if verbose:
        print(
            "\ncosim: ",
            np.mean(np.mean(cosims).round(4)),
            "\nr2_scores: ",
            np.mean(np.mean(r2_scores).round(4)),
            "\nrmse_scores: ",
            np.mean(np.mean(rmse_scores).round(4)),
            "\nspearman_scores: ",
            np.mean(np.mean(spearman_scores).round(4)),
            "\ntop_1_accuracy: ",
            np.array(top_1).mean().round(4),
            "\ntop_3_accuracy: ",
            np.array(top_3).mean().round(4),
            "\ntop_5_accuracy: ",
            np.array(top_5).mean().round(4),
            "\ntop_20_accuracy: ",
            np.array(top_20).mean().round(4),
        )
    if to_plot:
        plot_top_k_curves(top_20_accuracies, method)
        if energy_type == Energy_Type.BINDING:
            vmin = -30
            vmax = 5
        else:
            vmin = 16
            vmax = 23
        plot_matrix(true, "regression_truth", vmin=vmin, vmax=vmax)
        plot_matrix(pred, "regression_prediction", vmin=vmin, vmax=vmax)
    if to_write:
        df = pd.DataFrame(top_20_accuracies).T
        df.to_pickle(PERFORMANCE_METRICS)
    return results


def calculate_top_k_accuracy(all_true, ntk_predictions, k, by_row=True, verbose=False):
    """
    Note that -ntk_predictions is used because for both templating and binding energies,
    lower is better, but top_k_accuracy_score looks at, well, the top scores
    """
    if by_row:
        lowest_mask = (all_true.T == all_true.min(axis=1)).T
        _col_nums, top_indices = np.where(lowest_mask)
        pred = -ntk_predictions
    else:
        lowest_mask = all_true == all_true.min(axis=0)
        _col_nums, top_indices = np.where(lowest_mask)
        pred = -ntk_predictions.T
    score = top_k_accuracy_score(top_indices, pred, k=k, labels=range(pred.shape[1]))
    if verbose:
        print(k, score)
    return score


def calculate_top_n_in_top_k_accuracy(
    all_true, ntk_predictions, k, by_row=True, n=None, verbose=False
):
    """
    Note that for both templating and binding energies, lower is better
    """
    ### FOR DEBUGGING ###
    # rng = np.random.default_rng(12345)
    # ntk_predictions = rng.integers(low=0, high=1194, size=ntk_predictions.shape)
    ### FOR DEBUGGING ###
    if not by_row:
        all_true = all_true.T
        ntk_predictions = ntk_predictions.T

    if not n:
        top_indices = np.argsort(all_true, axis=1)[
            :, :k
        ]  # return indices that sorts an array in ascending order e.g. [4,2,1,5]-->[2,1,0,3]
    else:
        top_indices = np.argsort(all_true, axis=1)[:, :n]
    pred = np.argsort(ntk_predictions, axis=1)[:, :k]
    common = [
        len(np.intersect1d(top_indices[row, :], pred[row, :]))
        for row, _ in enumerate(top_indices)
    ]
    if not n:
        score = sum(common) / k / all_true.shape[0]
    else:
        score = sum(common) / n / all_true.shape[0]
    if verbose:
        print(n, k, score)
    return score


# Useful for plotting two histograms of energy distributions together.
def plot_double_histogram(ground_truth, predicted_energies, name="Zeo-1"):
    bins = np.linspace(0, 40, 50)
    plt.hist(ground_truth, bins=bins, alpha=0.5, label="ground_truth")
    plt.hist(predicted_energies, bins=bins, alpha=0.5, label=name)
    plt.title("Ground Truth & Hypothetical Templating Energies for " + name)
    plt.legend(loc="upper right")
    plt.yscale("log")
    plt.show()


# Plot zeolites by volume
def plot_volume(zeolite_priors):
    # bins = np.linspace(0, 40, 50)
    plt.hist(zeolite_priors["volume"], bins=30, label="ground_truth")
    plt.title("Zeolites Binned by Volume ")
    plt.xlabel("Angstroms Cubed")
    plt.ylabel("Zeolite Frequency")
    plt.show()


# This method is to try and determine which priors in osda correlate with predicted
# energies. Pretty caveman stuff. Used it once & not sure if it was actually helpful.
def examine_osda_feature_causes(prediction_ntk):
    from prior import osda_prior

    osda_priors = osda_prior(prediction_ntk.T, normalize=False, identity_weight=0.0)
    feature_to_regression = {}
    # now time to find correlation for each of the potential priors...
    for col in osda_priors.columns:
        feature = osda_priors[col]
        predicted_templating = prediction_ntk.T[0]
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(
            feature, predicted_templating
        )
        feature_to_regression[col] = {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err,
        }
    print(feature_to_regression)
    print("all done")
