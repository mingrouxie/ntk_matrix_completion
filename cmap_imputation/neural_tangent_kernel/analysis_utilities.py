from statistics import mode
import sys
import pathlib
import os

from pandas.core.frame import DataFrame
from prior import make_prior
from cli import validate_zeolite_inputs

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.metrics import top_k_accuracy_score

from package_matrix import Energy_Type

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))
from utilities import (
    plot_matrix,
)

sys.path.insert(
    1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical")
)


def plot_top_k_curves(top_accuracies):
    plt.plot(top_accuracies)
    plt.title("Top K Accuracy for Zeolites per OSDA")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(0, len(top_accuracies) + 1, step=5))
    plt.show()
    plt.draw()
    plt.savefig("top_k_accuracies.png", dpi=100)


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
    print(
        "Hey, just a heads up the min row_length is "
        + str(min_row_len)
        + ".\n Keep that in mind, how valid is r2_score & spearman_scores? ",
    )
    return cosims, r2_scores, rmse_scores, spearman_scores


def calculate_metrics(pred, true, mask=None, energy_type=Energy_Type.TEMPLATING):
    """
    All metrics as calculated by ROW
    """
    cosims, r2_scores, rmse_scores, spearman_scores = calculate_row_metrics(
        true, pred, mask
    )
    top_1 = calculate_top_k_accuracy(true, pred, 1)
    top_3 = calculate_top_k_accuracy(true, pred, 3)
    top_5 = calculate_top_k_accuracy(true, pred, 5)
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
        top_1.round(4),
        "\ntop_3_accuracy: ",
        top_3.round(4),
        "\ntop_5_accuracy: ",
        top_5.round(4),
    )
    top_20_accuracies = [calculate_top_k_accuracy(true, pred, k) for k in range(0, 21)]
    plot_top_k_curves(top_20_accuracies)
    if energy_type == Energy_Type.BINDING:
        vmin = -30
        vmax = 5
    else:
        vmin = 16
        vmax = 23
    plot_matrix(true, "regression_truth", vmin=vmin, vmax=vmax)
    plot_matrix(pred, "regression_prediction", vmin=vmin, vmax=vmax)


def calculate_top_k_accuracy(all_true, ntk_predictions, k, by_row=True):
    if by_row:
        lowest_mask = (all_true.T == all_true.min(axis=1)).T
        _col_nums, top_indices = np.where(lowest_mask)
        pred = -ntk_predictions
    else:
        lowest_mask = all_true == all_true.min(axis=0)
        _col_nums, top_indices = np.where(lowest_mask)
        pred = -ntk_predictions.T
    return top_k_accuracy_score(top_indices, pred, k=k, labels=range(pred.shape[1]))
