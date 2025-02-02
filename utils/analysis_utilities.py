from statistics import mode
import sys
import pathlib
import os
from tabnanny import verbose
import scipy as sp
import numpy as np
import pandas as pd
import pdb
import math
import argparse
import torch
import json

from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.metrics import top_k_accuracy_score

from ntk_matrix_completion.utils.package_matrix import Energy_Type
from ntk_matrix_completion.utils.path_constants import PERFORMANCE_METRICS, OUTPUT_DIR

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))
from utils.utilities import (
    plot_matrix,
)

sys.path.insert(
    1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical")
)

sys.path.append("/home/mrx")
from general.utils.figures import single_scatter, single_hist, format_axs, single_line

def plot_top_k_curves(top_accuracies, method):
    fig, axs = plt.subplots(figsize=(10, 10))
    axs.plot(top_accuracies)
    axs.set_title(f"{method} Accuracy")
    axs.set_xlabel("K")
    axs.set_ylabel("Accuracy")
    axs.set_xticks(np.arange(0, len(top_accuracies) + 1, step=5))
    fig.savefig(OUTPUT_DIR + f"/{method}_accuracies.png", dpi=100)


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
        cosims.append(get_cosims(np.array([i]), np.array([j])))
        rmse_scores.append(
            [math.sqrt(mean_squared_error(i, j, multioutput="raw_values"))]
        )
        if len(i) <= 1:
            print(
                "Whole row is unbinding or only has one value. R^2 and Spearman cannot be computed. Row value:",
                true[row_id][0],
            )
            continue
        r2_scores.append(r2_score(i, j, multioutput="raw_values"))
        spearman_scores.append([spearmanr(i, j).correlation])

    if verbose:
        print(
            "Hey, just a heads up the min row_length is "
            + str(min_row_len)
            + ".\n Keep that in mind, how valid is r2_score & spearman_scores? ",
        )
    print(len(cosims))
    print(len(r2_scores))
    print(len(rmse_scores))
    print(len(spearman_scores))
    return cosims, r2_scores, rmse_scores, spearman_scores


def calculate_metrics(
    pred,
    true,
    mask=None,
    energy_type=Energy_Type.BINDING,
    verbose=True,
    meta=None,
    method="top_k_in_top_k",
    to_plot=True,
    top_x_accuracy=20,
    top_x_accuracies_file=PERFORMANCE_METRICS,
):
    """
    All metrics as calculated by ROW.
    """
    print(
        "analysis_utilities/calculate_metrics: truth, pred and mask are of shape",
        true.shape,
        pred.shape,
        mask.shape,
    )

    cosims, r2_scores, rmse_scores, spearman_scores = calculate_row_metrics(
        true, pred, mask
    )

    if method == "top_n_in_top_k":
        top_1 = []
        top_3 = []
        top_5 = []
        top_x_accuracies = []
        for n in range(1, 1195):  # TODO: this number needs to be changed
            top_1.append(calculate_top_n_in_top_k_accuracy(true, pred, k=1, n=n))
            top_3.append(calculate_top_n_in_top_k_accuracy(true, pred, k=3, n=n))
            top_5.append(calculate_top_n_in_top_k_accuracy(true, pred, k=5, n=n))
            top_20.append(calculate_top_n_in_top_k_accuracy(true, pred, k=20, n=n))
            if top_x_accuracy:
                top_x_accuracies.append(
                    [
                        calculate_top_n_in_top_k_accuracy(true, pred, k=k, n=n)
                        for k in range(1, top_x_accuracy)  # cannot do 0 because of division by k
                    ]
                )

    elif method == "top_k_in_top_k":
        top_1 = calculate_top_n_in_top_k_accuracy(true, pred, k=1)
        top_3 = calculate_top_n_in_top_k_accuracy(true, pred, k=3)
        top_5 = calculate_top_n_in_top_k_accuracy(true, pred, k=5)
        top_20 = calculate_top_n_in_top_k_accuracy(true, pred, k=20)
        if top_x_accuracy:
            top_x_accuracies = [
                calculate_top_n_in_top_k_accuracy(true, pred, k=k)
                for k in range(1, top_x_accuracy)
            ]

    elif method == "top_k":
        top_1 = calculate_top_k_accuracy(true, pred, 1)
        top_3 = calculate_top_k_accuracy(true, pred, 3)
        top_5 = calculate_top_k_accuracy(true, pred, 5)
        top_20 = calculate_top_k_accuracy(true, pred, 20)
        if top_x_accuracy:
            top_x_accuracies = [
                calculate_top_k_accuracy(true, pred, k) for k in range(0, top_x_accuracy)
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
    }
    if top_x_accuracy:
        results["top_x_accuracies"] = top_x_accuracies

    if verbose:
        print(
            "\ncosim: ",
            results["cosim"],
            "\nr2_scores: ",
            results["r2_scores"],
            "\nrmse_scores: ",
            results["rmse_scores"],
            "\nspearman_scores: ",
            results["spearman_scores"],
            "\ntop_1_accuracy: ",
            results["top_1_accuracy"],
            "\ntop_3_accuracy: ",
            results["top_3_accuracy"],
            "\ntop_5_accuracy: ",
            results["top_5_accuracy"],
            "\ntop_20_accuracy: ",
            results["top_20_accuracy"],
        )

    if to_plot:
        plot_top_k_curves(top_x_accuracies, method)
        if energy_type == Energy_Type.BINDING:
            vmin = -30
            vmax = 5
        else:
            vmin = 16
            vmax = 23
        plot_matrix(true, "regression_truth", vmin=vmin, vmax=vmax)
        plot_matrix(pred, "regression_prediction", vmin=vmin, vmax=vmax)

    if top_x_accuracy:
        df = pd.DataFrame(top_x_accuracies).T
    if top_x_accuracies_file:
        df.to_pickle(top_x_accuracies_file)

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


def plot_energy_dist(y, pred, save=False):
    '''Plot distribution of energies'''
    fig, axs = plt.subplots(figsize=(10,5))
    colors = ['#00429d', '#93003a']
    axs.hist(y, bins=100, label='Truth', color=colors[0], alpha=0.7)
    axs.hist(pred, bins=100, label='Prediction', color=colors[1], alpha=0.7)
    fig.legend(fontsize=15, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)
    axs = format_axs(axs, 30, 30, 3, "Binding energy (kJ/ mol Si)", "Count", 30, 30)
    
    if save:
        fig.savefig(save, dpi=300, bbox_inches = "tight")
    plt.close()
    

# Plot predicted energy distribution versus actual (only for binding)
def plot_load_dist(y, pred, save=False):
    '''Plot distribution of normalized loadings'''

    # TODO: idk if this will work
    if y.shape[1] > 1: # multiclass classification rather than regression
        y = y.T.idxmax(0) # load_1, or load_norm_1
        pred = pred.T.idxmax(0)

    fig, axs = plt.subplots(figsize=(10,5))
    colors = ['#00429d', '#93003a']
    axs.hist(y.values, bins=100, label='Truth', color=colors[0], alpha=0.7)
    axs.hist(pred.values, bins=100, label='Prediction', color=colors[1], alpha=0.7)
    fig.legend(fontsize=15, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)
    axs = format_axs(axs, 30, 30, 3, f"Normalized loading", "Count", 30, 30)

    if save:
        fig.savefig(save, dpi=300, bbox_inches = "tight")
    plt.close()


def plot_energy_parity(y, pred, save=False):
    '''Plot parity plot for energy'''
    min_e = min(min(y), min(pred))
    max_e = max(max(y), max(pred))
    fig, axs = single_scatter(
        x=y,
        y=pred,
        xlabel='Truth (kJ/mol Si)', 
        ylabel='Prediction (kJ/mol Si)', 
        limits={
            "x": [min_e, max_e],
            "y": [min_e, max_e]
        }, 
        colorbar="Count", tight_layout=True, savefig=None
    )
    
    if save:
        fig.savefig(save, dpi=300, bbox_inches = "tight")
    plt.close()


def plot_load_parity(y, pred, save=False):
    '''Plot parity plot for normalized loading'''

    # TODO: need to take out the actual value so we can make parity plots
    if y.shape[1] > 1: # multiclass classification rather than regression
        y = y.T.idxmax(0) # load_1, or load_norm_1
        pred = pred.T.idxmax(0)

    min_l = min(min(y.values), min(pred.values))
    max_l = max(max(y.values), max(pred.values))
    fig, axs = single_scatter(
        x=y.values,
        y=pred.values,
        xlabel='Truth', 
        ylabel='Prediction', 
        limits={
            "x": [min_l, max_l],
            "y": [min_l, max_l]
        }, 
        colorbar="Count", tight_layout=True, savefig=None
    )
    
    if save:
        fig.savefig(save, dpi=300, bbox_inches = "tight")
    plt.close()


def plot_heatmap(y, pred, index="SMILES", columns="Zeolite", values="Binding (SiO2)", save=False):
    '''Plots heatmap of values. Assumes that data can pivot into a full matrix'''
    y_mat = y.reset_index().pivot(values=values, index=index, columns=columns)
    pred_mat = pred.reset_index().pivot(values=values, index=index, columns=columns)
    breakpoint()
    vmin = min(y[values].values.min(), y[values].values.min())
    vmax = max(pred[values].values.max(), pred[values].values.max())

    fig, axs = plt.subplots(1, 2, figsize=(5,10), sharex=True, sharey=True)
    cmaps = 'inferno'
    pcm = axs[0].pcolormesh(y_mat, cmap=cmaps, vmin=vmin, vmax=vmax)
    pcm = axs[1].pcolormesh(pred_mat, cmap=cmaps, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(pcm, ax=axs[:], shrink=0.6, )
    cb.set_label(values, size=20)
    cb.ax.tick_params(labelsize=20)
    ylabels = [index, None]
    for idx, ax in enumerate(axs):
        ax = format_axs(ax, 20, 20, 1, columns, ylabels[idx], 20, 20)

    if save:
        fig.savefig(save, dpi=300, bbox_inches = "tight")
    plt.close()


def plot_topktopk(y, pred, mask, index="SMILES", columns="Zeolite", values="Binding (SiO2)", save=False):
    '''Plot top k in top k accuracy. Assumes that data can pivot into a full matrix'''
    y_mat = y.reset_index().pivot(values=values, index=index, columns=columns)
    pred_mat = pred.reset_index().pivot(values=values, index=index, columns=columns)

    metrics = calculate_metrics(
        y_mat, 
        pred_mat, 
        mask, 
        verbose=True,
        method="top_k_in_top_k",
        to_plot=False,
        top_x_accuracy=y_mat.shape[1]
        )

    fig, axs = plt.subplots(figsize=(5,5))
    axs.plot(range(1,209), metrics['top_x_accuracy'], 'k')
    axs = format_axs(axs, 20, 20, 2, "k", "Top k in top k accuracy", 20, 20)
    
    if save:
        fig.savefig(save, dpi=300, bbox_inches = "tight")
    plt.close()
    
def plot_errors(y, pred, mask, col, save=False):
    '''Plot distribution of errors'''
    err_test = y[mask.exists==1][col] - pred[mask.exists==1][col]
    err_test_unmasked = y[col] - pred[col]
    print("maximum error", err_test.max(), err_test_unmasked.max())

    fig, axs = plt.subplots(2, 1, figsize=(18,10))
    colors = ['#00429d', '#93003a']
    axs[0].hist(err_test,bins=100, color=colors[0])
    axs[1].hist(err_test_unmasked,bins=100, color=colors[0]);
    for idx, label in enumerate([" error (masked)", " error (unmasked)"]):
        ax = format_axs(axs[idx], 20, 20, 2, col+label, 'Count', 20, 20)
    if save:
        fig.savefig(save, dpi=300, bbox_inches = "tight")
    plt.close()


def plot_loss_curves(train_losses, val_losses, save=False):
    '''training versus test curve'''
    # there is another thing called the learning curve where the x-axis is the number of data points in the train dataset

    fig, axs = plt.subplots(figsize=(10,10))
    colors = ['#00429d', '#93003a']
    axs.plot(train_losses, label='Train loss', color=colors[0])
    axs.plot(val_losses, label='Validation loss', color=colors[1])
    axs = format_axs(
        axs, 30, 30, 3, "Epoch", "Loss", 30, 30
    )
    fig.legend(loc="upper right", borderaxespad=3, fontsize=20)
    if save:
        fig.savefig(save, dpi=300, bbox_inches = "tight")



# FEATURE IMPORTANCE TODO

# COMPUTE TEMPLATING ENERGY AFTER PREDICTING TODO

# THAT'S RIGHT, DO WE WANT TO CHECK ON THE SCIENCE PAPER DATASET AND ITS TEMPLATING ENERGY TODO

# ALSO THE ZACH+SCIENCE DATASET, IF POSSIBLE TODO 

def rescale(data, scaler):
    '''Rescale data back. TODO: check that multiple columns work'''
    if scaler['scaler_type'] == 'minmax':
        return data * (np.array(scaler['data_max']) - np.array(scaler['data_min'])) + np.array(scaler['data_min'])
    elif scaler['scaler_type'] == 'standard':
        return data * np.sqrt(np.array(scaler['std'])) + np.array(scaler['mean'])


def make_plots(y, pred, mask, kwargs, label="test", energy=True, load=True):
    # BINDING ONLY ENERGY
    if energy:
        plot_energy_dist(y.loc[mask.exists==1][kwargs["energy_label"]], pred.loc[mask.exists==1][kwargs["energy_label"]], save=os.path.join(op, f"{label}_e_dist.png"))
        plot_energy_parity(y.loc[mask.exists==1][kwargs["energy_label"]], pred.loc[mask.exists==1][kwargs["energy_label"]], save=os.path.join(op, f"{label}_e_par.png"))

    # BINDING ONLY LOADING
    if load:
        plot_load_dist(y.loc[mask.exists==1][kwargs["load_label"]], pred.loc[mask.exists==1][kwargs["load_label"]], save=os.path.join(op, f"{label}_l_dist_b.png"))
        plot_load_parity(y.loc[mask.exists==1][kwargs["load_label"]], pred.loc[mask.exists==1][kwargs["load_label"]], save=os.path.join(op, f"{label}_l_par_b.png"))

    # NON BINDING ONLY LOADING
    if load:
        plot_load_dist(y.loc[mask.exists!=1][kwargs["load_label"]], pred.loc[mask.exists!=1][kwargs["load_label"]], save=os.path.join(op, f"{label}_l_dist_nb.png"))
        plot_load_parity(y.loc[mask.exists!=1][kwargs["load_label"]], pred.loc[mask.exists!=1][kwargs["load_label"]], save=os.path.join(op, f"{label}_l_par_nb.png"))

    # HEATMAP
    if energy:
        plot_heatmap(y, pred, values=kwargs["energy_label"], save=os.path.join(op, f"{label}_heatmap_e.png"))
        plot_heatmap(
            y.loc[mask.exists==1], 
            pred.loc[mask.exists==1], 
            values=kwargs["energy_label"], save=os.path.join(op, f"{label}_heatmap_e_b.png"))
    if load:
        plot_heatmap(y, pred, values=kwargs["load_label"], save=os.path.join(op, f"{label}_heatmap_l.png"))
        plot_heatmap(
            y.loc[mask.exists==1], 
            pred.loc[mask.exists==1], 
            values=kwargs["load_label"], save=os.path.join(op, f"{label}_heatmap_l_b.png"))

    # ERRORS
    if energy:
        print("energy error")
        plot_errors(y, pred, mask, col=kwargs["energy_label"], save=os.path.join(op, f"{label}_err_e.png"))
    if load:
        print("loading error")
        plot_errors(y, pred, mask, col=kwargs["load_label"], save=os.path.join(op, f"{label}_err_l.png"))


if __name__ == '__main__':
    import yaml
    import re
    parser = argparse.ArgumentParser(description="analysis_utilities")
    parser.add_argument("--args", help="YAML file containing output directory path to process", required=True)
    parser.add_argument("--local", help="If true, alters the directory paths from Engaging to local", default=False, action='store_true')
    parser.add_argument("--load_label", help="single, load or load_norm", default="single", required=True)
    parser.add_argument("--energy_label", help="Binding (SiO2) for now", default="Binding (SiO2)", required=True)
    kwargs = parser.parse_args().__dict__

    # preprocess
    with open(kwargs['args'], "rb") as file:
        op = yaml.load(file, Loader=yaml.Loader)['output']
        if kwargs['local']:
            LOCAL_ROOT = '/home/mrx/projects/affinity_pool/'
            ENG_ROOT = '/pool001/mrx/projects/affinity/'
            op = re.sub(ENG_ROOT, LOCAL_ROOT, op, count=0)
    if kwargs["load_label"] == "single":
        kwargs["load_label"] = ["Loading"]
    elif kwargs["load_label"] == "load":
        kwargs["load_label"] = [f"load_{i}" for i in range(0,23)]
    elif kwargs["load_label"] == "load_norm":
        kwargs["load_label"] = [f"load_norm_{i}" for i in range(0,49)]
    
    print(f"Processing output folder {op} locally ({kwargs['local']})")

    # read files
    print("Reading files")
    train_indices = pd.read_csv(os.path.join(op, "pred_train_indices.csv"), index_col=0)
    train_mask = pd.read_csv(os.path.join(op, "pred_train_mask.csv"), index_col=0).set_index(pd.MultiIndex.from_frame(train_indices))
    train_y = pd.read_csv(os.path.join(op, "pred_train_ys.csv"), index_col=0).set_index(pd.MultiIndex.from_frame(train_indices))
    train_pred = pd.read_csv(os.path.join(op, "pred_train_y_preds.csv"), index_col=0).set_index(pd.MultiIndex.from_frame(train_indices))

    print("Reading indices")
    test_indices = pd.read_csv(os.path.join(op, "pred_test_indices.csv"), index_col=0)
    test_mask = pd.read_csv(os.path.join(op, "pred_test_mask.csv"), index_col=0).set_index(pd.MultiIndex.from_frame(test_indices))
    test_y = pd.read_csv(os.path.join(op, "pred_test_ys.csv"), index_col=0).set_index(pd.MultiIndex.from_frame(test_indices))
    test_pred = pd.read_csv(os.path.join(op, "pred_test_y_preds.csv"), index_col=0).set_index(pd.MultiIndex.from_frame(test_indices))
    model = torch.load(os.path.join(op, "model.pt"))

    print("Reading scalings")
    # TODO scaling
    try:
        with open(os.path.join(op, "input_scaling.json"), "r") as f:
            input_scaler = json.load(f)
        # TODO: scale the inputs back or something
    except FileNotFoundError:
        print("No input scaling")
    try:
        with open(os.path.join(op, "truth_load_scaling.json"), "r") as f:
            l_scaler = json.load(f)
        train_y[kwargs["load_label"]] = rescale(train_y[kwargs["load_label"]], l_scaler)
        train_pred[kwargs["load_label"]] = rescale(train_pred[kwargs["load_label"]], l_scaler)
        test_y[kwargs["load_label"]] = rescale(test_y[kwargs["load_label"]], l_scaler)
        test_pred[kwargs["load_label"]] = rescale(test_pred[kwargs["load_label"]], l_scaler)
    except FileNotFoundError:
        print("No loading scaling")
    try:
        with open(os.path.join(op, "truth_energy_scaling.json"), "r") as f:
            e_scaler = json.load(f)
        train_y[kwargs["energy_label"]] = rescale(train_y[kwargs["energy_label"]], e_scaler)
        train_pred[kwargs["energy_label"]] = rescale(train_pred[kwargs["energy_label"]], e_scaler)
        test_y[kwargs["energy_label"]] = rescale(test_y[kwargs["energy_label"]], e_scaler)
        test_pred[kwargs["energy_label"]] = rescale(test_pred[kwargs["energy_label"]], e_scaler)
    except FileNotFoundError:
        print("No energy scaling")

    # plot
    breakpoint()
    make_plots(train_y, train_pred, train_mask, kwargs, label="train")
    make_plots(test_y, test_pred, test_mask, kwargs, label="test")

    # LOSS CURVES
    plot_loss_curves(model["epoch_losses"], model["val_losses"])

    print("Analysis finished")

