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

sys.path.append("/home/mrx")
from general.utils.figures import single_scatter, single_hist, format_axs, single_line

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
    

# Plot predicted energy distribution versus actual (only for binding)
def plot_load_dist(y, pred, save=False):
    '''Plot distribution of normalized loadings'''
    fig, axs = plt.subplots(figsize=(10,5))
    colors = ['#00429d', '#93003a']
    axs.hist(y.values, bins=100, label='Truth', color=colors[0], alpha=0.7)
    axs.hist(pred.values, bins=100, label='Prediction', color=colors[1], alpha=0.7)
    fig.legend(fontsize=15, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)
    axs = format_axs(axs, 30, 30, 3, f"Normalized loading", "Count", 30, 30)

    if save:
        fig.savefig(save, dpi=300, bbox_inches = "tight")


def plot_energy_parity(y, pred, save=False):
    '''Plot parity plot for energy'''
    fig, axs = single_scatter(
        x=y,
        y=pred,
        xlabel='Truth (kJ/mol Si)', 
        ylabel='Prediction (kJ/mol Si)', 
        limits={
            "x": [min(y), 5],
            "y": [min(pred), 5]
        }, 
        colorbar="Count", tight_layout=True, savefig=None
    )
    
    if save:
        fig.savefig(save, dpi=300, bbox_inches = "tight")


def plot_load_parity(y, pred, save=False):
    '''Plot parity plot for normalized loading'''
    fig, axs = single_scatter(
        x=y,
        y=pred,
        xlabel='Truth', 
        ylabel='Prediction', 
        # limits={
        #     "x": [min(y), 5],
        #     "y": [min(pred), 5]
        # }, 
        colorbar="Count", tight_layout=True, savefig=None
    )
    
    if save:
        fig.savefig(save, dpi=300, bbox_inches = "tight")


def plot_heatmap(y, pred, index="SMILES", columns="Zeolite", values="Binding (SiO2)", save=False):
    '''Plots heatmap of values. Assumes that data can pivot into a full matrix'''
    y_mat = y.reset_index().pivot(values=values, index=index, columns=columns)
    pred_mat = pred.reset_index().pivot(values=values, index=index, columns=columns)

    fig, axs = plt.subplots(1, 2, figsize=(5,10), sharex=True, sharey=True)
    cmaps = 'inferno'
    pcm = axs[0].pcolormesh(y_mat, cmap=cmaps, vmin=-25, vmax=0)
    pcm = axs[1].pcolormesh(pred_mat, cmap=cmaps, vmin=-25, vmax=0)
    cb = fig.colorbar(pcm, ax=axs[:], shrink=0.6, )
    cb.set_label("Binding energy (kJ/mol Si)", size=20)
    cb.ax.tick_params(labelsize=20)
    ylabels = ['OSDA', None]
    for idx, ax in enumerate(axs):
        ax = format_axs(ax, 20, 20, 1, "Zeolite", ylabels[idx], 20, 20)

    if save:
        fig.savefig(save, dpi=300, bbox_inches = "tight")


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
    
def plot_errors(y, pred, mask, col, save=False):
    '''Plot distribution of errors'''
    err_test = y[mask.exists==1][col] - pred[mask.exists==1][col]
    err_test_unmasked = y[col] - pred[col]
    print("maximum error", err_test.max(), err_test_unmasked.max())

    fig, axs = plt.subplots(2, 2, figsize=(18,10))
    colors = ['#00429d', '#93003a']
    axs[0,1].hist(err_test,bins=100, color=colors[0])
    axs[1,1].hist(err_test_unmasked,bins=100, color=colors[0]);
    for i in [0,1]:
        for j in [0,1]:
            ax = format_axs(axs[i,j], 20, 20, 2, col, 'Count', 20, 20)
    if save:
        fig.savefig(save, dpi=300, bbox_inches = "tight")


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="analysis_utilities")
    parser.add_argument("--config", help="Config file", required=True)
    args = parser.parse_args()
    kwargs = args.__dict__
    op = kwargs['config']

    train_indices = pd.read_csv(os.path.join(op, "pred_train_indices.csv"), index_col=0)
    train_mask = pd.read_csv(os.path.join(op, "pred_train_mask.csv"), index_col=0).set_index(pd.MultiIndex.from_frame(train_indices))
    train_y = pd.read_csv(os.path.join(op, "pred_train_ys.csv"), index_col=0).set_index(pd.MultiIndex.from_frame(train_indices))
    train_pred = pd.read_csv(os.path.join(op, "pred_train_y_preds.csv"), index_col=0).set_index(pd.MultiIndex.from_frame(train_indices))

    test_indices = pd.read_csv(os.path.join(op, "pred_test_indices.csv"), index_col=0)
    test_mask = pd.read_csv(os.path.join(op, "pred_test_mask.csv"), index_col=0).set_index(pd.MultiIndex.from_frame(test_indices))
    test_y = pd.read_csv(os.path.join(op, "pred_test_ys.csv"), index_col=0).set_index(pd.MultiIndex.from_frame(test_indices))
    test_pred = pd.read_csv(os.path.join(op, "pred_test_y_preds.csv"), index_col=0).set_index(pd.MultiIndex.from_frame(test_indices))
    model = torch.load(os.path.join(op, "model.pt"))

    # TODO: check dimensions lol for the nb what is it

    # TODO: what about training set prediction >.< can we predict that too

    def make_plots(y, pred, mask, label="test"):
        # BINDING ONLY ENERGY
        plot_energy_dist(y.loc[mask.exists==1]['Binding (SiO2)'], pred.loc[mask.exists==1]['Binding (SiO2)'], save=os.path.join(op, f"{label}_e_dist.png"))
        plot_energy_parity(y.loc[mask.exists==1]['Binding (SiO2)'], pred.loc[mask.exists==1]['Binding (SiO2)'], save=os.path.join(op, f"{label}_e_par.png"))

        # BINDING ONLY LOADING
        plot_load_dist(y.loc[mask.exists==1]['loading_norm'], pred.loc[mask.exists==1]['loading_norm'], save=os.path.join(op, f"{label}_l_dist_b.png"))
        plot_load_parity(y.loc[mask.exists==1]['loading_norm'], pred.loc[mask.exists==1]['loading_norm'], save=os.path.join(op, f"{label}_l_par_b.png"))

        # NON BINDING ONLY LOADING
        plot_load_dist(y.loc[mask.exists!=1]['loading_norm'], pred.loc[mask.exists!=1]['loading_norm'], save=os.path.join(op, f"{label}_l_dist_nb.png"))
        plot_load_parity(y.loc[mask.exists!=1]['loading_norm'], pred.loc[mask.exists!=1]['loading_norm'], save=os.path.join(op, f"{label}_l_par_nb.png"))

        # HEATMAP
        plot_heatmap(y, pred, values="Binding (SiO2)", save=os.path.join(op, f"{label}_heatmap_e.png"))
        plot_heatmap(y, pred, values="loading_norm", save=os.path.join(op, f"{label}_heatmap_l.png"))
        plot_heatmap(
            y.loc[mask.exists==1].loc[mask.exists==1]['loading_norm'], 
            pred.loc[mask.exists==1].loc[mask.exists==1]['loading_norm'], 
            values="loading_norm", save=os.path.join(op, f"{label}_heatmap_l_b.png"))

        # ERRORS
        plot_errors(y, pred, mask, col="Binding (SiO2)", save=os.path.join(op, f"{label}_err_e.png"))
        plot_errors(y, pred, mask, col="loading_norm", save=os.path.join(op, f"{label}_err_l.png"))


    make_plots(train_y, train_pred, train_mask, label="train")
    make_plots(test_y, test_pred, test_mask, label="test")

    # LOSS CURVES
    plot_loss_curves(model["epoch_losses"], model["val_losses"])

    print("Analysis finished")

# NOTES ON OUTPUT FILES

# # -rw-rw-r-- 1 mrx mrx  26259538 Jan 31 18:14 X_test_scaled.pkl
# multiindex SMILEs then Zeolite, cols are 0 to whatever (34 in this case)
# # -rw-rw-r-- 1 mrx mrx 242678898 Jan 31 18:14 X_train_scaled.pkl
# multiindex SMILEs then Zeolite, cols are 0 to whatever (34 in this case)
# # -rw-rw-r-- 1 mrx mrx      1637 Jan 31 18:08 args.yaml
# args.keys() = dict_keys(['batch_size', 'config', 'device', 'energy_scaler', 'energy_type', 'epochs', 'gpu', 'ignore_train', 'input_scaler', 'l_sizes', 'load_scaler', 'mask', 'model', 'optimizer', 'osda_prior_file', 'osda_prior_map', 'other_prior_to_concat', 'output', 'prior_method', 'prior_treatment', 'scheduler', 'seed', 'sieved_file', 'split_type', 'truth', 'tune', 'zeolite_prior_file', 'zeolite_prior_map'])
# # -rw-rw-r-- 1 mrx mrx      1429 Jan 31 18:13 input_scaling.json
# scalar_type
# mean in []
# var in []
# # -rw-rw-r-- 1 mrx mrx   1531419 Jan 31 18:14 mask_test.pkl
# SMILEs is index, col is Zeolite and exists
# # -rw-rw-r-- 1 mrx mrx  14359274 Jan 31 18:14 mask_train.pkl
# SMILEs is index, col is Zeolite and exists
# # -rw-rw-r-- 1 mrx mrx        85 Jan 31 18:13 truth_energy_scaling.json
# scaler_type is str
# scale is in []
# min is in []
# # -rw-rw-r-- 1 mrx mrx        72 Jan 31 18:13 truth_load_scaling.json
# # -rw-rw-r-- 1 mrx mrx   1867547 Jan 31 18:13 truth_test_scaled.pkl
# multiindex SMILEs then Zeolite, cols are Binding (SiO2) and loading_norm
# # -rw-rw-r-- 1 mrx mrx  17561137 Jan 31 18:13 truth_train_scaled.pkl
# multiindex SMILEs then Zeolite, cols are Binding (SiO2) and loading_norm
# # -rw-rw-r-- 1 mrx mrx  14508 Feb  1 16:05 test_indices.csv
# SMILES and Zeolite
# # -rw-rw-r-- 1 mrx mrx   1898 Feb  1 16:05 test_mask.csv
# exists
# # -rw-rw-r-- 1 mrx mrx   6282 Feb  1 16:05 test_y_preds.csv
# Binding (SiO2) and loading_norm
# # -rw-rw-r-- 1 mrx mrx   4133 Feb  1 16:05 test_ys.csv
# Binding (SiO2) and loading_norm
