import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch.nn as nn
import torch


def masked_rmse(truth, pred, mask):
    """
    Compute RMSE only for binding entries. 
    Mask should be 1 for binding, 0 for non-binding
    Inputs truth, pred and mask should be of the same shape
    """
    # print(f"[masked_rmse] {mask.sum()} binding out of {mask.size} entries")
    assert np.all(truth.shape == pred.shape), "[masked_rmse] Shapes do not match"
    assert np.all(truth.shape == mask.shape), "[masked_rmse] Shapes do not match"
    
    rmse = sqrt(mean_squared_error(truth[mask==1], pred[mask==1]))
    # print(f"[masked_rmse] RMSE={rmse}")
    return rmse

# def classifier_loss(y, pred):
#     '''"Classifier" loss. Since normalized loadings are continuous instead of integer numbers, this ends up being a MSE loss'''
    # if y.dim() == 1:
    #     y = y.reshape(y.shape[0], 1) # TODO: hardcoded
#     loss = nn.MSELoss(reduction='mean')

#     if np.isnan(loss(y, pred).detach().numpy()):
#         print("[classifier] nan present in loss")

#     return loss(y, pred)
#     # proba = pred.softmax(dim=1)
#     # return nn.CrossEntropyLoss(y, proba)

def mse_loss(y, pred, mask):
    '''Masked regression loss for binding energies'''
    if sum(mask) == 0:
        # all non-binding entries
        return torch.tensor([0.0], requires_grad=True)
    
    if y.dim() == 1:
        y = y.resize(y.shape[0], 1) # TODO: hardcoded
    y_masked = y[mask==1]
    pred_masked = pred[mask==1]
    loss = nn.MSELoss(reduction='mean')

    if np.isnan(loss(y_masked, pred_masked).detach().numpy()):
        print("[regressor] nan present in loss")

    return loss(y_masked, pred_masked)

def cross_entropy_loss(y, pred):
    if y.dim() == 1:
        y = y.reshape(y.shape[0], 1) # TODO: hardcoded
    proba = pred.softmax(dim=1)
    loss = nn.CrossEntropyLoss()

    if np.isnan(loss(y, proba).detach().numpy()):
        print("[classifier] nan present in loss")

    return loss(y, proba)


def multitask_loss(y, y_preds, mask):
    '''
    Args:
        y: First column of y is load, second column of y is energies
        y_preds: (tuple) First item is load, second item is energies
        mask: 1D vector

    Returns:
        classfication loss, regression loss
    '''
    # c_loss = classifier_loss(y[:,0], y_preds[0])
    # c_loss = cross_entropy_loss(y[:,0], y_preds[0])
    # r_loss = mse_loss(y[:,1], y_preds[1], mask)

    # looks like first column of truth is meant to be loading
    # second column is energy. but old code also did be then load wth
    # ah the models do load then be sigh
    r_loss = mse_loss(y[:,0], y_preds[1], mask)
    c_loss = cross_entropy_loss(y[:,1:], y_preds[0])

    # but my debug file has energy first then loading oops
    # TODO oops this doesn't allow for cross entropy and variable number of columns 
    return c_loss, r_loss 