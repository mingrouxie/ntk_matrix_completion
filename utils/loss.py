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

def classifier_loss(y, pred):
    '''"Classifier" loss. Since normalized loadings are continuous instead of integer numbers, this ends up being a MSE loss'''
    y_rs = y.reshape(y.shape[0], 1) # TODO: hardcoded
    loss = nn.MSELoss(reduction='mean')

    if np.isnan(loss(y_rs, pred).detach().numpy()):
        print("[classifier] nan present in loss")

    return loss(y_rs, pred)
    # proba = pred.softmax(dim=1)
    # return nn.CrossEntropyLoss(y, proba)

def regressor_loss(y, pred, mask):
    '''Masked regression loss for binding energies'''
    if sum(mask) == 0:
        # all non-binding entries
        return torch.tensor([0.0], requires_grad=True)
        
    y_rs = y.resize(y.shape[0], 1) # TODO: hardcoded
    y_masked = y_rs[mask==1]
    pred_masked = pred[mask==1]
    loss = nn.MSELoss(reduction='mean')

    if np.isnan(loss(y_masked, pred_masked).detach().numpy()):
        print("[regressor] nan present in loss")

    return loss(y_masked, pred_masked)

def multitask_loss(y, y_preds, mask):
    '''
    Args:
        y: First column of y is load, second column of y is energies
        y_preds: (tuple) First item is load, second item is energies
        mask: 1D vector

    Returns:
        "classification" loss, regression loss
    '''
    c_loss = classifier_loss(y[:,0], y_preds[0])
    r_loss = regressor_loss(y[:,1], y_preds[1], mask)
    return c_loss, r_loss 