import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch.nn as nn
import torch

def masked_loss(y, pred, mask, loss_type='mse'):
    '''Masked regression loss for binding energies'''
    if sum(mask) == 0:
        # all non-binding entries
        return torch.tensor([0.0], requires_grad=True)
    
    if y.dim() == 1:
        y = y.resize(y.shape[0], 1) # TODO: hardcoded
    y_masked = y[mask==1]
    pred_masked = pred[mask==1]
    loss = LOSS_TYPES[loss_type]() # TODO: is this best practice 
    # loss = nn.MSELoss(reduction='mean')

    if np.isnan(loss(y_masked, pred_masked).detach().cpu().numpy()):
        print("[masked_mse_loss] nan present in loss")

    return loss(y_masked, pred_masked)

def mse_loss(y, pred):
    '''MSE loss'''
    
    if y.dim() == 1:
        y = y.resize(y.shape[0], 1) # TODO: hardcoded
    loss = nn.MSELoss(reduction='mean')

    if np.isnan(loss(y, pred).detach().cpu().numpy()):
        print("[mse_loss] nan present in loss")

    return loss(y, pred)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-16):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps # safety. if err is zero grad becomes nan
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    

def rmse_loss(y, pred):
    '''RMSE loss'''
    
    if y.dim() == 1:
        y = y.resize(y.shape[0], 1) # TODO: hardcoded
    loss = RMSELoss()

    if np.isnan(loss(y, pred).detach().cpu().numpy()):
        print("[rmse_loss] nan present in loss")

    return loss(y, pred)


def mae_loss(y, pred):
    '''MAE loss'''

    if y.dim() == 1:
        y = y.resize(y.shape[0], 1) # TODO: hardcoded
    loss = nn.L1Loss()

    if np.isnan(loss(y, pred).detach().cpu().numpy()):
        print("[mae_loss] nan present in loss")

    return loss(y, pred)


def cross_entropy_loss(y, pred):
    if y.dim() == 1:
        y = y.reshape(y.shape[0], 1) # TODO: hardcoded
    # proba = pred.softmax(dim=1)
    loss = nn.CrossEntropyLoss() 

    # if np.isnan(loss(y, proba).detach().cpu().numpy()):
    if np.isnan(loss(y, pred).detach().cpu().numpy()):
        print("[classifier] nan present in loss")

    # return loss(y, proba)
    return loss(y, pred)


def multitask_loss(y, y_preds, mask, loss_type='mse'):
    '''
    Args:
        y: (array) First column of y is energies, second column onwards of y is loads
        y_preds: (tuple) First item is load, second item is energies
        mask: 1D vector

    Returns:
        loading loss, energy loss
    '''

    # print("[multitaskloss debug]", min(y_preds[0]), max(y_preds[0]), min(y_preds[1]), max(y_preds[1]))

    # TODO: hardcoded for loading of variable length 
    energy_loss = masked_loss(y[:,0], y_preds[1], mask, loss_type=loss_type)
    # TODO: right now loss_type is not specified in train.py nor config file. Also, need separate loss_types for energy and for loading TODO TODO TODO 
    if y[:,1:].shape[1] > 1:
        load_loss = cross_entropy_loss(y[:,1:], y_preds[0])
    else:
        load_loss = mse_loss(y[:,1:], y_preds[0])
    return load_loss, energy_loss 

LOSS_TYPES = {
    'mse': nn.MSELoss,
    'mae': nn.L1Loss,
    'rmse': RMSELoss,
    'ce': nn.CrossEntropyLoss
}