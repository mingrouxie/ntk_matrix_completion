import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


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
