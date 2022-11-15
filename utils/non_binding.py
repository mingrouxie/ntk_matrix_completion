from enum import Enum, IntEnum
import numpy as np
import pandas as pd

class NonBinding(IntEnum):
    ROW_MEAN = 1
    SMALL_POS = 2
    LARGE_POS = 3
    MAX_PLUS = 4
    ZERO = 5

def fill_non_bind(mat: pd.DataFrame, nb_type: NonBinding):
    """
    Note that np mean returns NaN if one of the entries is Nan. DataFrame would simply compute the mean without that entry (in both numerator and denominator). We use the latter here.

    Input: Numpy array or DataFrame
    
    Returns: np.array
    """
    if nb_type == NonBinding.ROW_MEAN:
        return pd.DataFrame(mat).apply(lambda row: row.fillna(row.mean()), axis=1).values
    elif nb_type == NonBinding.SMALL_POS:
        return np.nan_to_num(mat, nan=1e-5, posinf=1e-5, neginf=None)
    elif nb_type == NonBinding.LARGE_POS:
        return np.nan_to_num(mat, nan=10, posinf=10, neginf=None)
        # return mat.apply(lambda row: row.fillna(5), axis=1)
        # return mat.apply(lambda row: row.fillna(10), axis=1)
        # return ground_tmatruth.apply(lambda row: row.fillna(30), axis=1)
    elif nb_type == NonBinding.MAX_PLUS:
        return mat.apply(lambda row: row.fillna(row.max() * 1.01), axis=1)
    elif nb_type == NonBinding.ZERO:
        return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=None)
    else:
        raise Exception("Non-binding treatment not recognised:", nb_type)
