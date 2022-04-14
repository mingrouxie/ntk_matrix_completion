import numpy as np
from numpy.linalg import norm
import pandas as pd
import pdb
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb
from functools import lru_cache
from auto_tqdm import tqdm
import os
import pathlib
import math
from scipy.sparse import csc_matrix
import scipy as sp
from functools import lru_cache

from sklearn.preprocessing import normalize, OneHotEncoder

VALID_METHODS = {
    "identity",
    "CustomOSDA",
    "OldCustomOSDA",
    "CustomZeolite",
    "random",
    "CustomOSDAandZeolite",
    "CustomOSDAandZeoliteAsRows",
    "CustomOSDAVector",
    "ManualZeolite",
    "CustomOSDAandZeoliteAsSparseMatrix",
}

ZEOLITE_PRIOR_LOOKUP = {
    "a": 1.0,
    "b": 1.0,
    "c": 1.0,
    "alpha": 1.0,
    "betta": 1.0,
    "gamma": 1.0,
    "volume": 1.0,
    # "rdls": 1.0,
    "framework_density": 1.0,
    # "td_10": 1.0,
    # "td": 1.0,
    "ring_size_0": 1.0,
    "ring_size_1": 1.0,
    "ring_size_2": 1.0,
    "included_sphere_diameter": 1.0,
    "diffused_sphere_diameter_a": 1.0,
    "diffused_sphere_diameter_b": 1.0,
    "diffused_sphere_diameter_c": 1.0,
    # "accessible_volume": 1.0,
    # "N_1": 1.0,
    # "N_2": 1.0,
    # "N_3": 1.0,
    # "N_4": 1.0,
    # "N_5": 1.0,
    # "N_6": 1.0,
    # "N_7": 1.0,
    # "N_8": 1.0,
    # "N_9": 1.0,
    # "N_10": 1.0,
    # "N_11": 1.0,
    # "N_12": 1.0,
    "ring_size_3": 1.0,
    "ring_size_4": 1.0,
    "ring_size_5": 1.0,
    "ring_size_6": 1.0,
    # "diffused_sphere_diameter_max_abc": 1.0,
    # "num_atoms": 1.0,
    # "cell_birth-of-top-1-1D-feature": 1.0,
    # "cell_birth-of-top-2-1D-feature": 1.0,
    # "cell_birth-of-top-3-1D-feature": 1.0,
    # "cell_birth-of-top-1-2D-feature": 1.0,
    # "cell_birth-of-top-2-2D-feature": 1.0,
    # "cell_birth-of-top-3-2D-feature": 1.0,
    # "a_projected-cell_birth-of-top-1-1D-feature": 1.0,
    # "a_projected-cell_birth-of-top-2-1D-feature": 1.0,
    # "a_projected-cell_birth-of-top-3-1D-feature": 1.0,
    # "b_projected-cell_birth-of-top-1-1D-feature": 1.0,
    # "b_projected-cell_birth-of-top-2-1D-feature": 1.0,
    # "b_projected-cell_birth-of-top-3-1D-feature": 1.0,
    # "c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    # "c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    # "c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    # "a+b_projected-cell_birth-of-top-1-1D-feature": 1.0,
    # "a+b_projected-cell_birth-of-top-2-1D-feature": 1.0,
    # "a+b_projected-cell_birth-of-top-3-1D-feature": 1.0,
    # "a-b_projected-cell_birth-of-top-1-1D-feature": 1.0,
    # "a-b_projected-cell_birth-of-top-2-1D-feature": 1.0,
    # "a-b_projected-cell_birth-of-top-3-1D-feature": 1.0,
    # "a+c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    # "a+c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    # "a+c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    # "a-c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    # "a-c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    # "a-c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    # "b+c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    # "b+c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    # "b+c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    # "b-c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    # "b-c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    # "b-c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    # "a+b+c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    # "a+b+c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    # "a+b+c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    # "a+b-c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    # "a+b-c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    # "a+b-c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    # "a-b+c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    # "a-b+c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    # "a-b+c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    # "a-b-c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    # "a-b-c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    # "a-b-c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    # "all-cell-projections_birth-of-top-1-1D-feature": 1.0,
    # "all-cell-projections_birth-of-top-2-1D-feature": 1.0,
    # "all-cell-projections_birth-of-top-3-1D-feature": 1.0,
    # "supercell_birth-of-top-1-1D-feature": 1.0,
    # "supercell_birth-of-top-2-1D-feature": 1.0,
    # "supercell_birth-of-top-3-1D-feature": 1.0,
    # "supercell_birth-of-top-1-2D-feature": 1.0,
    # "supercell_birth-of-top-2-2D-feature": 1.0,
    # "supercell_birth-of-top-3-2D-feature": 1.0,
    # "a_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    # "a_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    # "a_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    # "b_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    # "b_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    # "b_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    # "c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    # "c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    # "c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    # "a+b_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    # "a+b_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    # "a+b_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    # "a-b_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    # "a-b_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    # "a-b_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    # "a+c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    # "a+c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    # "a+c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    # "a-c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    # "a-c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    # "a-c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    # "b+c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    # "b+c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    # "b+c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    # "b-c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    # "b-c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    # "b-c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    # "a+b+c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    # "a+b+c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    # "a+b+c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    # "a+b-c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    # "a+b-c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    # "a+b-c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    # "a-b+c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    # "a-b+c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    # "a-b+c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    # "a-b-c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    # "a-b-c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    # "a-b-c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    # "all-supercell-projections_birth-of-top-1-1D-feature": 1.0,
    # "all-supercell-projections_birth-of-top-2-1D-feature": 1.0,
    # "all-supercell-projections_birth-of-top-3-1D-feature": 1.0,
    # "cell_death-of-top-1-1D-feature": 1.0,
    # "cell_death-of-top-2-1D-feature": 1.0,
    # "cell_death-of-top-3-1D-feature": 1.0,
    # "cell_death-of-top-1-2D-feature": 1.0,
    # "cell_death-of-top-2-2D-feature": 1.0,
    # "cell_death-of-top-3-2D-feature": 1.0,
    # "a_projected-cell_death-of-top-1-1D-feature": 1.0,
    # "a_projected-cell_death-of-top-2-1D-feature": 1.0,
    # "a_projected-cell_death-of-top-3-1D-feature": 1.0,
    # "b_projected-cell_death-of-top-1-1D-feature": 1.0,
    # "b_projected-cell_death-of-top-2-1D-feature": 1.0,
    # "b_projected-cell_death-of-top-3-1D-feature": 1.0,
    # "c_projected-cell_death-of-top-1-1D-feature": 1.0,
    # "c_projected-cell_death-of-top-2-1D-feature": 1.0,
    # "c_projected-cell_death-of-top-3-1D-feature": 1.0,
    # "a+b_projected-cell_death-of-top-1-1D-feature": 1.0,
    # "a+b_projected-cell_death-of-top-2-1D-feature": 1.0,
    # "a+b_projected-cell_death-of-top-3-1D-feature": 1.0,
    # "a-b_projected-cell_death-of-top-1-1D-feature": 1.0,
    # "a-b_projected-cell_death-of-top-2-1D-feature": 1.0,
    # "a-b_projected-cell_death-of-top-3-1D-feature": 1.0,
    # "a+c_projected-cell_death-of-top-1-1D-feature": 1.0,
    # "a+c_projected-cell_death-of-top-2-1D-feature": 1.0,
    # "a+c_projected-cell_death-of-top-3-1D-feature": 1.0,
    # "a-c_projected-cell_death-of-top-1-1D-feature": 1.0,
    # "a-c_projected-cell_death-of-top-2-1D-feature": 1.0,
    # "a-c_projected-cell_death-of-top-3-1D-feature": 1.0,
    # "b+c_projected-cell_death-of-top-1-1D-feature": 1.0,
    # "b+c_projected-cell_death-of-top-2-1D-feature": 1.0,
    # "b+c_projected-cell_death-of-top-3-1D-feature": 1.0,
    # "b-c_projected-cell_death-of-top-1-1D-feature": 1.0,
    # "b-c_projected-cell_death-of-top-2-1D-feature": 1.0,
    # "b-c_projected-cell_death-of-top-3-1D-feature": 1.0,
    # "a+b+c_projected-cell_death-of-top-1-1D-feature": 1.0,
    # "a+b+c_projected-cell_death-of-top-2-1D-feature": 1.0,
    # "a+b+c_projected-cell_death-of-top-3-1D-feature": 1.0,
    # "a+b-c_projected-cell_death-of-top-1-1D-feature": 1.0,
    # "a+b-c_projected-cell_death-of-top-2-1D-feature": 1.0,
    # "a+b-c_projected-cell_death-of-top-3-1D-feature": 1.0,
    # "a-b+c_projected-cell_death-of-top-1-1D-feature": 1.0,
    # "a-b+c_projected-cell_death-of-top-2-1D-feature": 1.0,
    # "a-b+c_projected-cell_death-of-top-3-1D-feature": 1.0,
    # "a-b-c_projected-cell_death-of-top-1-1D-feature": 1.0,
    # "a-b-c_projected-cell_death-of-top-2-1D-feature": 1.0,
    # "a-b-c_projected-cell_death-of-top-3-1D-feature": 1.0,
    # "all-cell-projections_death-of-top-1-1D-feature": 1.0,
    # "all-cell-projections_death-of-top-2-1D-feature": 1.0,
    # "all-cell-projections_death-of-top-3-1D-feature": 1.0,
    # "supercell_death-of-top-1-1D-feature": 1.0,
    # "supercell_death-of-top-2-1D-feature": 1.0,
    # "supercell_death-of-top-3-1D-feature": 1.0,
    # "supercell_death-of-top-1-2D-feature": 1.0,
    # "supercell_death-of-top-2-2D-feature": 1.0,
    # "supercell_death-of-top-3-2D-feature": 1.0,
    # "a_projected-supercell_death-of-top-1-1D-feature": 1.0,
    # "a_projected-supercell_death-of-top-2-1D-feature": 1.0,
    # "a_projected-supercell_death-of-top-3-1D-feature": 1.0,
    # "b_projected-supercell_death-of-top-1-1D-feature": 1.0,
    # "b_projected-supercell_death-of-top-2-1D-feature": 1.0,
    # "b_projected-supercell_death-of-top-3-1D-feature": 1.0,
    # "c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    # "c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    # "c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    # "a+b_projected-supercell_death-of-top-1-1D-feature": 1.0,
    # "a+b_projected-supercell_death-of-top-2-1D-feature": 1.0,
    # "a+b_projected-supercell_death-of-top-3-1D-feature": 1.0,
    # "a-b_projected-supercell_death-of-top-1-1D-feature": 1.0,
    # "a-b_projected-supercell_death-of-top-2-1D-feature": 1.0,
    # "a-b_projected-supercell_death-of-top-3-1D-feature": 1.0,
    # "a+c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    # "a+c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    # "a+c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    # "a-c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    # "a-c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    # "a-c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    # "b+c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    # "b+c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    # "b+c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    # "b-c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    # "b-c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    # "b-c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    # "a+b+c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    # "a+b+c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    # "a+b+c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    # "a+b-c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    # "a+b-c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    # "a+b-c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    # "a-b+c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    # "a-b+c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    # "a-b+c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    # "a-b-c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    # "a-b-c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    # "a-b-c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    # "all-supercell-projections_death-of-top-1-1D-feature": 1.0,
    # "all-supercell-projections_death-of-top-2-1D-feature": 1.0,
    # "all-supercell-projections_death-of-top-3-1D-feature": 1.0,
    # "cell_persistence-of-top-1-1D-feature": 1.0,
    # "cell_persistence-of-top-2-1D-feature": 1.0,
    # "cell_persistence-of-top-3-1D-feature": 1.0,
    # "cell_persistence-of-top-1-2D-feature": 1.0,
    # "cell_persistence-of-top-2-2D-feature": 1.0,
    # "cell_persistence-of-top-3-2D-feature": 1.0,
    # "a_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    # "a_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    # "a_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    # "b_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    # "b_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    # "b_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    # "c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    # "c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    # "c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    # "a+b_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    # "a+b_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    # "a+b_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    # "a-b_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    # "a-b_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    # "a-b_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    # "a+c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    # "a+c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    # "a+c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    # "a-c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    # "a-c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    # "a-c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    # "b+c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    # "b+c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    # "b+c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    # "b-c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    # "b-c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    # "b-c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    # "a+b+c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    # "a+b+c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    # "a+b+c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    # "a+b-c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    # "a+b-c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    # "a+b-c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    # "a-b+c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    # "a-b+c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    # "a-b+c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    # "a-b-c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    # "a-b-c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    # "a-b-c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    # "all-cell-projections_persistence-of-top-1-1D-feature": 1.0,
    # "all-cell-projections_persistence-of-top-2-1D-feature": 1.0,
    # "all-cell-projections_persistence-of-top-3-1D-feature": 1.0,
    # "supercell_persistence-of-top-1-1D-feature": 1.0,
    # "supercell_persistence-of-top-2-1D-feature": 1.0,
    # "supercell_persistence-of-top-3-1D-feature": 1.0,
    # "supercell_persistence-of-top-1-2D-feature": 1.0,
    # "supercell_persistence-of-top-2-2D-feature": 1.0,
    # "supercell_persistence-of-top-3-2D-feature": 1.0,
    # "a_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    # "a_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    # "a_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    # "b_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    # "b_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    # "b_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    # "c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    # "c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    # "c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    # "a+b_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    # "a+b_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    # "a+b_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    # "a-b_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    # "a-b_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    # "a-b_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    # "a+c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    # "a+c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    # "a+c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    # "a-c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    # "a-c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    # "a-c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    # "b+c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    # "b+c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    # "b+c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    # "b-c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    # "b-c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    # "b-c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    # "a+b+c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    # "a+b+c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    # "a+b+c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    # "a+b-c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    # "a+b-c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    # "a+b-c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    # "a-b+c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    # "a-b+c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    # "a-b+c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    # "a-b-c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    # "a-b-c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    # "a-b-c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    # "all-supercell-projections_persistence-of-top-1-1D-feature": 1.0,
    # "all-supercell-projections_persistence-of-top-2-1D-feature": 1.0,
    # "all-supercell-projections_persistence-of-top-3-1D-feature": 1.0,
    '0': 1.0,
    '1': 1.0,
    '2': 1.0,
    '3': 1.0,
    '4': 1.0,
    '5': 1.0,
    '6': 1.0,
    '7': 1.0,
    '8': 1.0,
    '9': 1.0,
    '10': 1.0,
    '11': 1.0,
    '12': 1.0,
    '13': 1.0,
    '14': 1.0,
    '15': 1.0,
    '16': 1.0,
    '17': 1.0,
    '18': 1.0,
    '19': 1.0,
    '20': 1.0,
    '21': 1.0,
    '22': 1.0,
    '23': 1.0,
    '24': 1.0,
    '25': 1.0,
    '26': 1.0,
    '27': 1.0,
    '28': 1.0,
    '29': 1.0,
    '30': 1.0,
    '31': 1.0,
    '32': 1.0,
    '33': 1.0,
    '34': 1.0,
    '35': 1.0,
    '36': 1.0,
    '37': 1.0,
    '38': 1.0,
    '39': 1.0,
    '40': 1.0,
    '41': 1.0,
    '42': 1.0,
    '43': 1.0,
    '44': 1.0,
    '45': 1.0,
    '46': 1.0,
    '47': 1.0,
    '48': 1.0,
    '49': 1.0,
    '50': 1.0,
    '51': 1.0,
    '52': 1.0,
    '53': 1.0,
    '54': 1.0,
    '55': 1.0,
    '56': 1.0,
    '57': 1.0,
    '58': 1.0,
    '59': 1.0,
    '60': 1.0,
    '61': 1.0,
    '62': 1.0,
    '63': 1.0,
    '64': 1.0,
    '65': 1.0,
    '66': 1.0,
    '67': 1.0,
    '68': 1.0,
    '69': 1.0,
    '70': 1.0,
    '71': 1.0,
    '72': 1.0,
    '73': 1.0,
    '74': 1.0,
    '75': 1.0,
    '76': 1.0,
    '77': 1.0,
    '78': 1.0,
    '79': 1.0,
    '80': 1.0,
    '81': 1.0,
    '82': 1.0,
    '83': 1.0,
    '84': 1.0,
    '85': 1.0,
    '86': 1.0,
    '87': 1.0,
    '88': 1.0,
    '89': 1.0,
    '90': 1.0,
    '91': 1.0,
    '92': 1.0,
    '93': 1.0,
    '94': 1.0,
    '95': 1.0,
    '96': 1.0,
    '97': 1.0,
    '98': 1.0,
    '99': 1.0,
    '100': 1.0,
    '101': 1.0,
    '102': 1.0,
    '103': 1.0,
    '104': 1.0,
    '105': 1.0,
    '106': 1.0,
    '107': 1.0,
    '108': 1.0,
    '109': 1.0,
    '110': 1.0,
    '111': 1.0,
    '112': 1.0,
    '113': 1.0,
    '114': 1.0,
    '115': 1.0,
    '116': 1.0,
    '117': 1.0,
    '118': 1.0,
    '119': 1.0,
    '120': 1.0,
    '121': 1.0,
    '122': 1.0,
    '123': 1.0,
    '124': 1.0,
    '125': 1.0,
    '126': 1.0,
    '127': 1.0,
    '128': 1.0,
    '129': 1.0,
    '130': 1.0,
    '131': 1.0,
    '132': 1.0,
    '133': 1.0,
    '134': 1.0,
    '135': 1.0,
    '136': 1.0,
    '137': 1.0,
    '138': 1.0,
    '139': 1.0,
    '140': 1.0,
    '141': 1.0,
    '142': 1.0,
    '143': 1.0,
    '144': 1.0,
    '145': 1.0,
    '146': 1.0,
    '147': 1.0,
    '148': 1.0,
    '149': 1.0,
    '150': 1.0,
    '151': 1.0,
    '152': 1.0,
    '153': 1.0,
    '154': 1.0,
    '155': 1.0,
    '156': 1.0,
    '157': 1.0,
    '158': 1.0,
    '159': 1.0,
    '160': 1.0,
    '161': 1.0,
    '162': 1.0,
    '163': 1.0,
    '164': 1.0,
    '165': 1.0,
    '166': 1.0,
    '167': 1.0,
    '168': 1.0,
    '169': 1.0,
    '170': 1.0,
    '171': 1.0,
    '172': 1.0,
    '173': 1.0,
    '174': 1.0,
    '175': 1.0,
    '176': 1.0,
    '177': 1.0,
    '178': 1.0,
    '179': 1.0,
    '180': 1.0,
    '181': 1.0,
    '182': 1.0,
    '183': 1.0,
    '184': 1.0,
    '185': 1.0,
    '186': 1.0,
    '187': 1.0,
    '188': 1.0,
    '189': 1.0,
    '190': 1.0,
    '191': 1.0,
    '192': 1.0,
    '193': 1.0,
    '194': 1.0,
    '195': 1.0,
    '196': 1.0,
    '197': 1.0,
    '198': 1.0,
    '199': 1.0,
    '200': 1.0,
    '201': 1.0,
    '202': 1.0,
    '203': 1.0,
    '204': 1.0,
    '205': 1.0,
    '206': 1.0,
    '207': 1.0,
    '208': 1.0,
    '209': 1.0,
    '210': 1.0,
    '211': 1.0,
    '212': 1.0,
    '213': 1.0,
    '214': 1.0,
    '215': 1.0,
    '216': 1.0,
    '217': 1.0,
    '218': 1.0,
    '219': 1.0,
    '220': 1.0,
    '221': 1.0,
    '222': 1.0,
    '223': 1.0,
    '224': 1.0,
    '225': 1.0,
    '226': 1.0,
    '227': 1.0,
    '228': 1.0,
    '229': 1.0,
    '230': 1.0,
    '231': 1.0,
    '232': 1.0,
    '233': 1.0,
    '234': 1.0,
    '235': 1.0,
    '236': 1.0,
    '237': 1.0,
    '238': 1.0,
    '239': 1.0,
    '240': 1.0,
    '241': 1.0,
    '242': 1.0,
    '243': 1.0,
    '244': 1.0,
    '245': 1.0,
    '246': 1.0,
    '247': 1.0,
    '248': 1.0,
    '249': 1.0,
    '250': 1.0,
    '251': 1.0,
    '252': 1.0,
    '253': 1.0,
    '254': 1.0,
    '255': 1.0,
    '256': 1.0,
    '257': 1.0,
    '258': 1.0,
    '259': 1.0,
    '260': 1.0,
    '261': 1.0,
    '262': 1.0,
    '263': 1.0,
    '264': 1.0,
    '265': 1.0,
    '266': 1.0,
    '267': 1.0,
    '268': 1.0,
    '269': 1.0,
    '270': 1.0,
    '271': 1.0,
    '272': 1.0,
    '273': 1.0,
    '274': 1.0,
    '275': 1.0,
    '276': 1.0,
    '277': 1.0,
    '278': 1.0,
    '279': 1.0,
    '280': 1.0,
    '281': 1.0,
    '282': 1.0,
    '283': 1.0,
    '284': 1.0,
    '285': 1.0,
    '286': 1.0,
    '287': 1.0,
    '288': 1.0,
    '289': 1.0,
    '290': 1.0,
    '291': 1.0,
    '292': 1.0,
    '293': 1.0,
    '294': 1.0,
    '295': 1.0,
    '296': 1.0,
    '297': 1.0,
    '298': 1.0,
    '299': 1.0,
    '300': 1.0,
    '301': 1.0,
    '302': 1.0,
    '303': 1.0,
    '304': 1.0,
    '305': 1.0,
    '306': 1.0,
    '307': 1.0,
    '308': 1.0,
    '309': 1.0,
}

OSDA_PRIOR_LOOKUP = {
    "mol_weights": 1.0,
    "volume": 1.0,
    "normalized_num_rotatable_bonds": 1.0,
    "formal_charge": 1.0,
    "asphericity": 1.0,
    "eccentricity": 1.0,
    "inertial_shape_factor": 1.0,
    "spherocity": 1.0,
    "gyration_radius": 1.0,
    "pmi1": 1.0,
    "pmi2": 1.0,
    "pmi3": 1.0,
    "npr1": 1.0,
    "npr2": 1.0,
    "free_sas": 1.0,
    "bertz_ct": 1.0,
}
from path_constants import (
    ZEOLITE_PRIOR_FILE,
    OSDA_PRIOR_FILE,
    OSDA_HYPOTHETICAL_PRIOR_FILE,
    OSDA_CONFORMER_PRIOR_FILE,
    ZEO_1_PRIOR,
    PERSISTENCE_ZEOLITE_PRIOR_FILE,
    GCNN_ZEOLITE_PRIOR_FILE
)


ZEOLITE_PRIOR_MAP = {
    "*CTH": "CTH",
    "*MRE": "MRE",
    "*PCS": "PCS",
    "*STO": "STO",
    "*UOE": "UOE",
    "*BEA": "BEA",
}


def save_matrix(matrix, file_name):
    file = os.path.abspath("")
    dir_main = pathlib.Path(file).parent.absolute()
    savepath = os.path.join(dir_main, file_name)
    matrix.to_pickle(savepath)


# Turns out conformers are not very useful at all...
# So this function is not being used right now...
def load_conformer_priors(
    target_index,
    precomputed_file_name=OSDA_CONFORMER_PRIOR_FILE,
    identity_weight=0.01,
    normalize=True,
):
    precomputed_prior = pd.read_pickle(precomputed_file_name)
    precomputed_prior = precomputed_prior.reindex(target_index)
    precomputed_prior = precomputed_prior.filter(items=["conformer"])
    exploded_prior = pd.DataFrame(
        columns=[
            "num_conformers",
            "mean_volume",
            "min_volume",
            "std_dev_volume",
            "mean_energy",
            "min_energy",
            "std_dev_energy",
        ]
    )
    for index, row in precomputed_prior.iterrows():
        if (
            row["conformer"] is np.NaN
            or isinstance(row["conformer"], float)
            or isinstance(row["conformer"].values[0], float)
            or len(row["conformer"].values[0]["volumes"]) == 0
        ):
            series = pd.Series(
                {
                    "num_conformers": 1.0,
                    "mean_volume": 0.0,
                    "min_volume": 0.0,
                    "std_dev_volume": 0.0,
                    "mean_energy": 0.0,
                    "min_energy": 0.0,
                    "std_dev_energy": 0.0,
                }
            )
        else:
            conformer_properties = row["conformer"].values[0]
            series = pd.Series(
                {
                    "num_conformers": len(conformer_properties["volumes"]),
                    # "mean_volume": np.mean(conformer_properties["volumes"]),
                    # "min_volume": min(conformer_properties["volumes"]),
                    # "std_dev_volume": np.std(conformer_properties["volumes"]),
                    # "mean_energy": np.mean(conformer_properties["energies"]),
                    # "min_energy": min(conformer_properties["energies"]),
                    # "std_dev_energy": np.std(conformer_properties["energies"]),
                }
            )
        series.name = index
        exploded_prior = exploded_prior.append(series)
    if normalize:
        exploded_prior = exploded_prior.apply(lambda x: x / x.max(), axis=0)

    conformer_prior = exploded_prior.to_numpy(dtype=float)
    normalized_conformer_prior = conformer_prior / (max(conformer_prior, key=sum).sum())
    return (1 - identity_weight) * normalized_conformer_prior


def load_vector_priors(
    target_index,
    vector_feature,
    precomputed_file_name=OSDA_PRIOR_FILE,
    identity_weight=0.01,
    normalize=True,
    other_prior_to_concat=OSDA_HYPOTHETICAL_PRIOR_FILE,
    replace_nan=0.0,
):
    precomputed_prior = pd.read_pickle(precomputed_file_name)
    if other_prior_to_concat:
        big_precomputed_priors = pd.read_pickle(other_prior_to_concat)
        precomputed_prior = pd.concat([big_precomputed_priors, precomputed_prior])
        precomputed_prior = precomputed_prior[
            ~precomputed_prior.index.duplicated(keep="first")
        ]
    vector_explode = lambda x: pd.Series(x[vector_feature])
    precomputed_prior = precomputed_prior.apply(vector_explode, axis=1)
    exploded_prior = precomputed_prior.reindex(target_index)

    if replace_nan is not None:
        exploded_prior = exploded_prior.fillna(replace_nan)
    # Normalize across the whole thing...
    # Normalize to the biggest value & across all of the elements
    biggest_value = exploded_prior.max().max()
    normalization_factor = biggest_value * precomputed_prior.shape[1] + identity_weight
    if normalize:
        exploded_prior = exploded_prior.apply(
            lambda x: x / normalization_factor, axis=0
        )
    return exploded_prior


# TODO(Yitong): Cache this
# @lru_cache(maxsize=32)
def load_prior(
    target_index,
    column_weights,
    precomputed_file_name,
    identity_weight=0.01,
    normalize=True,
    prior_index_map=None,
    other_prior_to_concat=OSDA_HYPOTHETICAL_PRIOR_FILE,
):
    precomputed_prior = pd.read_pickle(precomputed_file_name)
    if other_prior_to_concat:
        big_precomputed_priors = pd.read_pickle(other_prior_to_concat)
        precomputed_prior = pd.concat([big_precomputed_priors, precomputed_prior])
        precomputed_prior = precomputed_prior[
            ~precomputed_prior.index.duplicated(keep="first")
        ]
    # breakpoint()
    if prior_index_map:  # zeolite prior lookup MR
        x = lambda i: prior_index_map[i] if i in prior_index_map else i
        precomputed_prior.index = precomputed_prior.index.map(x)
    precomputed_prior = precomputed_prior.reindex(target_index)
    precomputed_prior = precomputed_prior.filter(items=list(column_weights.keys()))
    precomputed_prior = precomputed_prior.apply(pd.to_numeric)
    precomputed_prior = precomputed_prior.fillna(0.0)
    results = precomputed_prior

    if normalize:
        # Normalize down each column to between 0 & 1
        precomputed_prior = precomputed_prior.apply(lambda x: x / x.max(), axis=0)
        # Now time to weigh each column, taking into account identity_weight to make sure
        # later when we add the identity matrix we don't go over 1.0 total per row...
        normalization_factor = sum(column_weights.values())
        results = precomputed_prior.apply(
            lambda x: x * column_weights[x.name] / normalization_factor, axis=0
        )
    # breakpoint()
    return (1 - identity_weight) * results


def osda_prior(
    all_data_df,
    identity_weight=0.01,
    prior_map=None,
    normalize=True,
):
    return load_prior(
        all_data_df.index,
        prior_map if prior_map is not None else OSDA_PRIOR_LOOKUP,
        OSDA_PRIOR_FILE,
        identity_weight,
        normalize,
    )


def osda_vector_prior(
    all_data_df,
    vector_feature="getaway",
    identity_weight=0.01,
    normalize=True,
    other_prior_to_concat=None,
):
    prior = osda_prior(all_data_df, identity_weight).to_numpy()
    getaway_prior = load_vector_priors(
        all_data_df.index,
        vector_feature,
        OSDA_PRIOR_FILE,
        identity_weight,
        normalize,
        other_prior_to_concat=other_prior_to_concat,
    ).to_numpy()
    # Splitting the original prior and the vector prior 50-50
    normalized_getaway_prior = getaway_prior / (2 * max(getaway_prior, key=sum).sum())
    normalized_prior = prior / (2 * max(prior, key=sum).sum())
    stacked = np.hstack([normalized_prior, normalized_getaway_prior])

    # Make room for the identity weight
    stacked = (1 - identity_weight) * stacked
    return stacked


def zeolite_prior(
    all_data_df,
    feature_lookup,
    identity_weight=0.01,
    normalize=True,
):
    # breakpoint()
    return load_prior(
        all_data_df.index,
        ZEOLITE_PRIOR_LOOKUP if not feature_lookup else feature_lookup,
        # PERSISTENCE_ZEOLITE_PRIOR_FILE,  
        GCNN_ZEOLITE_PRIOR_FILE,
        # ZEOLITE_PRIOR_FILE,
        identity_weight,
        normalize,
        ZEOLITE_PRIOR_MAP,
        other_prior_to_concat=ZEO_1_PRIOR,
    )


def osda_zeolite_combined_prior(
    all_data_df,
    identity_weight=0.01,
    normalize=True,
):
    osda_prior = load_prior(
        [i[0] for i in all_data_df.index],
        OSDA_PRIOR_LOOKUP,
        OSDA_PRIOR_FILE,
        identity_weight,
        normalize,
    ).to_numpy()
    osda_vector_prior = load_vector_priors(
        target_index=[i[0] for i in all_data_df.index],
        vector_feature="getaway",
        precomputed_file_name=OSDA_PRIOR_FILE,
        identity_weight=identity_weight,
        normalize=normalize,
        other_prior_to_concat=None,
    ).to_numpy()
    zeolite_prior = load_prior(
        [i[1] for i in all_data_df.index],
        ZEOLITE_PRIOR_LOOKUP,
        ZEOLITE_PRIOR_FILE,
        identity_weight,
        normalize,
        ZEOLITE_PRIOR_MAP,
    ).to_numpy()
    normalized_osda_vector_prior = osda_vector_prior / (
        3 * max(osda_vector_prior, key=sum).sum()
    )
    normalized_osda_prior = osda_prior / (3 * max(osda_prior, key=sum).sum())
    normalized_zeolite_prior = zeolite_prior / (3 * max(zeolite_prior, key=sum).sum())
    stacked = np.hstack(
        [normalized_osda_vector_prior, normalized_osda_prior, normalized_zeolite_prior]
    )
    stacked = np.nan_to_num(stacked)
    # Make room for the identity weight
    return (1 - identity_weight) * stacked


def plot_matrix(M, file_name, mask=None, vmin=16, vmax=23):
    fig, ax = plt.subplots()
    cmap = mpl.cm.get_cmap()
    cmap.set_bad(color="white")
    if mask is not None:

        def invert_binary_mask(m):
            return np.logical_not(m).astype(int)

        inverted_mask = invert_binary_mask(mask)
        masked_M = np.ma.masked_where(inverted_mask, M)
    else:
        masked_M = M
    im = ax.imshow(masked_M, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im)
    fig.savefig(file_name + ".png", dpi=150)


def make_prior(
    train,
    test,
    method="identity",
    normalization_factor=0.001,
    prior_map=None,
    all_data=None,
    test_train_axis=0,
):
    """
    train: training set

    test: test set

    method: which prior do you want to use? Hint: probably CustomOSDAVector or CustomZeolite

    normalization_factor: what to normalize the identity matrix we concat to. This is necessary
    to specify since we need all rows in the prior to sum to 1 & if the identity matrix is going to have
    value 0.001 then the rest of the row must sum to at most 0.999.

    prior_map: For CustomZeolite, CustomOSDA, and CustomOSDAVector: how do you want to weight the
    individual descriptors? Default is to weight all descriptors equally (check out ZEOLITE_PRIOR_LOOKUP &
    OSDA_PRIOR_LOOKUP). This might be a good thing to tweak for calibrated ensemble uncertainty.

    all_data: This is gross, but you also have the option to just give all the data instead
    of separately specifying test & train sets. This is for when you're no longer testing with
    10-fold cross validation; when you are ready to take your method and infer energies
    on a new distribution & want to use all of your data to train the NTK.

    test_train_axis: kinda no longer useful, but originally created if you want to join
    test or train by row or column. I don't think you'll ever need to change this.
    prior_map:
    """
    assert method in VALID_METHODS, f"Invalid method used, pick one of {VALID_METHODS}"
    if all_data is not None:
        all_data_df = all_data
    else:
        if test_train_axis == 0:
            all_data = np.vstack((train.to_numpy(), test.to_numpy()))
            all_data_df = pd.concat([train, test])
        elif test_train_axis == 1:
            all_data = np.hstack((train.to_numpy(), test.to_numpy()))
            all_data_df = pd.concat([train, test], 1)
        else:
            all_data = None
            all_data_df = pd.concat([train, test], test_train_axis)

    prior = None

    if method == "identity":
        # This is our baseline prior.
        prior = np.eye(all_data.shape[0])
        return prior

    elif method == "CustomOSDA":
        # CustomOSDA uses only the handcrafted OSDA descriptors
        prior = osda_prior(
            all_data_df=all_data_df,
            identity_weight=normalization_factor,
            prior_map=prior_map,
        ).to_numpy()
        return np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])

    elif method == "CustomOSDAVector":
        # CustomOSDAVector takes all of the handcrafted OSDA descriptors
        # and appends it to the GETAWAY prior
        prior = osda_vector_prior(all_data_df, "getaway", normalization_factor)
        return np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])

    elif method == "CustomZeolite":
        # CustomZeolite takes all of the handcrafted Zeolite descriptors
        prior = zeolite_prior(all_data_df, prior_map).to_numpy()
        return np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])

    # This one is for the failed experiment
    elif method == "CustomOSDAandZeolite":
        osda_axis1_lengths = osda_prior(
            all_data_df, column_name="Axis 1 (Angstrom)", normalize=False
        ).to_numpy()
        zeolite_sphere_diameters = zeolite_prior(all_data_df).to_numpy()

        prior = np.zeros((len(osda_axis1_lengths), len(zeolite_sphere_diameters)))
        for i, osda_length in enumerate(osda_axis1_lengths):
            for j, zeolite_sphere_diameter in enumerate(zeolite_sphere_diameters):
                prior[i, j] = zeolite_sphere_diameter - osda_length

        if prior.min() < 0:
            prior = prior - prior.min()
        # Normalize prior across its rows:
        max = np.reshape(np.repeat(prior.sum(axis=1), prior.shape[1]), prior.shape)
        prior = prior / max
        prior = np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])
        return prior

    # This is the one for really skinny Matrices
    elif method == "CustomOSDAandZeoliteAsRows":
        prior = osda_zeolite_combined_prior(all_data_df, normalize=True)
        # For now remove the identity concat to test eigenpro
        return prior  # np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])

    # This is the one for really skinny Matrices with sparse matrices.
    elif method == "CustomOSDAandZeoliteAsSparseMatrix":
        prior = csc_matrix(osda_zeolite_combined_prior(all_data_df, normalize=True))
        return sp.sparse.hstack(
            [prior, normalization_factor * sp.sparse.identity(all_data.shape[0])]
        )

    elif method == "random":
        dim = 100
        prior = np.random.rand(all_data.shape[0], dim)

    if test_train_axis == 0:
        prior = np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])
    elif test_train_axis == 1:
        prior = np.hstack(
            [
                prior,
                normalization_factor
                * np.eye(all_data.shape[1])[0 : all_data.shape[0], 1:],
            ]
        )
    normalize(prior, axis=1, copy=False)
    return prior
