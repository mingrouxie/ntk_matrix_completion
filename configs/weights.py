import json
from ntk_matrix_completion.utils.path_constants import ZEOLITE_PRIOR_LOOKUP
from ntk_matrix_completion.utils.path_constants import OSDA_PRIOR_LOOKUP
    
ZEOLITE_PRIOR_LOOKUP_DICT = {
    "a": 1.0,
    "b": 1.0,
    "c": 1.0,
    "alpha": 1.0,
    "betta": 1.0,
    "gamma": 1.0,
    "volume": 1.0,
    "framework_density": 1.0,
    "included_sphere_diameter": 1.0,
    "diffused_sphere_diameter_a": 1.0,
    "diffused_sphere_diameter_b": 1.0,
    "diffused_sphere_diameter_c": 1.0,
    "diffused_sphere_diameter_max_abc": 1.0,
    "num_atoms_per_vol": 1.0,
    "num_atoms": 1.0,
    "rdls": 1.0,
    "td_10": 1.0,
    "td": 1.0,
    "ring_size_0": 1.0,
    "ring_size_1": 1.0,
    "ring_size_2": 1.0,
    "accessible_volume": 1.0,
    "N_1": 1.0,
    "N_2": 1.0,
    "N_3": 1.0,
    "N_4": 1.0,
    "N_5": 1.0,
    "N_6": 1.0,
    "N_7": 1.0,
    "N_8": 1.0,
    "N_9": 1.0,
    "N_10": 1.0,
    "N_11": 1.0,
    "N_12": 1.0,
    "ring_size_3": 1.0,
    "ring_size_4": 1.0,
    "ring_size_5": 1.0,
    "ring_size_6": 1.0,
    "cell_birth-of-top-1-0D-feature": 1.0,
    "cell_death-of-top-1-0D-feature": 1.0,
    "cell_persistence-of-top-1-0D-feature": 1.0,
    "cell_birth-of-top-2-0D-feature": 1.0,
    "cell_death-of-top-2-0D-feature": 1.0,
    "cell_persistence-of-top-2-0D-feature": 1.0,
    "cell_birth-of-top-3-0D-feature": 1.0,
    "cell_death-of-top-3-0D-feature": 1.0,
    "cell_persistence-of-top-3-0D-feature": 1.0,
    "supercell_birth-of-top-1-0D-feature": 1.0,
    "supercell_death-of-top-1-0D-feature": 1.0,
    "supercell_persistence-of-top-1-0D-feature": 1.0,
    "supercell_birth-of-top-2-0D-feature": 1.0,
    "supercell_death-of-top-2-0D-feature": 1.0,
    "supercell_persistence-of-top-2-0D-feature": 1.0,
    "supercell_birth-of-top-3-0D-feature": 1.0,
    "supercell_death-of-top-3-0D-feature": 1.0,
    "supercell_persistence-of-top-3-0D-feature": 1.0,
    "cell_birth-of-top-1-1D-feature": 1.0,
    "cell_birth-of-top-2-1D-feature": 1.0,
    "cell_birth-of-top-3-1D-feature": 1.0,
    "cell_birth-of-top-1-2D-feature": 1.0,
    "cell_birth-of-top-2-2D-feature": 1.0,
    "cell_birth-of-top-3-2D-feature": 1.0,
    "a_projected-cell_birth-of-top-1-1D-feature": 1.0,
    "a_projected-cell_birth-of-top-2-1D-feature": 1.0,
    "a_projected-cell_birth-of-top-3-1D-feature": 1.0,
    "b_projected-cell_birth-of-top-1-1D-feature": 1.0,
    "b_projected-cell_birth-of-top-2-1D-feature": 1.0,
    "b_projected-cell_birth-of-top-3-1D-feature": 1.0,
    "c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    "c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    "c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    "a+b_projected-cell_birth-of-top-1-1D-feature": 1.0,
    "a+b_projected-cell_birth-of-top-2-1D-feature": 1.0,
    "a+b_projected-cell_birth-of-top-3-1D-feature": 1.0,
    "a-b_projected-cell_birth-of-top-1-1D-feature": 1.0,
    "a-b_projected-cell_birth-of-top-2-1D-feature": 1.0,
    "a-b_projected-cell_birth-of-top-3-1D-feature": 1.0,
    "a+c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    "a+c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    "a+c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    "a-c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    "a-c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    "a-c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    "b+c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    "b+c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    "b+c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    "b-c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    "b-c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    "b-c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    "a+b+c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    "a+b+c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    "a+b+c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    "a+b-c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    "a+b-c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    "a+b-c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    "a-b+c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    "a-b+c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    "a-b+c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    "a-b-c_projected-cell_birth-of-top-1-1D-feature": 1.0,
    "a-b-c_projected-cell_birth-of-top-2-1D-feature": 1.0,
    "a-b-c_projected-cell_birth-of-top-3-1D-feature": 1.0,
    "all-cell-projections_birth-of-top-1-1D-feature": 1.0,
    "all-cell-projections_birth-of-top-2-1D-feature": 1.0,
    "all-cell-projections_birth-of-top-3-1D-feature": 1.0,
    "supercell_birth-of-top-1-1D-feature": 1.0,
    "supercell_birth-of-top-2-1D-feature": 1.0,
    "supercell_birth-of-top-3-1D-feature": 1.0,
    "supercell_birth-of-top-1-2D-feature": 1.0,
    "supercell_birth-of-top-2-2D-feature": 1.0,
    "supercell_birth-of-top-3-2D-feature": 1.0,
    "a_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    "a_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    "a_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    "b_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    "b_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    "b_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    "c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    "c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    "c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    "a+b_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    "a+b_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    "a+b_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    "a-b_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    "a-b_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    "a-b_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    "a+c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    "a+c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    "a+c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    "a-c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    "a-c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    "a-c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    "b+c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    "b+c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    "b+c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    "b-c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    "b-c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    "b-c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    "a+b+c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    "a+b+c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    "a+b+c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    "a+b-c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    "a+b-c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    "a+b-c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    "a-b+c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    "a-b+c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    "a-b+c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    "a-b-c_projected-supercell_birth-of-top-1-1D-feature": 1.0,
    "a-b-c_projected-supercell_birth-of-top-2-1D-feature": 1.0,
    "a-b-c_projected-supercell_birth-of-top-3-1D-feature": 1.0,
    "all-supercell-projections_birth-of-top-1-1D-feature": 1.0,
    "all-supercell-projections_birth-of-top-2-1D-feature": 1.0,
    "all-supercell-projections_birth-of-top-3-1D-feature": 1.0,
    "cell_death-of-top-1-1D-feature": 1.0,
    "cell_death-of-top-2-1D-feature": 1.0,
    "cell_death-of-top-3-1D-feature": 1.0,
    "cell_death-of-top-1-2D-feature": 1.0,
    "cell_death-of-top-2-2D-feature": 1.0,
    "cell_death-of-top-3-2D-feature": 1.0,
    "a_projected-cell_death-of-top-1-1D-feature": 1.0,
    "a_projected-cell_death-of-top-2-1D-feature": 1.0,
    "a_projected-cell_death-of-top-3-1D-feature": 1.0,
    "b_projected-cell_death-of-top-1-1D-feature": 1.0,
    "b_projected-cell_death-of-top-2-1D-feature": 1.0,
    "b_projected-cell_death-of-top-3-1D-feature": 1.0,
    "c_projected-cell_death-of-top-1-1D-feature": 1.0,
    "c_projected-cell_death-of-top-2-1D-feature": 1.0,
    "c_projected-cell_death-of-top-3-1D-feature": 1.0,
    "a+b_projected-cell_death-of-top-1-1D-feature": 1.0,
    "a+b_projected-cell_death-of-top-2-1D-feature": 1.0,
    "a+b_projected-cell_death-of-top-3-1D-feature": 1.0,
    "a-b_projected-cell_death-of-top-1-1D-feature": 1.0,
    "a-b_projected-cell_death-of-top-2-1D-feature": 1.0,
    "a-b_projected-cell_death-of-top-3-1D-feature": 1.0,
    "a+c_projected-cell_death-of-top-1-1D-feature": 1.0,
    "a+c_projected-cell_death-of-top-2-1D-feature": 1.0,
    "a+c_projected-cell_death-of-top-3-1D-feature": 1.0,
    "a-c_projected-cell_death-of-top-1-1D-feature": 1.0,
    "a-c_projected-cell_death-of-top-2-1D-feature": 1.0,
    "a-c_projected-cell_death-of-top-3-1D-feature": 1.0,
    "b+c_projected-cell_death-of-top-1-1D-feature": 1.0,
    "b+c_projected-cell_death-of-top-2-1D-feature": 1.0,
    "b+c_projected-cell_death-of-top-3-1D-feature": 1.0,
    "b-c_projected-cell_death-of-top-1-1D-feature": 1.0,
    "b-c_projected-cell_death-of-top-2-1D-feature": 1.0,
    "b-c_projected-cell_death-of-top-3-1D-feature": 1.0,
    "a+b+c_projected-cell_death-of-top-1-1D-feature": 1.0,
    "a+b+c_projected-cell_death-of-top-2-1D-feature": 1.0,
    "a+b+c_projected-cell_death-of-top-3-1D-feature": 1.0,
    "a+b-c_projected-cell_death-of-top-1-1D-feature": 1.0,
    "a+b-c_projected-cell_death-of-top-2-1D-feature": 1.0,
    "a+b-c_projected-cell_death-of-top-3-1D-feature": 1.0,
    "a-b+c_projected-cell_death-of-top-1-1D-feature": 1.0,
    "a-b+c_projected-cell_death-of-top-2-1D-feature": 1.0,
    "a-b+c_projected-cell_death-of-top-3-1D-feature": 1.0,
    "a-b-c_projected-cell_death-of-top-1-1D-feature": 1.0,
    "a-b-c_projected-cell_death-of-top-2-1D-feature": 1.0,
    "a-b-c_projected-cell_death-of-top-3-1D-feature": 1.0,
    "all-cell-projections_death-of-top-1-1D-feature": 1.0,
    "all-cell-projections_death-of-top-2-1D-feature": 1.0,
    "all-cell-projections_death-of-top-3-1D-feature": 1.0,
    "supercell_death-of-top-1-1D-feature": 1.0,
    "supercell_death-of-top-2-1D-feature": 1.0,
    "supercell_death-of-top-3-1D-feature": 1.0,
    "supercell_death-of-top-1-2D-feature": 1.0,
    "supercell_death-of-top-2-2D-feature": 1.0,
    "supercell_death-of-top-3-2D-feature": 1.0,
    "a_projected-supercell_death-of-top-1-1D-feature": 1.0,
    "a_projected-supercell_death-of-top-2-1D-feature": 1.0,
    "a_projected-supercell_death-of-top-3-1D-feature": 1.0,
    "b_projected-supercell_death-of-top-1-1D-feature": 1.0,
    "b_projected-supercell_death-of-top-2-1D-feature": 1.0,
    "b_projected-supercell_death-of-top-3-1D-feature": 1.0,
    "c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    "c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    "c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    "a+b_projected-supercell_death-of-top-1-1D-feature": 1.0,
    "a+b_projected-supercell_death-of-top-2-1D-feature": 1.0,
    "a+b_projected-supercell_death-of-top-3-1D-feature": 1.0,
    "a-b_projected-supercell_death-of-top-1-1D-feature": 1.0,
    "a-b_projected-supercell_death-of-top-2-1D-feature": 1.0,
    "a-b_projected-supercell_death-of-top-3-1D-feature": 1.0,
    "a+c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    "a+c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    "a+c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    "a-c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    "a-c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    "a-c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    "b+c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    "b+c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    "b+c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    "b-c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    "b-c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    "b-c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    "a+b+c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    "a+b+c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    "a+b+c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    "a+b-c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    "a+b-c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    "a+b-c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    "a-b+c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    "a-b+c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    "a-b+c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    "a-b-c_projected-supercell_death-of-top-1-1D-feature": 1.0,
    "a-b-c_projected-supercell_death-of-top-2-1D-feature": 1.0,
    "a-b-c_projected-supercell_death-of-top-3-1D-feature": 1.0,
    "all-supercell-projections_death-of-top-1-1D-feature": 1.0,
    "all-supercell-projections_death-of-top-2-1D-feature": 1.0,
    "all-supercell-projections_death-of-top-3-1D-feature": 1.0,
    "cell_persistence-of-top-1-1D-feature": 1.0,
    "cell_persistence-of-top-2-1D-feature": 1.0,
    "cell_persistence-of-top-3-1D-feature": 1.0,
    "cell_persistence-of-top-1-2D-feature": 1.0,
    "cell_persistence-of-top-2-2D-feature": 1.0,
    "cell_persistence-of-top-3-2D-feature": 1.0,
    "a_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    "a_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    "a_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    "b_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    "b_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    "b_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    "c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    "c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    "c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    "a+b_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    "a+b_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    "a+b_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    "a-b_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    "a-b_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    "a-b_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    "a+c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    "a+c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    "a+c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    "a-c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    "a-c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    "a-c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    "b+c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    "b+c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    "b+c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    "b-c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    "b-c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    "b-c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    "a+b+c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    "a+b+c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    "a+b+c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    "a+b-c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    "a+b-c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    "a+b-c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    "a-b+c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    "a-b+c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    "a-b+c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    "a-b-c_projected-cell_persistence-of-top-1-1D-feature": 1.0,
    "a-b-c_projected-cell_persistence-of-top-2-1D-feature": 1.0,
    "a-b-c_projected-cell_persistence-of-top-3-1D-feature": 1.0,
    "all-cell-projections_persistence-of-top-1-1D-feature": 1.0,
    "all-cell-projections_persistence-of-top-2-1D-feature": 1.0,
    "all-cell-projections_persistence-of-top-3-1D-feature": 1.0,
    "supercell_persistence-of-top-1-1D-feature": 1.0,
    "supercell_persistence-of-top-2-1D-feature": 1.0,
    "supercell_persistence-of-top-3-1D-feature": 1.0,
    "supercell_persistence-of-top-1-2D-feature": 1.0,
    "supercell_persistence-of-top-2-2D-feature": 1.0,
    "supercell_persistence-of-top-3-2D-feature": 1.0,
    "a_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    "a_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    "a_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    "b_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    "b_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    "b_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    "c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    "c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    "c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    "a+b_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    "a+b_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    "a+b_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    "a-b_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    "a-b_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    "a-b_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    "a+c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    "a+c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    "a+c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    "a-c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    "a-c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    "a-c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    "b+c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    "b+c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    "b+c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    "b-c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    "b-c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    "b-c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    "a+b+c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    "a+b+c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    "a+b+c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    "a+b-c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    "a+b-c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    "a+b-c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    "a-b+c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    "a-b+c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    "a-b+c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    "a-b-c_projected-supercell_persistence-of-top-1-1D-feature": 1.0,
    "a-b-c_projected-supercell_persistence-of-top-2-1D-feature": 1.0,
    "a-b-c_projected-supercell_persistence-of-top-3-1D-feature": 1.0,
    "all-supercell-projections_persistence-of-top-1-1D-feature": 1.0,
    "all-supercell-projections_persistence-of-top-2-1D-feature": 1.0,
    "all-supercell-projections_persistence-of-top-3-1D-feature": 1.0,
    "0": 1.0,
    "1": 1.0,
    "2": 1.0,
    "3": 1.0,
    "4": 1.0,
    "5": 1.0,
    "6": 1.0,
    "7": 1.0,
    "8": 1.0,
    "9": 1.0,
    "10": 1.0,
    "11": 1.0,
    "12": 1.0,
    "13": 1.0,
    "14": 1.0,
    "15": 1.0,
    "16": 1.0,
    "17": 1.0,
    "18": 1.0,
    "19": 1.0,
    "20": 1.0,
    "21": 1.0,
    "22": 1.0,
    "23": 1.0,
    "24": 1.0,
    "25": 1.0,
    "26": 1.0,
    "27": 1.0,
    "28": 1.0,
    "29": 1.0,
    "30": 1.0,
    "31": 1.0,
    "32": 1.0,
    "33": 1.0,
    "34": 1.0,
    "35": 1.0,
    "36": 1.0,
    "37": 1.0,
    "38": 1.0,
    "39": 1.0,
    "40": 1.0,
    "41": 1.0,
    "42": 1.0,
    "43": 1.0,
    "44": 1.0,
    "45": 1.0,
    "46": 1.0,
    "47": 1.0,
    "48": 1.0,
    "49": 1.0,
    "50": 1.0,
    "51": 1.0,
    "52": 1.0,
    "53": 1.0,
    "54": 1.0,
    "55": 1.0,
    "56": 1.0,
    "57": 1.0,
    "58": 1.0,
    "59": 1.0,
    "60": 1.0,
    "61": 1.0,
    "62": 1.0,
    "63": 1.0,
    "64": 1.0,
    "65": 1.0,
    "66": 1.0,
    "67": 1.0,
    "68": 1.0,
    "69": 1.0,
    "70": 1.0,
    "71": 1.0,
    "72": 1.0,
    "73": 1.0,
    "74": 1.0,
    "75": 1.0,
    "76": 1.0,
    "77": 1.0,
    "78": 1.0,
    "79": 1.0,
    "80": 1.0,
    "81": 1.0,
    "82": 1.0,
    "83": 1.0,
    "84": 1.0,
    "85": 1.0,
    "86": 1.0,
    "87": 1.0,
    "88": 1.0,
    "89": 1.0,
    "90": 1.0,
    "91": 1.0,
    "92": 1.0,
    "93": 1.0,
    "94": 1.0,
    "95": 1.0,
    "96": 1.0,
    "97": 1.0,
    "98": 1.0,
    "99": 1.0,
    "100": 1.0,
    "101": 1.0,
    "102": 1.0,
    "103": 1.0,
    "104": 1.0,
    "105": 1.0,
    "106": 1.0,
    "107": 1.0,
    "108": 1.0,
    "109": 1.0,
    "110": 1.0,
    "111": 1.0,
    "112": 1.0,
    "113": 1.0,
    "114": 1.0,
    "115": 1.0,
    "116": 1.0,
    "117": 1.0,
    "118": 1.0,
    "119": 1.0,
    "120": 1.0,
    "121": 1.0,
    "122": 1.0,
    "123": 1.0,
    "124": 1.0,
    "125": 1.0,
    "126": 1.0,
    "127": 1.0,
    "128": 1.0,
    "129": 1.0,
    "130": 1.0,
    "131": 1.0,
    "132": 1.0,
    "133": 1.0,
    "134": 1.0,
    "135": 1.0,
    "136": 1.0,
    "137": 1.0,
    "138": 1.0,
    "139": 1.0,
    "140": 1.0,
    "141": 1.0,
    "142": 1.0,
    "143": 1.0,
    "144": 1.0,
    "145": 1.0,
    "146": 1.0,
    "147": 1.0,
    "148": 1.0,
    "149": 1.0,
    "150": 1.0,
    "151": 1.0,
    "152": 1.0,
    "153": 1.0,
    "154": 1.0,
    "155": 1.0,
    "156": 1.0,
    "157": 1.0,
    "158": 1.0,
    "159": 1.0,
    "160": 1.0,
    "161": 1.0,
    "162": 1.0,
    "163": 1.0,
    "164": 1.0,
    "165": 1.0,
    "166": 1.0,
    "167": 1.0,
    "168": 1.0,
    "169": 1.0,
    "170": 1.0,
    "171": 1.0,
    "172": 1.0,
    "173": 1.0,
    "174": 1.0,
    "175": 1.0,
    "176": 1.0,
    "177": 1.0,
    "178": 1.0,
    "179": 1.0,
    "180": 1.0,
    "181": 1.0,
    "182": 1.0,
    "183": 1.0,
    "184": 1.0,
    "185": 1.0,
    "186": 1.0,
    "187": 1.0,
    "188": 1.0,
    "189": 1.0,
    "190": 1.0,
    "191": 1.0,
    "192": 1.0,
    "193": 1.0,
    "194": 1.0,
    "195": 1.0,
    "196": 1.0,
    "197": 1.0,
    "198": 1.0,
    "199": 1.0,
    "200": 1.0,
    "201": 1.0,
    "202": 1.0,
    "203": 1.0,
    "204": 1.0,
    "205": 1.0,
    "206": 1.0,
    "207": 1.0,
    "208": 1.0,
    "209": 1.0,
    "210": 1.0,
    "211": 1.0,
    "212": 1.0,
    "213": 1.0,
    "214": 1.0,
    "215": 1.0,
    "216": 1.0,
    "217": 1.0,
    "218": 1.0,
    "219": 1.0,
    "220": 1.0,
    "221": 1.0,
    "222": 1.0,
    "223": 1.0,
    "224": 1.0,
    "225": 1.0,
    "226": 1.0,
    "227": 1.0,
    "228": 1.0,
    "229": 1.0,
    "230": 1.0,
    "231": 1.0,
    "232": 1.0,
    "233": 1.0,
    "234": 1.0,
    "235": 1.0,
    "236": 1.0,
    "237": 1.0,
    "238": 1.0,
    "239": 1.0,
    "240": 1.0,
    "241": 1.0,
    "242": 1.0,
    "243": 1.0,
    "244": 1.0,
    "245": 1.0,
    "246": 1.0,
    "247": 1.0,
    "248": 1.0,
    "249": 1.0,
    "250": 1.0,
    "251": 1.0,
    "252": 1.0,
    "253": 1.0,
    "254": 1.0,
    "255": 1.0,
    "256": 1.0,
    "257": 1.0,
    "258": 1.0,
    "259": 1.0,
    "260": 1.0,
    "261": 1.0,
    "262": 1.0,
    "263": 1.0,
    "264": 1.0,
    "265": 1.0,
    "266": 1.0,
    "267": 1.0,
    "268": 1.0,
    "269": 1.0,
    "270": 1.0,
    "271": 1.0,
    "272": 1.0,
    "273": 1.0,
    "274": 1.0,
    "275": 1.0,
    "276": 1.0,
    "277": 1.0,
    "278": 1.0,
    "279": 1.0,
    "280": 1.0,
    "281": 1.0,
    "282": 1.0,
    "283": 1.0,
    "284": 1.0,
    "285": 1.0,
    "286": 1.0,
    "287": 1.0,
    "288": 1.0,
    "289": 1.0,
    "290": 1.0,
    "291": 1.0,
    "292": 1.0,
    "293": 1.0,
    "294": 1.0,
    "295": 1.0,
    "296": 1.0,
    "297": 1.0,
    "298": 1.0,
    "299": 1.0,
    "300": 1.0,
    "301": 1.0,
    "302": 1.0,
    "303": 1.0,
    "304": 1.0,
    "305": 1.0,
    "306": 1.0,
    "307": 1.0,
    "308": 1.0,
    "309": 1.0,
    'MFI': 1.0, 
    'OWE': 1.0, 
    'IWR': 1.0, 
    'NSI': 1.0, 
    'BPH': 1.0, 
    'STW': 1.0, 
    'TON': 1.0, 
    'MEL': 1.0, 
    'ITT': 1.0, 
    'CGS': 1.0, 
    'KFI': 1.0, 
    'STT': 1.0, 
    'OFF': 1.0, 
    'SAF': 1.0, 
    'MEI': 1.0, 
    'UFI': 1.0, 
    'STO': 1.0, 
    'LOS': 1.0, 
    'LIO': 1.0, 
    'JOZ': 1.0, 
    'PHI': 1.0, 
    'TER': 1.0, 
    'AHT': 1.0, 
    'IWS': 1.0, 
    'ATT': 1.0, 
    'NPT': 1.0, 
    'AVL': 1.0, 
    'RHO': 1.0, 
    'AFR': 1.0, 
    'JRY': 1.0, 
    'HEU': 1.0, 
    'ITR': 1.0, 
    'MOR': 1.0, 
    'AEN': 1.0, 
    'STI': 1.0, 
    'SAO': 1.0, 
    'IFO': 1.0, 
    'UTL': 1.0, 
    'DAC': 1.0, 
    'MTT': 1.0, 
    'SBT': 1.0, 
    'TOL': 1.0, 
    'POR': 1.0, 
    'AFV': 1.0, 
    'JSW': 1.0, 
    'ERI': 1.0, 
    'EDI': 1.0, 
    'ITH': 1.0, 
    'SVV': 1.0, 
    'DOH': 1.0, 
    'AFO': 1.0, 
    'VET': 1.0, 
    'PCS': 1.0, 
    'RUT': 1.0, 
    'SOD': 1.0, 
    'AWW': 1.0, 
    'MEP': 1.0, 
    'CON': 1.0, 
    'DDR': 1.0, 
    'GME': 1.0, 
    'MOZ': 1.0, 
    'SSY': 1.0, 
    'IFY': 1.0, 
    'SAT': 1.0, 
    'PWO': 1.0, 
    'UOZ': 1.0, 
    'ISV': 1.0, 
    'EMT': 1.0, 
    'PAU': 1.0, 
    'MRE': 1.0, 
    'AFY': 1.0, 
    'IRR': 1.0, 
    'LTA': 1.0, 
    'CHA': 1.0, 
    'LAU': 1.0, 
    'MTN': 1.0, 
    'SOS': 1.0, 
    'TUN': 1.0, 
    'UEI': 1.0, 
    'MSE': 1.0, 
    'NON': 1.0, 
    'PCR': 1.0, 
    'SFO': 1.0, 
    'APD': 1.0, 
    'LOV': 1.0, 
    'SAV': 1.0, 
    'IMF': 1.0, 
    'RRO': 1.0, 
    'MTW': 1.0, 
    'SFS': 1.0, 
    'THO': 1.0, 
    'VFI': 1.0, 
    'BEA': 1.0, 
    'MSO': 1.0, 
    'ASV': 1.0, 
    'NAT': 1.0, 
    'PTY': 1.0, 
    'SOF': 1.0, 
    'AEI': 1.0, 
    'IFR': 1.0, 
    'IFW': 1.0, 
    'GIS': 1.0, 
    'OBW': 1.0, 
    'FRA': 1.0, 
    'SFW': 1.0, 
    'SSF': 1.0, 
    'ITW': 1.0, 
    'SBE': 1.0, 
    'VNI': 1.0, 
    'DFT': 1.0, 
    'CZP': 1.0, 
    'EPI': 1.0, 
    'ESV': 1.0, 
    'DFO': 1.0, 
    'BEC': 1.0, 
    'AFS': 1.0, 
    'FAR': 1.0, 
    'LTF': 1.0, 
    'PWN': 1.0, 
    'POS': 1.0, 
    'MER': 1.0, 
    'IRN': 1.0, 
    'USI': 1.0, 
    'MAZ': 1.0, 
    'EWS': 1.0, 
    'AFI': 1.0, 
    'AFG': 1.0, 
    'ETV': 1.0, 
    'SFN': 1.0, 
    'MRT': 1.0, 
    'JSN': 1.0, 
    'ATN': 1.0, 
    'BOZ': 1.0, 
    'STF': 1.0, 
    'AWO': 1.0, 
    'UOE': 1.0, 
    'LEV': 1.0, 
    'BSV': 1.0, 
    'SFH': 1.0, 
    'IHW': 1.0, 
    'OKO': 1.0, 
    'ATS': 1.0, 
    'UOV': 1.0, 
    'MTF': 1.0, 
    'AST': 1.0, 
    'ACO': 1.0, 
    'ETR': 1.0, 
    'ITE': 1.0, 
    'JST': 1.0, 
    'MFS': 1.0, 
    'PWW': 1.0, 
    'SAS': 1.0, 
    'ETL': 1.0, 
    'EON': 1.0, 
    'ATO': 1.0, 
    'CTH': 1.0, 
    'EZT': 1.0, 
    'TSC': 1.0, 
    'SEW': 1.0, 
    'CGF': 1.0, 
    'MWW': 1.0, 
    'GIU': 1.0, 
    'SOV': 1.0, 
    'SBS': 1.0, 
    'AFX': 1.0, 
    'CDO': 1.0, 
    'EAB': 1.0, 
    'BOG': 1.0, 
    'RTH': 1.0, 
    'AFN': 1.0, 
    'OSO': 1.0, 
    'SGT': 1.0, 
    'JSR': 1.0, 
    'APC': 1.0, 
    'BOF': 1.0, 
    'FER': 1.0, 
    'AEL': 1.0, 
    'ITG': 1.0, 
    'BRE': 1.0, 
    'CAN': 1.0, 
    'SWY': 1.0, 
    'SFF': 1.0, 
    'PUN': 1.0, 
    'UOS': 1.0, 
    'AFT': 1.0, 
    'SIV': 1.0, 
    'EWO': 1.0, 
    'OSI': 1.0, 
    'RTE': 1.0, 
    'LTL': 1.0, 
    'CFI': 1.0, 
    'FAU': 1.0, 
    'ZON': 1.0, 
    'SZR': 1.0, 
    'PON': 1.0, 
    'SFE': 1.0, 
    'EEI': 1.0, 
    'GON': 1.0, 
    'CSV': 1.0, 
    'AVE': 1.0, 
    'MAR': 1.0, 
    'SBN': 1.0, 
    'NPO': 1.0, 
    'AET': 1.0, 
    'IWW': 1.0, 
    'SFG': 1.0, 
    'SOR': 1.0, 
    'EUO': 1.0, 
    'JNT': 1.0, 
}

OSDA_PRIOR_LOOKUP_DICT = {
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


if __name__ == '__main__':
    with open(ZEOLITE_PRIOR_LOOKUP, 'w') as z:
        json.dump(ZEOLITE_PRIOR_LOOKUP_DICT, z)
    with open(OSDA_PRIOR_LOOKUP, 'w') as o:
        json.dump(OSDA_PRIOR_LOOKUP_DICT, o)