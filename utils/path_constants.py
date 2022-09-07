import argparse

# TODO: add some path wizardry to make sure these work despite the directory it's called in...
DEBUG = True

"""
INPUT DATA
"""
# Format binding.csv from Schwalbe Koda's work: https://github.com/learningmatter-mit/Zeolite-Phase-Competition/blob/main/data/binding.csv
BINDING_CSV = "ntk_matrix_completion/data/daniels_data/science_paper/binding.csv"
BINDING_NB_CSV = "ntk_matrix_completion/data/daniels_data/science_paper/binding_nb_rowmean.csv"  # with row mean for non binding entries
MASK_CSV = "ntk_matrix_completion/data/daniels_data/science_paper/mask.csv"
BINDING_GROUND_TRUTH = "ntk_matrix_completion/data/truths/BindingSiO2GroundTruth.pkl"
TEMPLATING_GROUND_TRUTH = "ntk_matrix_completion/data/truths/TemplatingGroundTruth.pkl"

# Data enumerating 78K hypothetical OSDAs along with their binding energies & boxes (not included in repository)
HYPOTHETICAL_OSDA_ENERGIES = (
    "ntk_matrix_completion/data/daniels_data/211221_energies.csv"
)
HYPOTHETICAL_OSDA_BOXES = "ntk_matrix_completion/data/daniels_data/211221_boxes.csv"

# Debugging files - uncomment when debugging
if DEBUG:
    BINDING_CSV = (
        "ntk_matrix_completion/data/daniels_data/science_paper/binding_debug.csv"
    ) # note this does not contain all the pairs found in the NB_CSV version 
    BINDING_NB_CSV = "ntk_matrix_completion/data/daniels_data/science_paper/binding_nb_rowmean_debug.csv"  # with row mean for non binding entries
    MASK_CSV = "ntk_matrix_completion/data/daniels_data/science_paper/mask_debug.csv"

"""
GENERATED PRIORS
"""
ZEOLITE_PRIOR_FILE = (
    "ntk_matrix_completion/data/handcrafted/scraped_zeolite_data_with_rings.pkl"
)
HANDCRAFTED_ZEOLITE_PRIOR_FILE = "ntk_matrix_completion/data/handcrafted/handcrafted.pkl"  # ZEOLITE_PRIOR_FILE + a bit more
PERSISTENCE_ZEOLITE_PRIOR_FILE = (
    "ntk_matrix_completion/data/nick_persistent/numeric_zeolite_df.pkl"
)
ZEOLITE_GCNN_EMBEDDINGS_FILE = "ntk_matrix_completion/data/swagata_gcnn/gcnn_priors.pkl"
ZEOLITE_ALL_PRIOR_FILE = "ntk_matrix_completion/data/zeolite_all_priors.pkl"
# TEMP_0D_PRIOR_FILE = "/Users/mr/Documents/Work/MIT/PhD/projects/matrix_completion/persistent_homology/20220421_nick_has_0D/no_diag_0D.pkl"
""""""
ZEO_1_PRIOR = "ntk_matrix_completion/data/handcrafted/zeo_1.pkl"
# OSDA_PRIOR_FILE = "ntk_matrix_completion/data/priors_old/precomputed_OSDA_prior_10_with_whims.pkl"
OSDA_PRIOR_FILE = (
    "ntk_matrix_completion/data/priors_old/precomputed_OSDA_prior_10_with_whims_v2.pkl"
)
# OSDA_CONFORMER_PRIOR_FILE = "ntk_matrix_completion/data/OSDA_priors_with_conjugates.pkl"
OSDA_CONFORMER_PRIOR_FILE = "ntk_matrix_completion/data/priors/IZC_conformer_priors.pkl"
OSDA_CONFORMER_PRIOR_FILE_SIEVED = (
    "ntk_matrix_completion/data/priors/IZC_conformer_priors_sieved_getaway.pkl"
)
OSDA_CONFORMER_PRIOR_FILE_CLIPPED = (
    "ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl"
)
# OSDA_CONFORMER_PRIOR_FILE = "ntk_matrix_completion/data/priors/IZC_docked_priors.pkl" # TODO: how to make use of this hmmm
OSDA_ZEO1_PRIOR_FILE = (
    "ntk_matrix_completion/data/tricyclohexylmethylphosphonium_prior.pkl"
)
OSDA_HYPOTHETICAL_PRIOR_FILE = (
    "ntk_matrix_completion/data/priors_old/precomputed_energies_78616by196WithWhims.pkl"
)

"""
NTK OUTPUTS
"""
OSDA_HYPOTHETICAL_PREDICTED_ENERGIES = (
    "ntk_matrix_completion/output/predicted_energies_for_78K_OSDAs.pkl"
)
ZEOLITE_HYPOTHETICAL_PREDICTED_ENERGIES = (
    "ntk_matrix_completion/output/predicted_energies_for_zeo1.pkl"
)
TEN_FOLD_CROSS_VALIDATION_ENERGIES = (
    "ntk_matrix_completion/output/energy_predictions.pkl"
)
PERFORMANCE_METRICS = "ntk_matrix_completion/output/peformance_metrics.pkl"
ZEOLITE_PRIOR_SELECTION_FILE = (
    "ntk_matrix_completion/output/zeolite_prior_selection.csv"
)
OUTPUT_DIR = "ntk_matrix_completion/output/"
XGBOOST_OUTPUT_DIR = OUTPUT_DIR + "2022_IZC/xgb_with_nb_hyperopt/"

# TODO: need a parser.py to collate all parsers from everywhere
# And then in the main file, import from parser.py and then delegate to whichever methods are called
# train, model, etc.
# right now experiments.sh is still WIP - calling xgb.py for xgb, calling train.py for ntk stuff. Ugh
# parser = argparse.ArgumentParser(description="Path constant parser")
# parser.add_argument(
#     "--debug", help="Uses test data files for debugging", action="store_true"
# )
