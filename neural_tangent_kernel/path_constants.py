# TODO: add some path wizardry to make sure these work despite the directory it's called in...

"""
INPUT DATA
"""
# Format binding.csv from Schwalbe Koda's work: https://github.com/learningmatter-mit/Zeolite-Phase-Competition/blob/main/data/binding.csv
BINDING_CSV = "data/binding.csv"
BINDING_GROUND_TRUTH = "data/BindingSiO2GroundTruth.pkl"
TEMPLATING_GROUND_TRUTH = "data/TemplatingGroundTruth.pkl"

# Data enumerating 78K hypothetical OSDAs along with their binding energies & boxes (not included in repository)
HYPOTHETICAL_OSDA_ENERGIES = "data/daniels_data/211221_energies.csv"
HYPOTHETICAL_OSDA_BOXES = "data/daniels_data/211221_boxes.csv"


"""
GENERATED PRIORS
"""
ZEOLITE_PRIOR_FILE = "data/scraped_zeolite_data.pkl"
"""TODO: Mingrou pick and select priors we want with the code, rather than having to curate a new pickle everytime?"""
PERSISTENCE_ZEOLITE_PRIOR_FILE = '/Users/mr/Documents/Work/MIT/PhD/projects/matrix_completion/persistent_homology/20220412_nick/numeric_zeolite_df_2.pkl'
ZEOLITE_GCNN_EMBEDDINGS_FILE = "/Users/mr/Documents/Work/MIT/PhD/projects/matrix_completion/gcnn_zeolite_priors/gcnn_priors.pkl"
# ZEOLITE_GCNN_EMBEDDINGS_FILE = "data/gcnn_zeolite_embeddings.pkl"
ZEOLITE_ALL_PRIOR_FILE = "data/zeolite_all_priors.pkl"
TEMP_0D_PRIOR_FILE = '/Users/mr/Documents/Work/MIT/PhD/projects/matrix_completion/persistent_homology/20220421_nick_has_0D/no_diag_0D.pkl'
""""""
ZEO_1_PRIOR = "data/zeo_1.pkl"
OSDA_PRIOR_FILE = "data/precomputed_OSDA_prior_10_with_whims.pkl"
OSDA_CONFORMER_PRIOR_FILE = "data/OSDA_priors_with_conjugates.pkl"
OSDA_ZEO1_PRIOR_FILE = "data/tricyclohexylmethylphosphonium_prior.pkl"
OSDA_HYPOTHETICAL_PRIOR_FILE = (
    "data/daniels_data/precomputed_energies_78616by196WithWhims.pkl"
)

"""
NTK OUTPUTS
"""
OSDA_HYPOTHETICAL_PREDICTED_ENERGIES = (
    "data/predicted_templating_energies_for_78K_OSDAs.pkl"
)
ZEOLITE_HYPOTHETICAL_PREDICTED_ENERGIES = "data/predicted_binding_energies_for_zeo1.pkl"
TEN_FOLD_CROSS_VALIDATION_ENERGIES = "data/templating_pred.pkl"
PERFORMANCE_METRICS = 'data/top_n_in_top_k_accuracies.csv'