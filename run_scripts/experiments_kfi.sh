#!/bin/bash

# Run this from inside ntk_matrix_completion directory - not good because on SLURM it"ll be so finicky...
# How do you make it such that the paths are recognizable irrelevant of where you call the script from
# i.e. bash ~/..../mat_comp/ntk_mat_comp/... works, as does bash ntk_mat_comp/...

source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/ntk

ROOT="/home/mrx/projects/matrix_completion/ntk_matrix_completion"
OUTPUT="${ROOT}/output/2022_kfi/xgb_with_nb_hyperopt"
# OSDA_CONFORMER_PRIOR_FILE_CLIPPED="ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl"

echo "root" $ROOT
echo "output" $OUTPUT

#################################################################

echo "Run 1) structural osda, structural zeolite" $(date)
echo "No tuning, using tuned model"
TRUTH="${ROOT}/data/truths/iza_all/energies.csv"
MASK="${ROOT}/data/truths/iza_all/mask.csv"
MODEL="${ROOT}/output/2022_IZC/xgb_with_nb_hyperopt_2022108_223835/xgboost.json"
echo "truth file" $TRUTH
echo "mask file" $MASK
echo "model" $MODEL
OSDA_PRIOR_FILE="${ROOT}/data/priors/iza_all/osda_priors_20221116_14810_0.pkl"
# ZEO_PRIOR_FILE="${ROOT}/data/handcrafted/iza_zeolites.pkl"
ZEO_PRIOR_FILE="${ROOT}/data/priors/iza_all/zeolite_priors_20221116_20831.pkl"
OSDA_PRIOR_MAP="${ROOT}/configs/osda_weights_structural.json"
ZEO_PRIOR_MAP="${ROOT}/configs/zeolite_weights_structural_extendable.json"
python $ROOT/models/xgboost/xgb.py --output $OUTPUT --prior_method "CustomOSDAandZeoliteAsRows" --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 3 --truth $TRUTH --mask $MASK  --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --model_file $MODEL --debug
# -m cProfile -o $OUTPUT/program.prof

