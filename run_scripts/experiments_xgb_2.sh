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

# echo "==============================================================================================="
# echo "Run 0) Science, hyperparameter tuning, mean" $(date) #DONE 
# TRUTH="${ROOT}/data/truths/221216/science/mean/20221216_232046_truth.csv"
# MASK="${ROOT}/data/truths/221216/science/mean/20221216_232046_mask.csv"
# OSDA_PRIOR_FILE="${ROOT}/data/priors/221216/science/osda_priors_0.pkl"
# ZEO_PRIOR_FILE="${ROOT}/data/priors/zeolite_priors_20221118_25419.pkl"
# OSDA_PRIOR_MAP="${ROOT}/configs/osda_weights_structural_v2.json"
# ZEO_PRIOR_MAP="${ROOT}/configs/zeolite_weights_structural_extendable.json"

# echo "truth file        " $TRUTH
# echo "mask file         " $MASK
# echo "model             " $MODEL
# echo "osda prior        " $OSDA_PRIOR_FILE
# echo "osda map          " $OSDA_PRIOR_MAP
# echo "zeolite prior     " $ZEO_PRIOR_FILE
# echo "zeolite map       " $ZEO_PRIOR_MAP

# python $ROOT/models/xgboost/xgb.py --output $OUTPUT --prior_method "CustomOSDAandZeoliteAsRows" --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --sieved_file $OSDA_PRIOR_FILE --prior_treatment 3 --truth $TRUTH --mask $MASK  --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --input_scaler minmax --tune
# echo "==============================================================================================="
# echo ""

#################################################################

echo "==============================================================================================="
echo "Run 1) Science, hyperparameter tuning, zero" $(date) 
TRUTH="${ROOT}/data/truths/221216/science/zero/20221216_232046_truth.csv"
MASK="${ROOT}/data/truths/221216/science/zero/20221216_232046_mask.csv"
OSDA_PRIOR_FILE="${ROOT}/data/priors/221216/science/osda_priors_0.pkl"
ZEO_PRIOR_FILE="${ROOT}/data/priors/zeolite_priors_20221118_25419.pkl"
OSDA_PRIOR_MAP="${ROOT}/configs/osda_weights_structural_v2.json"
ZEO_PRIOR_MAP="${ROOT}/configs/zeolite_weights_structural_extendable.json"

echo "truth file        " $TRUTH
echo "mask file         " $MASK
echo "model             " $MODEL
echo "osda prior        " $OSDA_PRIOR_FILE
echo "osda map          " $OSDA_PRIOR_MAP
echo "zeolite prior     " $ZEO_PRIOR_FILE
echo "zeolite map       " $ZEO_PRIOR_MAP

python $ROOT/models/xgboost/xgb.py --output $OUTPUT --prior_method "CustomOSDAandZeoliteAsRows" --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --sieved_file $OSDA_PRIOR_FILE --prior_treatment 3 --truth $TRUTH --mask $MASK  --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --input_scaler minmax --tune
echo "==============================================================================================="
echo ""

#################################################################

echo "==============================================================================================="
echo "Run 2) Science, hyperparameter tuning, small_pos" $(date) 
TRUTH="${ROOT}/data/truths/221216/science/small_pos/20221216_232046_truth.csv"
MASK="${ROOT}/data/truths/221216/science/small_pos/20221216_232046_mask.csv"
OSDA_PRIOR_FILE="${ROOT}/data/priors/221216/science/osda_priors_0.pkl"
ZEO_PRIOR_FILE="${ROOT}/data/priors/zeolite_priors_20221118_25419.pkl"
OSDA_PRIOR_MAP="${ROOT}/configs/osda_weights_structural_v2.json"
ZEO_PRIOR_MAP="${ROOT}/configs/zeolite_weights_structural_extendable.json"

echo "truth file        " $TRUTH
echo "mask file         " $MASK
echo "model             " $MODEL
echo "osda prior        " $OSDA_PRIOR_FILE
echo "osda map          " $OSDA_PRIOR_MAP
echo "zeolite prior     " $ZEO_PRIOR_FILE
echo "zeolite map       " $ZEO_PRIOR_MAP

python $ROOT/models/xgboost/xgb.py --output $OUTPUT --prior_method "CustomOSDAandZeoliteAsRows" --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --sieved_file $OSDA_PRIOR_FILE --prior_treatment 3 --truth $TRUTH --mask $MASK  --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --input_scaler minmax --tune
echo "==============================================================================================="
echo ""

#################################################################

echo "==============================================================================================="
echo "Run 3) Science, hyperparameter tuning, large_pos" $(date) 
TRUTH="${ROOT}/data/truths/221216/science/large_pos/20221216_232046_truth.csv"
MASK="${ROOT}/data/truths/221216/science/large_pos/20221216_232046_mask.csv"
OSDA_PRIOR_FILE="${ROOT}/data/priors/221216/science/osda_priors_0.pkl"
ZEO_PRIOR_FILE="${ROOT}/data/priors/zeolite_priors_20221118_25419.pkl"
OSDA_PRIOR_MAP="${ROOT}/configs/osda_weights_structural_v2.json"
ZEO_PRIOR_MAP="${ROOT}/configs/zeolite_weights_structural_extendable.json"

echo "truth file        " $TRUTH
echo "mask file         " $MASK
echo "model             " $MODEL
echo "osda prior        " $OSDA_PRIOR_FILE
echo "osda map          " $OSDA_PRIOR_MAP
echo "zeolite prior     " $ZEO_PRIOR_FILE
echo "zeolite map       " $ZEO_PRIOR_MAP

python $ROOT/models/xgboost/xgb.py --output $OUTPUT --prior_method "CustomOSDAandZeoliteAsRows" --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --sieved_file $OSDA_PRIOR_FILE --prior_treatment 3 --truth $TRUTH --mask $MASK  --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --input_scaler minmax --tune
echo "==============================================================================================="
echo ""


#################################################################

echo "==============================================================================================="
echo "Run 4) Science, hyperparameter tuning, max_plus" $(date) 
TRUTH="${ROOT}/data/truths/221216/science/small_pos/20221216_232046_truth.csv"
MASK="${ROOT}/data/truths/221216/science/small_pos/20221216_232046_mask.csv"
OSDA_PRIOR_FILE="${ROOT}/data/priors/221216/science/osda_priors_0.pkl"
ZEO_PRIOR_FILE="${ROOT}/data/priors/zeolite_priors_20221118_25419.pkl"
OSDA_PRIOR_MAP="${ROOT}/configs/osda_weights_structural_v2.json"
ZEO_PRIOR_MAP="${ROOT}/configs/zeolite_weights_structural_extendable.json"

echo "truth file        " $TRUTH
echo "mask file         " $MASK
echo "model             " $MODEL
echo "osda prior        " $OSDA_PRIOR_FILE
echo "osda map          " $OSDA_PRIOR_MAP
echo "zeolite prior     " $ZEO_PRIOR_FILE
echo "zeolite map       " $ZEO_PRIOR_MAP

python $ROOT/models/xgboost/xgb.py --output $OUTPUT --prior_method "CustomOSDAandZeoliteAsRows" --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --sieved_file $OSDA_PRIOR_FILE --prior_treatment 3 --truth $TRUTH --mask $MASK  --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --input_scaler minmax --tune
echo "==============================================================================================="
echo ""




