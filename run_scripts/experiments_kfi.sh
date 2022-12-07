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
echo "Run 0) structural osda, structural zeolite, original files, just testing model loading + retraining" $(date) 
# echo "No tuning, using tuned model"
# echo "TESTED. Recovers original, see analysis2.ipynb"
# TRUTH="${ROOT}/data/daniels_data/science_paper/binding_nb_rowmean.csv" # binding_nb_rowmean_1193.csv" #truths/iza_all/new.csv" # iza/iza_lit_binding/energies.csv"
# MASK="${ROOT}/data/daniels_data/science_paper/mask.csv" # truths/iza_all/mask.csv" # iza/iza_lit_binding/mask.csv"
# MODEL="${ROOT}/output/2022_IZC/xgb_with_nb_hyperopt_2022108_223835/xgboost.json" # struct&struct
# OSDA_PRIOR_FILE="${ROOT}/data/priors/IZC_conformer_priors_clipped.pkl" # iza_all/osda_priors_20221120_6230_0.pkl" # osda_priors_20221118_3659_0.pkl"
# ZEO_PRIOR_FILE="${ROOT}/data/handcrafted/iza_zeolites.pkl" # priors/zeolite_priors_20221118_25419.pkl"
# OSDA_PRIOR_MAP="${ROOT}/configs/osda_weights_structural_izc.json"
# ZEO_PRIOR_MAP="${ROOT}/configs/zeolite_weights_structural_extendable_izc.json"

# echo "truth file        " $TRUTH
# echo "mask file         " $MASK
# echo "model             " $MODEL
# echo "osda prior        " $OSDA_PRIOR_FILE
# echo "osda map          " $OSDA_PRIOR_MAP
# echo "zeolite prior     " $ZEO_PRIOR_FILE
# echo "zeolite map       " $ZEO_PRIOR_MAP

# python $ROOT/models/xgboost/xgb.py --output $OUTPUT --prior_method "CustomOSDAandZeoliteAsRows" --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 3 --truth $TRUTH --mask $MASK  --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --model_file $MODEL --debug --sieved_file $OSDA_PRIOR_FILE --input_scaler minmax 
# # -m cProfile -o $OUTPUT/program.prof
# echo "==============================================================================================="

#################################################################

echo "==============================================================================================="
echo "Run 1) Same as Run 0, but using the new priors and prior map" $(date) 
# echo "No tuning, using tuned model"
# echo "TESTED. About 0.6% worse RMSE wise than Run 0. Topk _looks_ the same"
# TRUTH="${ROOT}/data/daniels_data/science_paper/binding_nb_rowmean.csv" # binding_nb_rowmean_1193.csv" #truths/iza_all/new.csv" # iza/iza_lit_binding/energies.csv"
# MASK="${ROOT}/data/daniels_data/science_paper/mask.csv" # truths/iza_all/mask.csv" # iza/iza_lit_binding/mask.csv"
# MODEL="${ROOT}/output/2022_IZC/xgb_with_nb_hyperopt_2022108_223835/xgboost.json" # struct&struct
# OSDA_PRIOR_FILE="${ROOT}/data/priors/iza_all/osda_priors_20221120_6230_0.pkl" # osda_priors_20221118_3659_0.pkl"
# ZEO_PRIOR_FILE="${ROOT}/data/priors/zeolite_priors_20221118_25419.pkl"
# OSDA_PRIOR_MAP="${ROOT}/configs/osda_weights_structural.json"
# ZEO_PRIOR_MAP="${ROOT}/configs/zeolite_weights_structural_extendable.json"

# echo "truth file        " $TRUTH
# echo "mask file         " $MASK
# echo "model             " $MODEL
# echo "osda prior        " $OSDA_PRIOR_FILE
# echo "osda map          " $OSDA_PRIOR_MAP
# echo "zeolite prior     " $ZEO_PRIOR_FILE
# echo "zeolite map       " $ZEO_PRIOR_MAP

# python $ROOT/models/xgboost/xgb.py --output $OUTPUT --prior_method "CustomOSDAandZeoliteAsRows" --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 3 --truth $TRUTH --mask $MASK  --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --model_file $MODEL --debug --sieved_file $OSDA_PRIOR_FILE --input_scaler minmax 
# # -m cProfile -o $OUTPUT/program.prof
# echo "==============================================================================================="

#################################################################

echo "==============================================================================================="
echo "Run 2) Same as Run 1, but with the new truth files" $(date) 
echo "No tuning, using tuned model"
TRUTH="${ROOT}/data/truths/science_paper/_20221120_213153_energies.csv"
# _20221120_21112_energies_nb.csv"
#/data/truths/iza_all/_20221120_174646_energies_nb.csv"
#/data/daniels_data/science_paper/binding_nb_rowmean.csv" # binding_nb_rowmean_1193.csv" #truths/iza_all/new.csv" # iza/iza_lit_binding/energies.csv"
MASK="${ROOT}/data/truths/science_paper/_20221120_213153_mask.csv"
# _20221120_21112_mask.csv
# /data/truths/iza_all/_20221120_174646_mask.csv" # truths/iza_all/mask.csv" # iza/iza_lit_binding/mask.csv"
MODEL="${ROOT}/output/2022_IZC/xgb_with_nb_hyperopt_2022108_223835/xgboost.json" # struct&struct
OSDA_PRIOR_FILE="${ROOT}/data/priors/iza_all/osda_priors_20221120_6230_0.pkl" # osda_priors_20221118_3659_0.pkl"
ZEO_PRIOR_FILE="${ROOT}/data/priors/zeolite_priors_20221118_25419.pkl"
OSDA_PRIOR_MAP="${ROOT}/configs/osda_weights_structural.json"
ZEO_PRIOR_MAP="${ROOT}/configs/zeolite_weights_structural_extendable.json"

echo "truth file        " $TRUTH
echo "mask file         " $MASK
echo "model             " $MODEL
echo "osda prior        " $OSDA_PRIOR_FILE
echo "osda map          " $OSDA_PRIOR_MAP
echo "zeolite prior     " $ZEO_PRIOR_FILE
echo "zeolite map       " $ZEO_PRIOR_MAP

python $ROOT/models/xgboost/xgb.py --output $OUTPUT --prior_method "CustomOSDAandZeoliteAsRows" --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 3 --truth $TRUTH --mask $MASK  --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --model_file $MODEL --debug --sieved_file $OSDA_PRIOR_FILE --input_scaler minmax 
# -m cProfile -o $OUTPUT/program.prof
echo "==============================================================================================="
# echo "==============================================================================================="

#################################################################

echo "==============================================================================================="
echo "Run 3) Using previously extracted files for iza-all" $(date) 
echo "No tuning, using tuned model"
TRUTH="${ROOT}/data/truths/"
MASK="${ROOT}/data/truths/science_paper/_20221120_213153_mask.csv"
MODEL="${ROOT}/output/2022_IZC/xgb_with_nb_hyperopt_2022108_223835/xgboost.json" # struct&struct
OSDA_PRIOR_FILE="${ROOT}/data/priors/iza_all/osda_priors_20221120_6230_0.pkl" # osda_priors_20221118_3659_0.pkl"
ZEO_PRIOR_FILE="${ROOT}/data/priors/zeolite_priors_20221118_25419.pkl"
OSDA_PRIOR_MAP="${ROOT}/configs/osda_weights_structural.json"
ZEO_PRIOR_MAP="${ROOT}/configs/zeolite_weights_structural_extendable.json"

echo "truth file        " $TRUTH
echo "mask file         " $MASK
echo "model             " $MODEL
echo "osda prior        " $OSDA_PRIOR_FILE
echo "osda map          " $OSDA_PRIOR_MAP
echo "zeolite prior     " $ZEO_PRIOR_FILE
echo "zeolite map       " $ZEO_PRIOR_MAP

python $ROOT/models/xgboost/xgb.py --output $OUTPUT --prior_method "CustomOSDAandZeoliteAsRows" --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 3 --truth $TRUTH --mask $MASK  --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --model_file $MODEL --debug --sieved_file $OSDA_PRIOR_FILE --input_scaler minmax 
# -m cProfile -o $OUTPUT/program.prof
echo "==============================================================================================="
# echo "==============================================================================================="