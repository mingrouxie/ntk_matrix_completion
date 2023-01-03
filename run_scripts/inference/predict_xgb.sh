source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/ntk

echo 'This code works'

##### DEBUGGING #######################################################################################
# ROOT=/home/mrx/projects/matrix_completion/ntk_matrix_completion/
# OP=$ROOT/output/20221208_inference
# MODEL=$ROOT/output/2022_kfi/xgb_with_nb_hyperopt_20221120_163652 # see analysis2
# HYP_OSDA=$ROOT/data/priors/hyp_osdas_others # this works too
# # HYP_OSDA=$ROOT/data/priors/hyp_osdas_others_old # for debug
# IZA_ZEO=$ROOT/data/priors/zeolite_priors_20221118_25419.pkl # this definitely works
# # LIG="C[C@H]1CC[N@+]12C[C@H]1[C@H]3CC[C@H](C3)[C@H]12" # this works
# # SUB=(KFI LTA) # this works
# LWEIGHTS=$ROOT/configs/osda_weights_structural_v2_debug.json
# SWEIGHTS=$ROOT/configs/zeolite_weights_structural_extendable.json

# python /home/mrx/projects/matrix_completion/ntk_matrix_completion/models/xgboost/predict.py --output $OP --model_dir $MODEL --new_lig_dir $HYP_OSDA --new_sub_file $IZA_ZEO --lig_weights $LWEIGHTS --sub_weights $SWEIGHTS 

# --new_lig $LIG --new_sub "${SUB[@]}"

echo ''

##### predict KFI selected molecules for different frameworks ###########################################

echo ''

ROOT=/home/mrx/projects/matrix_completion/ntk_matrix_completion/
OP=$ROOT/output/20221224_inference
MODEL=$ROOT/output/2022_kfi/xgb_with_nb_hyperopt_20221218_223455 # see analysis11
HYP_OSDA=/home/mrx/projects/basf/221223_hyp_osdas/221224_batch1/0.pkl
IZA_ZEO=$ROOT/data/priors/zeolite_priors_20221118_25419.pkl # this definitely works
LWEIGHTS=$ROOT/configs/osda_weights_structural_v2.json
SWEIGHTS=$ROOT/configs/zeolite_weights_structural_extendable.json

python /home/mrx/projects/matrix_completion/ntk_matrix_completion/models/xgboost/predict.py --output $OP --model_dir $MODEL --new_lig_file $HYP_OSDA --new_sub_file $IZA_ZEO --lig_weights $LWEIGHTS --sub_weights $SWEIGHTS 

# --new_lig $LIG --new_sub "${SUB[@]}"

