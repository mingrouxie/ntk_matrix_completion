#!/bin/bash

# Run this from inside ntk_matrix_completion directory - not good because on SLURM it'll be so finicky...
# How do you make it such that the paths are recognizable irrelevant of where you call the script from
# i.e. bash ~/..../mat_comp/ntk_mat_comp/... works, as does bash ntk_mat_comp/...

source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/ntk

ROOT="/home/mrx/projects/matrix_completion/ntk_matrix_completion"
OUTPUT="${ROOT}/output/2022_IZC/xgb_with_nb_hyperopt"

## NTK train
# python $ROOT/run_scripts/train.py

## XGB
# python $ROOT/models/xgboost/xgb.py
OSDA_CONFORMER_PRIOR_FILE_CLIPPED='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
# TRUTH='ntk_matrix_completion/data/daniels_data/science_paper/binding_nb_rowmean_debug.csv'
# MASK='ntk_matrix_completion/data/daniels_data/science_paper/mask_debug.csv'
TRUTH='ntk_matrix_completion/data/daniels_data/science_paper/binding_nb_rowmean.csv'
MASK='ntk_matrix_completion/data/daniels_data/science_paper/mask.csv'
# MODEL='/home/mrx/projects/matrix_completion/ntk_matrix_completion/output/2022_IZC/xgb_with_nb_hyperopt_2022916_104456/xgboost.json'

echo 'root' $ROOT
echo 'output' $OUTPUT
echo 'truth file' $TRUTH
echo 'mask file' $MASK

# echo 'debug'
# echo 'debug tune' `date`
# OSDA_PRIOR_FILE='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
# python -m cProfile -o $ROOT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --stack_combined_priors 'osda' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --tune --debug
# echo 'debug load from model file' `date`
# python -m cProfile -o $ROOT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --osda_prior_file $OSDA_PRIOR_FILE --stack_combined_priors 'osda' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --debug --model_file $MODEL

# echo '200 max_evals, full file, tune' `date`
# OSDA_PRIOR_FILE='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
# python -m cProfile -o $ROOT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --osda_prior_file $OSDA_PRIOR_FILE --stack_combined_priors 'osda' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --tune

# TODO: zeolite prior map input
# TODO: how to have osda vector prior but not structural prior? use weights file to specify zero weights for structural descriptors?

# echo 'Run 1) structural osda, getaway and one hot encoding zeolite' $(date)
# OSDA_PRIOR_FILE='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
# ZEO_PRIOR_FILE='ntk_matrix_completion/data/priors/zeolite_ohe.pkl'
# OSDA_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/osda_weights_structural.json'
# ZEO_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/zeolite_weights_ohe.json'
# python -m cProfile -o $OUTPUT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 5 --prior_method 'CustomOSDAandZeoliteAsRows' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --tune --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP
### --stack_combined_priors 'all'

echo 'Run 2) structural osda and one hot encoding zeolite' $(date)
OSDA_PRIOR_FILE='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
ZEO_PRIOR_FILE='ntk_matrix_completion/data/priors/zeolite_ohe.pkl'
OSDA_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/osda_weights_structural.json'
ZEO_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/zeolite_weights_ohe.json'
python -m cProfile -o $OUTPUT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 3 --prior_method 'CustomOSDAandZeoliteAsRows' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --tune --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP
### --stack_combined_priors 'all'

echo 'Run 3) structural osda and structural zeolite' $(date)
OSDA_PRIOR_FILE='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
ZEO_PRIOR_FILE='/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/handcrafted/iza_zeolites.pkl'
OSDA_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/osda_weights_structural.json'
ZEO_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/zeolite_weights_structural_extendable.json'
python -m cProfile -o $OUTPUT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 3 --prior_method 'CustomOSDAandZeoliteAsRows' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --tune --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP
### --stack_combined_priors 'all'

echo 'Run 4) structural osda and gcnn zeolite' $(date)
OSDA_PRIOR_FILE='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
ZEO_PRIOR_FILE='ntk_matrix_completion/data/swagata_gcnn/gcnn_priors.pkl'
OSDA_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/osda_weights_structural.json'
ZEO_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/zeolite_weights_gcnn.json'
python -m cProfile -o $OUTPUT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 3 --prior_method 'CustomOSDAandZeoliteAsRows' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --tune --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP
### --stack_combined_priors 'all'

echo 'Run 5) structural osda and PH zeolite' $(date)
OSDA_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/osda_weights_structural.json'
OSDA_PRIOR_FILE='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
ZEO_PRIOR_FILE='/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/nick_persistent/from_gdrive/iza_zeolites_topological-features.pkl'
ZEO_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/zeolite_ph_v2.json'
python -m cProfile -o $OUTPUT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 3 --prior_method 'CustomOSDAandZeoliteAsRows' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --tune --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP
### --stack_combined_priors 'all'

echo 'Run 6) getaway osda and one hot encoding zeolite' $(date)
OSDA_PRIOR_FILE='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
ZEO_PRIOR_FILE='ntk_matrix_completion/data/priors/zeolite_ohe.pkl'
OSDA_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/osda_weights_nostructural.json'
ZEO_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/zeolite_weights_ohe.json'
python -m cProfile -o $OUTPUT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 4 --prior_method 'CustomOSDAandZeoliteAsRows' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --tune --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP
### --stack_combined_priors 'all'

echo 'Run 7) getaway osda and structural zeolite' $(date)
OSDA_PRIOR_FILE='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
ZEO_PRIOR_FILE='/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/handcrafted/iza_zeolites.pkl'
OSDA_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/osda_weights_nostructural.json'
ZEO_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/zeolite_weights_structural_extendable.json'
python -m cProfile -o $OUTPUT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 4 --prior_method 'CustomOSDAandZeoliteAsRows' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --tune --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP
### --stack_combined_priors 'all'
#
echo 'Run 8) getaway osda and gcnn zeolite' $(date)
OSDA_PRIOR_FILE='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
ZEO_PRIOR_FILE='ntk_matrix_completion/data/swagata_gcnn/gcnn_priors.pkl'
OSDA_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/osda_weights_nostructural.json'
ZEO_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/zeolite_weights_gcnn.json'
python -m cProfile -o $OUTPUT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 4 --prior_method 'CustomOSDAandZeoliteAsRows' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --tune --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP
### --stack_combined_priors 'all'
#
echo 'Run 9) getaway osda and ph zeolite' $(date)
OSDA_PRIOR_FILE='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
ZEO_PRIOR_FILE='/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/nick_persistent/from_gdrive/iza_zeolites_topological-features.pkl'
OSDA_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/osda_weights_nostructural.json'
ZEO_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/zeolite_ph_v2.json'
python -m cProfile -o $OUTPUT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 4 --prior_method 'CustomOSDAandZeoliteAsRows' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --tune --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP
### --stack_combined_priors 'all'

# echo 'Run 10) whim osda and one hot encoding zeolite' `date`
# OSDA_PRIOR_FILE='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
# ZEO_PRIOR_FILE='ntk_matrix_completion/data/priors/zeolite_ohe.pkl'
# OSDA_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/osda_weights_nnostructural.json'
# ZEO_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/zeolite_weights_ohe.json'
# python -m cProfile -o $OUTPUT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 4 --prior_method 'CustomOSDAandZeoliteAsRows' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --tune --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --debug

# echo 'Run 11) whim osda and structural zeolite' `date`
# OSDA_PRIOR_FILE='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
# ZEO_PRIOR_FILE='/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/handcrafted/iza_zeolites.pkl'
# OSDA_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/osda_weights_nnostructural.json'
# ZEO_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/zeolite_weights_structural_extendable.json'
# python -m cProfile -o $OUTPUT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 4 --prior_method 'CustomOSDAandZeoliteAsRows' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --tune --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --debug

# echo 'Run 12) whim osda and gcnn zeolite' `date`
# OSDA_PRIOR_FILE='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
# ZEO_PRIOR_FILE='ntk_matrix_completion/data/swagata_gcnn/gcnn_priors.pkl'
# OSDA_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/osda_weights_nnostructural.json'
# ZEO_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/zeolite_weights_gcnn.json'
# python -m cProfile -o $OUTPUT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 4 --prior_method 'CustomOSDAandZeoliteAsRows' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --tune --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --debug

# echo 'Run 13) whim osda and ph zeolite' `date`
# OSDA_PRIOR_FILE='ntk_matrix_completion/data/priors/IZC_conformer_priors_clipped.pkl'
# ZEO_PRIOR_FILE='/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/nick_persistent/from_gdrive/iza_zeolites_topological-features.pkl'
# OSDA_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/osda_weights_nnostructural.json'
# ZEO_PRIOR_MAP='/home/mrx/projects/matrix_completion/ntk_matrix_completion/configs/zeolite_ph_v2.json'
# python -m cProfile -o $OUTPUT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --prior_treatment 4 --prior_method 'CustomOSDAandZeoliteAsRows' --truth $TRUTH --mask $MASK --search_type 'hyperopt' --tune --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --debug
