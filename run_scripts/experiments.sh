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
python -m cProfile -o $OUTPUT/program.prof $ROOT/models/xgboost/xgb.py --output $OUTPUT --stack_combined_priors 'osda' --search_type 'hyperopt'