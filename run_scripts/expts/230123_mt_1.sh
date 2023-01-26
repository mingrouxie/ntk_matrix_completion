#!/usr/bin/env bash

source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/ntk

ROOT="/home/mrx/projects/binding_energies/ntk_matrix_completion"
OUTPUT="${ROOT}/output/2022_IZC/xgb_with_nb_hyperopt"

python $ROOT/models/multitask/train.py --config $ROOT/run_scripts/expts/230123_mt_1.yml
