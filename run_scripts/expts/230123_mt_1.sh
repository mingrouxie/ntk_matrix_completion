#!/usr/bin/env bash

source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/ntk

ROOT="/home/mrx/projects/binding_energies"
SRC="${ROOT}/ntk_matrix_completion"
LOGS="${ROOT}/logs/m23_mt"

python $SRC/models/multitask/train.py --config $SRC/run_scripts/expts/230123_mt_1.yml


# python -m cProfile -o $LOGS/230123_mt_1.prof $ROOT/models/multitask/train.py --config $ROOT/run_scripts/expts/230123_mt_1.yml