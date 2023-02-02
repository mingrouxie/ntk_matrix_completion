#!/bin/bash
#SBATCH --job-name=230123_mt_2
#SBATCH --output=230123_mt_2_%j.log 
#SBATCH --partition=sched_mit_rafagb,sched_mit_hill,sched_opportunist
#SBATCH --time=20:00:00                  # hh:mm:ss, alternatively 1-00:00 dd-hh:mm
#SBATCH -n 12                           # cores. 8 for voronoi, 1 for dreiding. Doing 12 now and I think my code does nthread=5 for xgb
#SBATCH -N 1                            # nodes. 1 for voronoi, 1 for dreiding
#SBATCH --mem-per-cpu=5000              # mb. 3800 for voronoi, 1000 for dreiding

# https://engaging-web.mit.edu/eofe-wiki/slurm/resources/ 

source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/ntk

ROOT="/home/mrx/projects/binding_energies"
SRC="${ROOT}/ntk_matrix_completion"
LOGS="${ROOT}/logs/23_mt"

echo 'Experiment with multitasknnsep and prior_treatment: 3'

python $SRC/models/multitask/train.py --config $SRC/run_scripts/expts/230123_mt_3.yml

echo 'done'
# python -m cProfile -o $LOGS/230123_mt_1.prof $ROOT/models/multitask/train.py --config $ROOT/run_scripts/expts/230123_mt_1.yml
