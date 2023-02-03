#!/bin/bash
#SBATCH --job-name=run_1
#SBATCH --output=run_1_%j.log 
#SBATCH --partition=sched_mit_rafagb,sched_mit_hill,sched_opportunist
#SBATCH --time=20:00:00                  # hh:mm:ss, alternatively 1-00:00 dd-hh:mm
#SBATCH -n 12                           # cores. 8 for voronoi, 1 for dreiding. Doing 12 now and I think my code does nthread=5 for xgb
#SBATCH -N 1                            # nodes. 1 for voronoi, 1 for dreiding
#SBATCH --mem-per-cpu=5000              # mb. 3800 for voronoi, 1000 for dreiding

# https://engaging-web.mit.edu/eofe-wiki/slurm/resources/ 

source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/ntk

ROOT="/home/mrx/projects/affinity"
SRC="${ROOT}/ntk_matrix_completion"
OUTPUT="/pool001/mrx/projects/affinity/ntk_matrix_completion/output"

echo 'Experiment with multitaskcorr and prior_treatment: 3, structural features'

python $SRC/models/multitask/train.py --config $SRC/run_scripts/expts/230131_mt/run_1.yml

echo 'done'
# python -m cProfile -o $LOGS/230123_mt_1.prof $ROOT/models/multitask/train.py --config $ROOT/run_scripts/expts/230123_mt_1.yml
