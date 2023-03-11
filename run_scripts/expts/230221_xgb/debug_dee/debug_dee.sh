#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --output=debug_%j.log 
#SBATCH --partition=sched_mit_rafagb
#SBATCH --time=20:00:00                  # hh:mm:ss, alternatively 1-00:00 dd-hh:mm
#SBATCH -n 20                           # cores. 8 for voronoi, 1 for dreiding. Doing 12 now and I think my code does nthread=5 for xgb
#SBATCH -N 1                            # nodes. 1 for voronoi, 1 for dreiding
#SBATCH --gres=gpu:1

# https://engaging-web.mit.edu/eofe-wiki/slurm/resources/ 
# ,sched_mit_hill,sched_opportunist

source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/ntk

ROOT="/home/mrx/projects/affinity"
SRC="${ROOT}/ntk_matrix_completion"
OUTPUT="/pool001/mrx/projects/affinity/ntk_matrix_completion/output"
OUTPUT="/home/mrx/projects/affinity_pool/ntk_matrix_completion/output"

python3 $SRC/models/xgboost/xgb_v2.py --config $SRC/run_scripts/expts/230221_xgb/debug_dee/debug_dee.yml
