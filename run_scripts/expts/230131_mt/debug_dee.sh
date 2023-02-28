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

python $SRC/models/multitask/train.py --config $SRC/run_scripts/expts/230131_mt/debug_dee.yml

# python $SRC/utils/analysis_utilities.py --args $SRC/run_scripts/expts/230131_mt/run_3/run_3_args.yaml --local --energy_label "Binding (SiO2)" --load_label single

# python $SRC/utils/analysis_utilities.py --config /home/mrx/projects/affinity/output/2023_multitask/multitask_202321_225523

# python $SRC/utils/analysis_utilities.py --config $OUTPUT/2023_multitask/multitask_202322_163715

echo 'done'
# python -m cProfile -o $LOGS/230123_mt_1.prof $ROOT/models/multitask/train.py --config $ROOT/run_scripts/expts/230123_mt_1.yml
