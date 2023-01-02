#!/bin/bash
#SBATCH --job-name=test_eofe7
#SBATCH --output=test_eofe7_%j.log 
#SBATCH --partition=sched_mit_rafagb,sched_mit_hill,sched_opportunist
#SBATCH --time=20:00:00                  # hh:mm:ss, alternatively 1-00:00 dd-hh:mm
#SBATCH -n 12                           # cores. 8 for voronoi, 1 for dreiding. Doing 12 now and I think my code does nthread=5 for xgb
#SBATCH -N 1                            # nodes. 1 for voronoi, 1 for dreiding
#SBATCH --mem-per-cpu=2000              # mb. 3800 for voronoi, 1000 for dreiding


source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/ntk

ROOT="/home/mrx/projects/matrix_completion/ntk_matrix_completion"
OUTPUT="${ROOT}/output/2022_kfi/xgb_with_nb_hyperopt"

echo "root              " $ROOT
echo "output            " $OUTPUT

echo "==============================================================================================="

echo "Run 1) Science, hyperparameter tuning, zero" $(date) 

TRUTH="${ROOT}/data/truths/221216/science/zero/20221216_232046_truth.csv"
MASK="${ROOT}/data/truths/221216/science/zero/20221216_232046_mask.csv"
OSDA_PRIOR_FILE="${ROOT}/data/priors/221216/science/osda_priors_0.pkl"
ZEO_PRIOR_FILE="${ROOT}/data/priors/zeolite_priors_20221118_25419.pkl"
OSDA_PRIOR_MAP="${ROOT}/configs/osda_weights_structural_v2.json"
ZEO_PRIOR_MAP="${ROOT}/configs/zeolite_weights_structural_extendable.json"

echo "truth file        " $TRUTH
echo "mask file         " $MASK
echo "model             " $MODEL
echo "osda prior        " $OSDA_PRIOR_FILE
echo "osda map          " $OSDA_PRIOR_MAP
echo "zeolite prior     " $ZEO_PRIOR_FILE
echo "zeolite map       " $ZEO_PRIOR_MAP

python $ROOT/models/xgboost/xgb.py --output $OUTPUT --prior_method "CustomOSDAandZeoliteAsRows" --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --sieved_file $OSDA_PRIOR_FILE --prior_treatment 3 --truth $TRUTH --mask $MASK  --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --input_scaler minmax --tune

echo "==============================================================================================="
echo ""