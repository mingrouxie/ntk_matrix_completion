#!/bin/bash
#SBATCH --job-name=230103_xgb_2
#SBATCH --output=230103_xgb_2_%j.log 
#SBATCH --partition=sched_mit_rafagb,sched_mit_hill,sched_opportunist
#SBATCH --time=20:00:00                  
#SBATCH -n 24                        # tasks or cpu cores   
#SBATCH -N 1                         # num of nodes to spread requested cores across   
#SBATCH --mem-per-cpu=2000  
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=mrx@mit.edu            

source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/ntk

ROOT="/home/mrx/projects/binding_energies/ntk_matrix_completion"
OUTPUT="${ROOT}/output/2022_kfi/xgb_with_nb_hyperopt"

echo "Run 0) Science, hyperparameter tuning, mean" $(date) #DONE 
TRUTH="${ROOT}/data/truths/221216/science/mean/20221216_232046_truth.csv"
MASK="${ROOT}/data/truths/221216/science/mean/20221216_232046_mask.csv"
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

cd $ROOT

python models/xgboost/xgb.py --output $OUTPUT --prior_method "CustomOSDAandZeoliteAsRows" --osda_prior_file $OSDA_PRIOR_FILE --zeolite_prior_file $ZEO_PRIOR_FILE --sieved_file $OSDA_PRIOR_FILE --prior_treatment 3 --truth $TRUTH --mask $MASK  --osda_prior_map $OSDA_PRIOR_MAP --zeolite_prior_map $ZEO_PRIOR_MAP --input_scaler minmax --tune --nthread 24

echo "done"