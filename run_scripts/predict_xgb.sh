source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/ntk

ROOT=/home/mrx/projects/matrix_completion/ntk_matrix_completion/
OP=$ROOT/output/20221208_inference
MODEL=$ROOT/output/2022_kfi/xgb_with_nb_hyperopt_20221120_163652
HYP_OSDA=$ROOT/data/priors/hyp_osdas_others
HYP_OSDA=$ROOT/data/priors/hyp_osdas_others_old # DEBUG PURPOSES
IZA_ZEO=$ROOT/data/priors/zeolite_priors_20221118_25419.pkl
# LIG=
SUB=(KFI LTA)
LWEIGHTS=$ROOT/configs/osda_weights_structural_izc_v2.json
SWEIGHTS=$ROOT/configs/zeolite_weights_structural_extendable.json

python /home/mrx/projects/matrix_completion/ntk_matrix_completion/models/xgboost/predict.py --output $OP --model_dir $MODEL --new_lig_dir $HYP_OSDA --new_sub_file $IZA_ZEO --lig_weights $LWEIGHTS --sub_weights $SWEIGHTS  --new_sub "${SUB[@]}"

# --new_lig $LIG

