source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/htvs
DJANGOCHEMDIR=/home/mrx/htvs/djangochem

CLUSTER_INBOX=/home/mrx/mnt/cluster/jobs/inbox
CLUSTER_COMPLETED=/home/mrx/mnt/cluster/jobs/completed
CLUSTER_STORAGE=/home/mrx/mnt/storage

export USER=$(whoami)

# OUTPUT_FILE=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/testing
################################################

PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/hyp_osdas/

echo 'Creating prior file' 
FP_NAMES=(box volume axes whim getaway num_bonds num_rot_bonds asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct)
python ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --features "${FP_NAMES[@]}" --osda --batch_size 10000 --exc 1 --ms 210720_omar_diquaternary 210720_omar_quaternary
