source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/htvs
DJANGOCHEMDIR=/home/mrx/htvs/djangochem

CLUSTER_INBOX=/home/mrx/mnt/cluster/jobs/inbox
CLUSTER_COMPLETED=/home/mrx/mnt/cluster/jobs/completed
CLUSTER_STORAGE=/home/mrx/mnt/storage

export USER=$(whoami)

# OUTPUT_FILE=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/testing
################################################

TRUTH_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_all/
TRUTH_FILE=$TRUTH_DIR/energies.csv
PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/iza_all/

echo 'Creating truth file'
python /home/mrx/projects/matrix_completion/ntk_matrix_completion/utils/create_truth.py --op $TRUTH_DIR --ms literature 210720_omar_quaternary 210720_omar_diquaternary --cs iza --exc 1 --nb 5

echo 'Creating prior file' 
FP_NAMES=(box volume axes whim getaway num_bonds num_rot_bonds asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct)
python /home/mrx/projects/matrix_completion/ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --features "${FP_NAMES[@]}" --osda --batch_size 10000 --truth_file $TRUTH_FILE --exc 1

# ################################################

TRUTH_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_lit/
TRUTH_FILE=$TRUTH_DIR/energies.csv
PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/iza_lit/

echo 'Creating truth file'
python /home/mrx/projects/matrix_completion/ntk_matrix_completion/utils/create_truth.py --op $TRUTH_DIR --ms literature --cs iza --exc 1 --nb 5

echo 'Creating prior file' 
FP_NAMES=(box volume axes whim getaway num_bonds num_rot_bonds asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct)
python ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --features "${FP_NAMES[@]}" --osda --batch_size 10000 --truth_file $TRUTH_FILE --exc 1

################################################

TRUTH_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_hyp/
TRUTH_FILE=$TRUTH_DIR/energies.csv
PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/iza_hyp/

echo 'Creating truth file'
python /home/mrx/projects/matrix_completion/ntk_matrix_completion/utils/create_truth.py --op $TRUTH_DIR --ms 210720_omar_quaternary 210720_omar_diquaternary --cs iza --exc 1 --nb 5

echo 'Creating prior file' 
FP_NAMES=(box volume axes whim getaway num_bonds num_rot_bonds asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct)
python ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --features "${FP_NAMES[@]}" --osda --batch_size 10000 --truth_file $TRUTH_FILE --exc 1

