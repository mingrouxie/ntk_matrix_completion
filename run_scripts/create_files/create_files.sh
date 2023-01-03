source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/htvs
DJANGOCHEMDIR=/home/mrx/htvs/djangochem

CLUSTER_INBOX=/home/mrx/mnt/cluster/jobs/inbox
CLUSTER_COMPLETED=/home/mrx/mnt/cluster/jobs/completed
CLUSTER_STORAGE=/home/mrx/mnt/storage

export USER=$(whoami)


################################################

# echo 'Creating OSDA prior file for all based on Science paper' DONE

# TRUTH_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/daniels_data/science_paper
# TRUTH_FILE=$TRUTH_DIR/binding_nb_rowmean.csv
# PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/daniels_data/science_paper

# FP_NAMES=(box volume axes whim getaway num_bonds num_rot_bonds asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct num_bonds num_rot_bonds num_atoms c_charge_ratio)
# python ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --features "${FP_NAMES[@]}" --osda --batch_size 10000 --truth_file $TRUTH_FILE --exc 1

# ################################################


# TRUTH_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/science_paper/
# TRUTH_FILE=$TRUTH_DIR/energies.csv
# PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/science_paper/

# echo 'Creating truth file for Science paper' DONE
# python /home/mrx/projects/matrix_completion/ntk_matrix_completion/utils/create_truth.py --op $TRUTH_DIR --science --exc 1 --nb 1 --nan_after_nb drop

# ################################################

# echo 'Creating truth file for IZA-(lit+hyp)'
# TRUTH_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_all_no_nb/
# TRUTH_FILE=$TRUTH_DIR/energies.csv
# PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/iza_all_no_nb/

# python /home/mrx/projects/matrix_completion/ntk_matrix_completion/utils/create_truth.py --op $TRUTH_DIR --ms literature 210720_omar_quaternary 210720_omar_diquaternary --substrate iza_parse --exc 1

# echo 'Creating OSDA prior file for all based on truth file' 
# FP_NAMES=(box volume axes whim getaway num_bonds num_rot_bonds asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct num_bonds num_rot_bonds num_atoms c_charge_ratio)
# python ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --features "${FP_NAMES[@]}" --osda --batch_size 10000 --truth_file $TRUTH_FILE --exc 1

# # # ################################################

# TRUTH_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_lit/
# TRUTH_FILE=$TRUTH_DIR/energies.csv
# PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/iza_lit/

# echo 'Creating truth file for lit'
# python /home/mrx/projects/matrix_completion/ntk_matrix_completion/utils/create_truth.py --op $TRUTH_DIR --ms literature --substrate iza_parse --exc 1 --nb 1 --nan_after_nb drop

# echo 'Creating OSDA prior file for lit based on truth file' 
# FP_NAMES=(box volume axes whim getaway num_bonds num_rot_bonds asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct num_bonds num_rot_bonds num_atoms c_charge_ratio)
# python ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --features "${FP_NAMES[@]}" --osda --batch_size 10000 --truth_file $TRUTH_FILE --exc 1

# # ################################################

# TRUTH_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_hyp/
# TRUTH_FILE=$TRUTH_DIR/energies.csv
# PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/iza_hyp/

# echo 'Creating truth file for hyp'
# python /home/mrx/projects/matrix_completion/ntk_matrix_completion/utils/create_truth.py --op $TRUTH_DIR --ms 210720_omar_quaternary 210720_omar_diquaternary --substrate iza_parse --exc 1 --nb 1 --nan_after_nb drop

# echo 'Creating OSDA prior file for hyp based on truth file' 
# FP_NAMES=(box volume axes whim getaway num_bonds num_rot_bonds asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct num_bonds num_rot_bonds num_atoms c_charge_ratio)
# python ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --features "${FP_NAMES[@]}" --osda --batch_size 10000 --truth_file $TRUTH_FILE --exc 1

