source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/htvs
DJANGOCHEMDIR=/home/mrx/htvs/djangochem

CLUSTER_INBOX=/home/mrx/mnt/cluster/jobs/inbox
CLUSTER_COMPLETED=/home/mrx/mnt/cluster/jobs/completed
CLUSTER_STORAGE=/home/mrx/mnt/storage

export USER=$(whoami)

ROOT=/home/mrx/projects/matrix_completion/ntk_matrix_completion

# echo 'Creating OSDA prior file for all based on Science paper' # DONE

# TRUTH_DIR=$ROOT/data/truths/221216/science/mean
# TRUTH_FILE=$TRUTH_DIR/20221216_232046_truth.csv
# PRIOR_DIR=$ROOT/data/priors/221216/science

# FP_NAMES=(mol_weight mol_volume asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct num_rot_bonds num_bonds formal_charge box axes getaway whim)

# python ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --features "${FP_NAMES[@]}" --osda --batch_size 10000 --truth_file $TRUTH_FILE --exc 1


# old
# FP_NAMES=(box volume axes whim getaway num_bonds num_rot_bonds asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct num_bonds num_rot_bonds num_atoms)


echo 'Creating OSDA prior file for all IZA datapoints' 

TRUTH_DIR=$ROOT/data/truths/221216/iza_all/mean
TRUTH_FILE=$TRUTH_DIR/20221217_45051_truth_before_nb.csv
PRIOR_DIR=$ROOT/data/priors/221216/iza_all

FP_NAMES=(mol_weight mol_volume asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct num_rot_bonds num_bonds formal_charge box axes getaway whim)

python ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --features "${FP_NAMES[@]}" --osda --batch_size 10000 --truth_file $TRUTH_FILE --exc 1