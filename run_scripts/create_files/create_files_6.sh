source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/htvs
DJANGOCHEMDIR=/home/mrx/htvs/djangochem

CLUSTER_INBOX=/home/mrx/mnt/cluster/jobs/inbox
CLUSTER_COMPLETED=/home/mrx/mnt/cluster/jobs/completed
CLUSTER_STORAGE=/home/mrx/mnt/storage

export USER=$(whoami)

ROOT=/home/mrx/projects/binding_energies/ntk_matrix_completion
FP_NAMES=(mol_weight mol_volume asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct num_rot_bonds num_bonds formal_charge box axes getaway whim)

echo "Creating data files from testing_3" 
TRUTH_FILE=$ROOT/data/truths/testing_3/truth.csv
OP_DIR=$ROOT/data/priors/testing_3/
python $ROOT/features/create_prior_file.py --op $OP_DIR --truth_file $TRUTH_FILE --osda --features "${FP_NAMES[@]}" --exc 1 --batch_size 100000


