source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/htvs
DJANGOCHEMDIR=/home/mrx/htvs/djangochem

CLUSTER_INBOX=/home/mrx/mnt/cluster/jobs/inbox
CLUSTER_COMPLETED=/home/mrx/mnt/cluster/jobs/completed
CLUSTER_STORAGE=/home/mrx/mnt/storage

export USER=$(whoami)

ROOT=/home/mrx/projects/matrix_completion/ntk_matrix_completion
TRUTH_DIR=$ROOT/utils/test

echo 'Creating test truth file'
python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --science