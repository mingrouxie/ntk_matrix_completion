source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/htvs
DJANGOCHEMDIR=/home/mrx/htvs/djangochem

CLUSTER_INBOX=/home/mrx/mnt/cluster/jobs/inbox
CLUSTER_COMPLETED=/home/mrx/mnt/cluster/jobs/completed
CLUSTER_STORAGE=/home/mrx/mnt/storage

export USER=$(whoami)

hostname

ROOT=/home/mrx/projects/affinity/ntk_matrix_completion
DATAROOT=/home/mrx/projects/affinity_pool/ntk_matrix_completion

TRUTH_DIR=$DATAROOT/data/truths/testing_3/new # got deleted, but it was re-saved in testing_7
python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 5 --cs 2021_apriori_dskoda --nan_after_nb drop