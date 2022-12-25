source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/htvs
DJANGOCHEMDIR=/home/mrx/htvs/djangochem

CLUSTER_INBOX=/home/mrx/mnt/cluster/jobs/inbox
CLUSTER_COMPLETED=/home/mrx/mnt/cluster/jobs/completed
CLUSTER_STORAGE=/home/mrx/mnt/storage

export USER=$(whoami)

ROOT=/home/mrx/projects/matrix_completion/ntk_matrix_completion
# TRUTH_DIR=$ROOT/utils/test

# oops forgot to stdout to log file owell

echo 'Creating Science truth file'
TRUTH_DIR=$ROOT/data/truths/221216/science/no_nb
python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --science

TRUTH_DIR=$ROOT/data/truths/221216/science/mean
python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --science --nb 1

TRUTH_DIR=$ROOT/data/truths/221216/science/small_pos
python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --science --nb 2

TRUTH_DIR=$ROOT/data/truths/221216/science/large_pos
python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --science --nb 3

TRUTH_DIR=$ROOT/data/truths/221216/science/max_plus
python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --science --nb 4

TRUTH_DIR=$ROOT/data/truths/221216/science/zero
python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --science --nb 5



