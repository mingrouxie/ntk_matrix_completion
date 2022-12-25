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

echo 'Creating IZA truth files'

# TRUTH_DIR=$ROOT/data/truths/221216/iza_all/no_nb # DONE
# python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --substrate iza_parse #>> $TRUTHDIR/log 2>&1

###### TODO: could do these IF we figure out how to use k nearest neighbors, otherwise nope

# TRUTH_DIR=$ROOT/data/truths/221216/iza_all/mean
# python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 1 --substrate iza_parse # >> $TRUTHDIR/log 2>&1

# TRUTH_DIR=$ROOT/data/truths/221216/iza_all/small_pos
# python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 2 --substrate iza_parse #>> $TRUTHDIR/log 2>&1

# TRUTH_DIR=$ROOT/data/truths/221216/iza_all/large_pos
# python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 3 --substrate iza_parse #>> $TRUTHDIR/log 2>&1

# TRUTH_DIR=$ROOT/data/truths/221216/iza_all/max_plus
# python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 4 --substrate iza_parse #>> $TRUTHDIR/log 2>&1

# TRUTH_DIR=$ROOT/data/truths/221216/iza_all/zero
# python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 5 --substrate iza_parse #>> $TRUTHDIR/log 2>&1




