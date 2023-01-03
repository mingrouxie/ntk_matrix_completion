source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/htvs
DJANGOCHEMDIR=/home/mrx/htvs/djangochem

CLUSTER_INBOX=/home/mrx/mnt/cluster/jobs/inbox
CLUSTER_COMPLETED=/home/mrx/mnt/cluster/jobs/completed
CLUSTER_STORAGE=/home/mrx/mnt/storage

export USER=$(whoami)

ROOT=/home/mrx/projects/matrix_completion/ntk_matrix_completion


echo "Creating data files from science and subset of zach's molecules, mean by inchikey" 
TRUTH_DIR=$ROOT/data/truths/221227/iza_gen_model_subset/mean_ik/
python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 1 --substrate iza_parse --ms 221227_generative_model_subset 2021_apriori_dskoda

echo "Creating data files from science and subset of zach's molecules, mean by zeolite" 
TRUTH_DIR=$ROOT/data/truths/221227/iza_gen_model_subset/mean_fw/
python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 1 --substrate iza_parse --index Zeolite --columns InchiKey --ms 221227_generative_model_subset 2021_apriori_dskoda 

echo "Creating data files from science and subset of zach's molecules, rsmall_pos" 
TRUTH_DIR=$ROOT/data/truths/221227/iza_gen_model_subset/small_pos/
python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 2 --substrate iza_parse --ms 221227_generative_model_subset --ms 221227_generative_model_subset 2021_apriori_dskoda 

echo "Creating data files from science and subset of zach's molecules, large_pos" 
TRUTH_DIR=$ROOT/data/truths/221227/iza_gen_model_subset/large_pos/
python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 3 --substrate iza_parse --ms 221227_generative_model_subset 2021_apriori_dskoda

echo "Creating data files from science and subset of zach's molecules, max_plus" 
TRUTH_DIR=$ROOT/data/truths/221227/iza_gen_model_subset/max_plus/
python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 4 --substrate iza_parse --ms 221227_generative_model_subset 2021_apriori_dskoda

echo "Creating data files from science and subset of zach's molecules, zero" 
TRUTH_DIR=$ROOT/data/truths/221227/iza_gen_model_subset/zero/
python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 5 --substrate iza_parse --ms 221227_generative_model_subset 2021_apriori_dskoda

####################################################################################

# echo "Creating data files from science and all of zach's molecules, row mean" 
# TRUTH_DIR=$ROOT/data/truths/221225/iza_all/mean
# python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 1 --substrate iza_parse --ms 200721_zach_leftovers 200628_zach_neutral 200717_zach_organic 201029_zach_sfw zach_osda 2021_apriori_dskoda

# echo "Creating data files from science and all of zach's molecules, SMALL_POS" 
# TRUTH_DIR=$ROOT/data/truths/221225/iza_all/small_pos
# python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 2 --substrate iza_parse --ms 200721_zach_leftovers 200628_zach_neutral 200717_zach_organic 201029_zach_sfw zach_osda 2021_apriori_dskoda

# echo "Creating data files from science and all of zach's molecules, LARGE_POS" 
# TRUTH_DIR=$ROOT/data/truths/221225/iza_all/large_pos
# python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 3 --substrate iza_parse --ms 200721_zach_leftovers 200628_zach_neutral 200717_zach_organic 201029_zach_sfw zach_osda 2021_apriori_dskoda

# echo "Creating data files from science and all of zach's molecules, MAX_PLUS" 
# TRUTH_DIR=$ROOT/data/truths/221225/iza_all/max_plus
# python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 4 --substrate iza_parse --ms 200721_zach_leftovers 200628_zach_neutral 200717_zach_organic 201029_zach_sfw zach_osda 2021_apriori_dskoda

# echo "Creating data files from science and all of zach's molecules, ZERO" 
# TRUTH_DIR=$ROOT/data/truths/221225/iza_all/zero
# python $ROOT/utils/create_truth.py --op $TRUTH_DIR --nan_after_nb drop --nb 5 --substrate iza_parse --ms 200721_zach_leftovers 200628_zach_neutral 200717_zach_organic 201029_zach_sfw zach_osda 2021_apriori_dskoda
