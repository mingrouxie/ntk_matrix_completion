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

# FP_NAMES=(box volume axes whim getaway num_bonds num_rot_bonds asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct)
# python ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --features "${FP_NAMES[@]}" --osda --batch_size 10000 --truth_file $TRUTH_FILE --exc 1

# ################################################

# echo "Creating zeolite file for iza zeolites" DONE

# PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/

# FEATURES=(a b c alpha beta gamma num_atoms_per_vol num_atoms volume td cbu iza rdls td_10 ring_size_0 ring_size_1 ring_size_2 ring_size_3 ring_size_4 ring_size_5 ring_size_6 isdisordered isinterrupted framework_density largest_free_sphere accessible_volume_izc largest_free_sphere_a largest_free_sphere_b largest_free_sphere_c largest_free_sphere_izc largest_included_sphere largest_free_sphere_a_izc largest_free_sphere_b_izc largest_free_sphere_c_izc largest_included_sphere_a largest_included_sphere_b largest_included_sphere_c largest_included_sphere_fsp largest_included_sphere_izc)

# FWS_CONFIG=(iza_parse)
# # FWS=(MOR JZO)
# python /home/mrx/projects/matrix_completion/ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --zeolite --features "${FEATURES[@]}" --batch_size 10000 --fws_config "${FWS_CONFIG[@]}"
# # --fws "${FWS[@]}"

# ################################################

echo "Creating zeolite file for deem zeolites" DONE

PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/deem_zeolites

FEATURES=(a b c alpha beta gamma num_atoms largest_free_sphere largest_free_sphere_a largest_free_sphere_b largest_free_sphere_c largest_included_sphere largest_included_sphere_a largest_included_sphere_b largest_included_sphere_c largest_included_sphere_fsp n_tsites density relative_energy energy_per_tsite)

FWS_CONFIG=(ase_db_parse)
# FWS=(MOR JZO)
python /home/mrx/projects/matrix_completion/ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --zeolite --features "${FEATURES[@]}" --batch_size 10000 --fws_config "${FWS_CONFIG[@]}"
# --fws "${FWS[@]}"

# ################################################

# echo "Creating zeolite file for stack zeolites" DONE

# PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/stack_zeolites

# FEATURES=(a b c alpha beta gamma num_atoms cage_1 cage_2 cage_3 cage_4 cage_5 cage_6 cage_7 cage_8 cage_9 cage_10 density space_group relative_energy stacking_layers stacking_sequence largest_free_sphere stacking_compactness largest_free_sphere_a largest_free_sphere_b largest_free_sphere_c largest_channel_opening largest_included_sphere largest_included_sphere_a largest_included_sphere_b largest_included_sphere_c largest_included_sphere_fsp)

# FWS_CONFIG=(stacking_parse)
# # FWS=(MOR JZO)
# python /home/mrx/projects/matrix_completion/ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --zeolite --features "${FEATURES[@]}" --batch_size 10000 --fws_config "${FWS_CONFIG[@]}"
# # --fws "${FWS[@]}"

# # # ################################################

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
# FP_NAMES=(box volume axes whim getaway num_bonds num_rot_bonds asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct)
# python ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --features "${FP_NAMES[@]}" --osda --batch_size 10000 --truth_file $TRUTH_FILE --exc 1

# # # ################################################

# TRUTH_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_lit/
# TRUTH_FILE=$TRUTH_DIR/energies.csv
# PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/iza_lit/

# echo 'Creating truth file for lit'
# python /home/mrx/projects/matrix_completion/ntk_matrix_completion/utils/create_truth.py --op $TRUTH_DIR --ms literature --substrate iza_parse --exc 1 --nb 1 --nan_after_nb drop

# echo 'Creating OSDA prior file for lit based on truth file' 
# FP_NAMES=(box volume axes whim getaway num_bonds num_rot_bonds asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct)
# python ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --features "${FP_NAMES[@]}" --osda --batch_size 10000 --truth_file $TRUTH_FILE --exc 1

# # ################################################

# TRUTH_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/iza_hyp/
# TRUTH_FILE=$TRUTH_DIR/energies.csv
# PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/iza_hyp/

# echo 'Creating truth file for hyp'
# python /home/mrx/projects/matrix_completion/ntk_matrix_completion/utils/create_truth.py --op $TRUTH_DIR --ms 210720_omar_quaternary 210720_omar_diquaternary --substrate iza_parse --exc 1 --nb 1 --nan_after_nb drop

# echo 'Creating OSDA prior file for hyp based on truth file' 
# FP_NAMES=(box volume axes whim getaway num_bonds num_rot_bonds asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct)
# python ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --features "${FP_NAMES[@]}" --osda --batch_size 10000 --truth_file $TRUTH_FILE --exc 1

