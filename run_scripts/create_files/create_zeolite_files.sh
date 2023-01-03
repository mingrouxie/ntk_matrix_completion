source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/htvs
DJANGOCHEMDIR=/home/mrx/htvs/djangochem

CLUSTER_INBOX=/home/mrx/mnt/cluster/jobs/inbox
CLUSTER_COMPLETED=/home/mrx/mnt/cluster/jobs/completed
CLUSTER_STORAGE=/home/mrx/mnt/storage

export USER=$(whoami)

z_map=(a b c alpha beta gamma num_atoms_per_vol num_atoms volume largest_free_sphere largest_free_sphere_a largest_free_sphere_b largest_free_sphere_c largest_included_sphere largest_included_sphere_a largest_included_sphere_b largest_included_sphere_c largest_included_sphere_fsp)

# echo "Creating zeolite file for iza zeolites" 

# PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/zeolites_iza

# FEATURES=(a b c alpha beta gamma num_atoms_per_vol num_atoms volume largest_free_sphere largest_free_sphere_a largest_free_sphere_b largest_free_sphere_c largest_included_sphere largest_included_sphere_a largest_included_sphere_b largest_included_sphere_c largest_included_sphere_fsp td cbu iza rdls td_10 ring_size_0 ring_size_1 ring_size_2 ring_size_3 ring_size_4 ring_size_5 ring_size_6 isdisordered isinterrupted framework_density largest_free_sphere_izc largest_free_sphere_a_izc largest_free_sphere_b_izc largest_free_sphere_c_izc largest_included_sphere_izc largest_included_sphere_a_izc largest_included_sphere_b_izc largest_included_sphere_c_izc)

# FWS_CONFIG=(iza_parse)
# # FWS=(MOR JZO)
# python /home/mrx/projects/matrix_completion/ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --zeolite --features "${FEATURES[@]}" --batch_size 10000 --fws_config "${FWS_CONFIG[@]}"
# # --fws "${FWS[@]}"

# ################################################

echo "Creating zeolite file for deem zeolites" 

PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/zeolites_deem

FEATURES=(a b c alpha beta gamma num_atoms_per_vol num_atoms volume largest_free_sphere largest_free_sphere_a largest_free_sphere_b largest_free_sphere_c largest_included_sphere largest_included_sphere_a largest_included_sphere_b largest_included_sphere_c largest_included_sphere_fsp n_tsites density relative_energy energy_per_tsite)

FWS_CONFIG=(ase_db_parse)
# FWS=(MOR JZO)
python /home/mrx/projects/matrix_completion/ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --zeolite --features "${FEATURES[@]}" --batch_size 10000 --fws_config "${FWS_CONFIG[@]}"
# --fws "${FWS[@]}"

# ################################################

echo "Creating zeolite file for stack zeolites" 

PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/zeolites_stack

FEATURES=(a b c alpha beta gamma num_atoms_per_vol num_atoms volume largest_free_sphere largest_free_sphere_a largest_free_sphere_b largest_free_sphere_c largest_included_sphere largest_included_sphere_a largest_included_sphere_b largest_included_sphere_c largest_included_sphere_fsp cage_1 cage_2 cage_3 cage_4 cage_5 cage_6 cage_7 cage_8 cage_9 cage_10 density space_group relative_energy stacking_layers stacking_sequence stacking_compactness a_parsed b_parsed c_parsed)

FWS_CONFIG=(stacking_parse)
# FWS=(MOR JZO)
python /home/mrx/projects/matrix_completion/ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --zeolite --features "${FEATURES[@]}" --batch_size 10000 --fws_config "${FWS_CONFIG[@]}"
# --fws "${FWS[@]}"