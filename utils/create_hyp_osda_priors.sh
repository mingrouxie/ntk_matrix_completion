source /home/mrx/.bashrc
source /home/mrx/bin/miniconda3/bin/activate /home/mrx/bin/miniconda3/envs/htvs
DJANGOCHEMDIR=/home/mrx/htvs/djangochem

CLUSTER_INBOX=/home/mrx/mnt/cluster/jobs/inbox
CLUSTER_COMPLETED=/home/mrx/mnt/cluster/jobs/completed
CLUSTER_STORAGE=/home/mrx/mnt/storage

export USER=$(whoami)

# OUTPUT_FILE=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/truths/testing
################################################

PRIOR_DIR=/home/mrx/projects/matrix_completion/ntk_matrix_completion/data/priors/hyp_osdas_others/

echo 'Creating prior file' 
FP_NAMES=(box volume axes whim getaway num_bonds num_rot_bonds asphericity eccentricity inertial_shape_factor spherocity_index gyration_radius pmi1 pmi2 pmi3 npr1 npr2 free_sasa bertz_ct)
python ntk_matrix_completion/features/create_prior_file.py --op $PRIOR_DIR --features "${FP_NAMES[@]}" --osda --batch_size 10000 --exc 1 --ms 180613_sda_organic 200525_spiro 200528_indolinium 200528_isoquinolinium 200529_imidazolium 200529_phenylammonium 200608_diels_alder 200612_test_stereo 200614_industry 200617_stereochemistry 200628_zach_neutral 200706_neutral_osda_vendors 200717_zach_organic 200721_zach_leftovers 200723_carbonization_test 200725_carbonization_sweep 200914_adamantane_sas 200914_double_quinuclidine_dabco 200914_doublespiro 200916_discovery_itq 201029_zach_sfw 210211_intergrowth_osda 210216_acetane_ig_osdas 210216_pyrrolidine_ig_osdas 210303_benzylbutyl_sfs 210324_spiro_patent 210331_deem_pnas 210331_okubo 210422_roman_aldolactonization 210428_denox 210502_TS_bigpore 210511_itq_bicyclic 210617_roman_glicolate 211022_osdas_intergrowth 211113_itq 211116_itq_diethylbenzene 220127_new_osdas 220510_itq_heart 220722_roman_lactones 221107_low_sodium_cha adamantane add_DB basf dabco deb denox diels_alder discovery doublespiro expt_active_learning generative_model imidazolium indolinium industry intergrowth —-isomer —-isomers isomers isoquinolinium itq lactones literature missing_smiles neutral_osda olivetti phenylammonium quinuclidine rdkit_osdas reference roman sdaitq smiles spiro stereochemistry tsguess vendors_osda zach_osda


# inside hyp_osdas: 210720_omar_diquaternary 210720_omar_quaternary
