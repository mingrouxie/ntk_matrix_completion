# run_scripts record

## expts

- expts_ntk.sh: Training model with NTK pre Aug 22
- expts_xgb.sh: Attempt 1 at training model (was used for Oct 22 CC GM etc.)
- expts_xgb_kfi.sh: Attempt 2 after some major code rewiring
- expts_xgb_2.sh: new truth and prior files. Didn't actually fully use this before setting up Engaging

- 230103_xgb_1.sh: Testing XGB training on Engaging to see if results are the same as on deepware (default nthread which is 6, although -n is 12)
- 230103_xgb_2.sh: 230103_xgb_1 but with -n 24 and --nthread 24
- 230103_xgb_3.sh: Testing XGB training on Engaging 

- 230123_mt_1.sh: Multitask model - currently WIP

## create_files

- create_files.sh: 
- create_files_2.sh: create truth files for Science paper dataset, various nb treatments
- create_files_3.sh: attempt to create truth files for all OSDAs and IZA-only zeolites but only got the no-nb treatment one
- create_files_4.sh: OSDA prior file for all IZA datapoints
- create_fles_5_zach.sh: retrieve Science paper and Zach OSDAs' binding energy data
- create_hyp_osda_priors.sh: create prior files for all the hyp OSDA files to do inference on
- zeo_1.py: data file for ZEO-1, outdated but just keeping it for now

## inference

- predict_xgb.sh: script for inference for given ligands and substrates

## feature_selection

- substrate_prior_selection.py: Selecting persistent homology features