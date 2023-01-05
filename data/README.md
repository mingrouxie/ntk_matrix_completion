# data README

- daniels_data
- fws_ls
    - science paper frameworks
- handcrafted
    - Old descriptors scrapped by Yitong from IZA website and ZEO-1
    - iza_zeolites.pkl was used in expts_xgb.sh when comparing non-binding treatment and descriptor treatment
- logs
    - baseline_log 
- nick_persistent: persistent homology priors created by Nick
    - from_gdrive should have the newest ones
- priors: contains files created with code in this repo
- swagata_gcnn: GCNN embeddings from Swagata
- truths: truth files created with code in this repo

## priors/ 

### TLDR

- Use 221216/ for OSDA prior files
- Use zeolite_priors_20221118_25419.pkl for zeolite structural prior file

### Relevant

- 221216: prior files created, this should be the _FINAL_ one to be used forever
    - iza_all: 
    - science: 1 file, science paper priors
- 221221_hyp_osdas_omar: 153 prior files - only Omar's quat and diquats
- 221221_hyp_osdas_others: 8 prior files to make use of other molecules also in the database
- IZC_conformer_priors_clipped.pkl: used for IZC 2022 poster
    - IZC_conformer_priors_sieved_getaway: only priors that had reasonable GETAWAY
    - IZC_conformer_priors.pkl: original with nothing altered I THINK
    - IZC_docked_priors.pkl: docked OSDA priors
    - iza_parse_literature_docked_priors.pkl: must have been a pain to extract this
- zeolite_ohe.pkl: used in expts_xgb.sh
- zeolite_priors_20221118_25419.pkl: used in expts_xgb_kfi.sh when we tried to retrain the model but still using Science paper dataset

### Unknown or relic

- science_paper: why is this empty
- zeolite_ignore
- conformer_priors.pkl