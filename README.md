# Installation

Run `pip install -r requirements.txt`.

# Usage

Run `bash ntk_matrix_completion/run_scripts/experiments.sh` after specifying the required parameters in `experiments.sh`. This codebase is still undergoing changes so this section will be updated accordingly. Currently, `experiments.sh` contains code to run XGBoost tuning and fitting. 

# Code structure

### config/ 
`config` contains weights for descriptors, as well as hyperparameter search spaces.

### data/
`data` contains prior files in the `prior` folder, as well as ground truth files. This can also be a folder to contain output files.

### features/ 
'features' contains `prior.py` for parsing prior information. Other scripts help to generate prior files from other source files and are mostly obsolete now.

### models/
`models` contains folders for each class of models. Currently, the models are `ntk`, `xgboost` and `two_tower_nn`.

### run_scripts/
`run_scripts` contains bash scripts to run multiple experiments easily, as well as python scripts that run cross-validation, feature selection, inference and so on. 

- `experiments.sh` is coded to run XGBoost hyperparameter tuning and model fitting and inference.
- The rest are a little outdated and need to be refactored.

### runs/
`runs/` contains some output scripts from two tower NN. This should be moved to `data`.

### tests/
`tests` contains scripts for testing the codebase. Currently only contains a script for testing isomer clustering.

### utils/
`utils` contains scripts used across the codebase. 
- `analysis_utilities` helps in data analysis
- `hyperopt` uses Hyperopt for parameter tuning
- `loss` contains custom loss functions
- `non-binding` contains treatment for non-binding entries
- `package_matrix` helps to format the ground truth
- `path_constants` contains file paths. This is in the process of being transformed into a script containing default file paths, as we move to using `argparse` for user-specified file paths. 
- `pca` contains code to do Principal Component Analysis
- `random_seed` contains a record of random seeds used. This is in the process of being transformed into a script containing default seeds, as we move to using `argparse` for user-specified file paths. 
- `utilities` contains miscellaneous common code. The most important is the chunking and isomer clustering used to create data splits
- `zeolite_scrapper` is used to scrap information from the IZA website. Note that the website format changed in July 2022


# Old README

* package_matrix.py: You'll want to run this with the binding.csv (https://github.com/learningmatter-mit/Zeolite-Phase-Competition/blob/main/data/binding.csv). Check out path_constants.py to see where that goes.
* precompute_osda_priors.py: This is what you use to populate OSDA priors before running the NTK. This can take a bit if you give it tons of OSDAs. Probably best to move to cluster.
* zeolite_scraper.py: scrapes https://america.iza-structure.org for features on all the zeolites hosted there. Built on beautiful soup and should be pretty easy to extend and scrape for additional info.
* zeo_1.py: Holds all of the features for ZEO-1 (thanks Mingrou!)

Meat of the algorithm:
* ntk.py: The actual implementation of the NTK & 10-fold cross validation using it.
* prior.py: The actual meat of where priors for zeolite & ntk are formatted (before you run this you must precompute the priors or scrape them with zeolite_scraper.py)
* utilities.py

Analysis:
* analysis_utilities.py: Computes all the good stuff like RMSE, spearman, top_k. Written to take into account non-binding cells.
* pca.py: This is where all the PCA plots with GETAWAY descriptors are generated. A bit messy right now, but definitely a good place to pull code from if you'd like to make some figures.

Actually user facing:
* train.py (used to be run_method_development.py): This file contains all of the tools for prior development. I.e. it just holds a ton of functions that run 10-fold cross validation with the ground truth energy matrix.
* run_predict.py: The code in this file was written after the algorithm/priors were looking good. It takes the complete ground_truth matrix as training and then outputs predictions for either new OSDA or new zeolite rows (depending whether or not you transpose).

Miscellaneous:
* eigen_pro/: this folder is work in progress for getting skinny_matrix to scale well.
* ooc_matrix_multiplication.py: also work in progress for doing the ntk on disc rather in RAM for when the matrices (specifically the kernels) get too big to handle.
* path_constants.py: this holds all of the path constants for .pkl/.csv data files, where the program expects them to be etc.



TL;DR The two commands you'll be running for the most part:
```python neural_tangent_kernel/run_predict.py```
```python neural_tangent_kernel/run_method_development.py```
All input & output data files are specified in `neural_tangent_kernel/path_constants.py`.

The files you'll care the most about at run-time:
* `neural_tangent_kernel/ntk.py`: Loads the data & does all the computation
* `neural_tangent_kernel/prior.py`: Prepares the priors (aka just loads priors pkl files)

The files you'll care the most about before run-time:
* `neural_tangent_kernel/precompute_prior.py`: precomputes OSDA priors with rdkit
* `neural_tangent_kernel/zeolite_scraper.py`: scrapes Zeolite priors from http://www.iza-structure.org/databases/ 
* `neural_tangent_kernel/package_matrix.py`: scrapes Zeolite priors from http://www.iza-structure.org/databases/ 


