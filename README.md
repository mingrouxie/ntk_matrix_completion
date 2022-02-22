This is Yitong's remake of the README for Zeolite/OSDA matrix completion.

Requirements to run this repository can be found in requirements.txt. Just run: `pip install -r requirements.txt`
If any requirements are missing then try running: https://github.com/YitongTseo/OSDA_Generator/blob/main/setup.py
(sorry if there is anything missing ... please let me know or add it)


Here's the outline of the relevant parts:
Data generation:
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
* run_method_development.py: This file contains all of the tools for prior development. I.e. it just holds a ton of functions that run 10-fold cross validation with the ground truth energy matrix.
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


