This is Yitong's remake of the README for Zeolite/OSDA matrix completion.

Requirements to run this repository can be found in requirements.txt. Just run: `pip install -r requirements.txt`
If any requirements are missing then try running: https://github.com/YitongTseo/OSDA_Generator/blob/main/setup.py
(sorry if there is anything missing ... please let me know!)

What you'll be running for the most part:
```python neural_tangent_kernel/two_step_ntk.py -i data/zeoliteTensor.pkl -i2 data/zeoliteNonbindingTensor.pkl -r -o```
Which just feeds the zeoliteTensor.pkl & zeoliteNonbindingTensor.pkl (the ground truth for the regression & binary classification tasks respectively)

The files you'll care the most about at run-time:
* `neural_tangent_kernel/two_step_ntk.py`: Loads the data & does all the computation
* `neural_tangent_kernel/prior.py`: Prepares the priors (aka just loads priors pkl files)

The files you'll care the most about at run-time:
* `neural_tangent_kernel/precompute_prior.py`: precomputes OSDA priors with rdkit
* `neural_tangent_kernel/zeolite_scraper.py`: scrapes Zeolite priors from http://www.iza-structure.org/databases/ 


TODOs: 
* Try the newly discovered Zeolite (I left TODO(Mingrou) in the code of prior.py and two_step_ntk.py to help with that).
* Work on the rest of the zeolites with parameters from 
* Keep cracking with as many priors for OSDAs & Zeolites as we can muster!
* Keep cracking on the skinny matrices to see if they can scale & combine row/col priors.
* Move the calculations of correctness outside of the run_ntk function... that's just sloppy. We can do that all after it runs with just the pooled results & and the true. We just have to make sure we split by rows/columns correctly...



* Unfortunately I remembered now why it doesn't work to run the row priors, then the col priors in series or in parallel. The problem comes from the fact that we need to partition the data into the same test & train for col & row priors. And taking out rows when the columns have the priors or vice versa doesn't make sense mathematically. the algorithm is written only to hold out rows with row priors or columns with column priors. 