# Prepare Data
python prepare_data.py

# Trains DNPP on Sparse Subset (Seed 149)
python dnpp/dnpp.py -b 2 -k 0 -s 149

# Trains DNPP on 10-Fold for All Data (Seed 149)
python dnpp/dnpp.py -s 149

# Trains Means on 10-Fold for All Data (Seed 149)
python means/means.py -s 149

# Train NTK with Zeolite dataset for 10-Fold 
# this is the one with really skinny matrices...
# python neural_tangent_kernel/two_step_ntk.py -i /Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/zeoliteOSDAIndexedMatrix.pkl -i2 /Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/zeoliteOSDANonbindingIndexedMatrix.pkl -r -o
# python neural_tangent_kernel/two_step_ntk.py -i /Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/zeoliteOSDAIndexedMatrix.pkl -i2 /Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/smaller_skinny_matrix.pkl -r -o


# This is the one with regular matrices
python neural_tangent_kernel/two_step_ntk.py -i /Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/zeoliteTensor.pkl -i2 /Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/zeoliteNonbindingTensor.pkl -r -o
# python neural_tangent_kernel/two_step_ntk.py -i /Users/yitongtseo/Documents/GitHub/ntk_matrix_completion/cmap_imputation/data/moleules_from_daniel/prior_precomputed_energies_100by196.pkl -r -o

# Train NTK with MCF7 + Cell Averages for 10-Fold (Seed 149)
python neural_tangent_kernel/ntk.py -s 149 -x only_train_cell_average

# Train NTK with Combination DNPP and MCF7 Priors (Seed 149)
python neural_tangent_kernel/dnpp_mcf7_mixed_prior.py -s 149

# Train NTK with Combination FaLRTC and MCF7 Priors (Seed 149)
python neural_tangent_kernel/falrtc_mcf7_mixed_prior.py -s 149

# Generate Comparative Graphics for DNPP vs Combination (Seed 149)
python graphical/comparative_graphics.py -d ./predictions/DNPPKFold10Folds149SeedlargeTensorTrainSourceAllMetrics.pkl -n ./predictions/MixedDNPPNTKKFold10Folds149SeedlargeTensorTrainSourceAllMetrics.pkl -s 149
