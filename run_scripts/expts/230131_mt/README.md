- run_1: multitasknncorr, cpu, 
- run_2: multitasknnsep, cpu

# 16 Feb 23
Input scaler: standard, energy_scaler: minmax, load_scaler: null but I think we need to scale it... 

## softmax is in the NN model, and not in the cross entropy loss

- run_3: full file, multitasknnsep, load single, cpu, 1000 epochs
- run_4: full file, multitasknncorr, load single, cpu, 1000 epochs
- run_5: full file, multitasknnsep_v2, load load, cpu, 1000 epochs
- run_6: full file, multitasknnsep_v2, load load_norm, cpu, 1000 epochs
- run_6_2: run_6, with 2000 epochs
- run_6_3: run_6, with 5000 epochs, with early stopping
- run_6_4: run_6, with 5000 epochs and 2 more layers size 18, with early stopping
- run_7: full file, multitasknncorr_v2, load load, cpu, 1000 epochs
- run_8: full file, multitasknncorr_v2, load load_norm, cpu, 1000 epochs
- run_9: repeat run_6, with more NN layers, smaller overfit file, 3000 epochs - try to overfit and see what happens
### With batch norm, on gpu unless otherwise stated
- run_10: run_3, but with 5 256 layers, 1000 epochs
- run_10_2: run_10, but with 5 256 layers, 1000 epochs, without batch norm
- run_10_3: run_10, but with 5 256 layers, 1000 epochs, with softmax (IDK why but just try)
- run_10_4: run_10, but with 5 256 layers, 1000 epochs, with dropout (test))
    - Better behaved than without dropout, but has occasional (crazy) spikes in val loss 
    - Change to softer early stopping (same as run_13_2)
- run_11: run_4, "
    - bug, rerun
    - Change to no dropout in last layer and softer early stopping (same as run_13_2)
- run_12: run_5, "
    - Val loss exploded at epoch 8
    - Change to no dropout in last layer and softer early stopping (same as run_13_2)
- run_13: run_6, "
    - Early stopping at epoch 9 but losses were stable (is patience too low?)
- run_13_2: run_13, but with patience 100 and min_delta 0.5
    - Seems to be going (1.5h)
- run_14: run_7, "
    - same folder name as 15. run 15 first then rerun. - job id 47101189
    - loss has spikes
    - Change to no dropout in last layer and softer early stopping (same as run_13_2) -  job id 47101713
- run_15: run_8, "
    - bug, rerun. same folder name as 14. rerun. 
    - loss also has spikes
    - Change to no dropout in last layer and softer early stopping (same as run_13_2)

However, just found out that the softmax is baked into the cross entropy loss. So technically, we don't need it in either place BUT it wasn't working when I removed softmax(dim=1) from both the NN and the loss function.

## 
