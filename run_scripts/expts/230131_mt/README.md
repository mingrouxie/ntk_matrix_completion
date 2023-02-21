- run_1: multitasknncorr, cpu, 
- run_2: multitasknnsep, cpu

# 16 Feb 23
Input scaler: standard, energy_scaler: minmax, load_scaler: null but I think we need to scale it... 

## softmax is in the NN model, and not in the cross entropy loss

- run_3: full file, multitasknnsep, load single, cpu, 1000 epochs
- run_4: full file, multitasknncorr, load single, cpu, 1000 epochs
- run_5: full file, multitasknnsep_v2, load load, cpu, 1000 epochs
- run_6: full file, multitasknnsep_v2, load load_norm, cpu, 1000 epochs
- run_7: full file, multitasknncorr_v2, load load, cpu, 1000 epochs
- run_8: full file, multitasknncorr_v2, load load_norm, cpu, 1000 epochs

However, just found out that the softmax is baked into the cross entropy loss. So technically, we don't need it in either place BUT it wasn't working when I removed softmax(dim=1) from both the NN and the loss function.

## 