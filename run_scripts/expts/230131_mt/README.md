- run_1: multitasknncorr, cpu, 
- run_2: multitasknnsep, cpu

16 Feb 23
Going to run everything at 1000 epochs because... yeah... 
Input scaler: standard, energy_scaler: minmax, load_scaler: null but I think we need to scale it... 
- run_3: full file, multitasknnsep, load single, cpu
- run_4: full file, multitasknncorr, load single, cpu
- run_5: full file, multitasknnsep_v2, load load, cpu
- run_6: full file, multitasknnsep_v2, load load_norm, cpu
- run_7: full file, multitasknncorr_v2, load load, cpu
- run_8: full file, multitasknncorr_v2, load load_norm, cpu
