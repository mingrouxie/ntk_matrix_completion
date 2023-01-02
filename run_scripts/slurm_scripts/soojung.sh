#!/bin/bash
#SBATCH --job-name=eofe_sj
#SBATCH --partition=sched_mit_rafagb
#SBATCH --time=1:00:00
#SBATCH --output=eofe_train_%j.log
#SBATCH -n 10 #Request 4 tasks (cores)
#SBATCH -N 1 #Request 1 node

#SBATCH --exclusive

# for engaging

echo `date`
echo 'hi'

# python train_model.py
