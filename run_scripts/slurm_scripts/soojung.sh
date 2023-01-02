#!/bin/bash
#SBATCH --job-name=train_model_3
#SBATCH --partition=sched_mit_rafagb
#SBATCH --time=5:00:00
#SBATCH --output=eofe_train_%j.log
#SBATCH -n 20 #Request 4 tasks (cores)
#SBATCH -N 1 #Request 1 node
#SBATCH --gres=gpu:1

#SBATCH --exclusive

# for engaging

python train_model.py
