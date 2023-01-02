#!/bin/bash
#SBATCH --job-name=test_eofe7
#SBATCH --output=./test_eofe7_%j.log 
#SBATCH --partition=sched_mit_rafagb,sched_mit_hill,sched_opportunist
#SBATCH --time=00:10:00                  # hh:mm:ss, alternatively 1-00:00 dd-hh:mm
#SBATCH -n 12                           # cores. 8 for voronoi, 1 for dreiding. Doing 12 now and I think my code does nthread=5 for xgb
#SBATCH -N 1                            # nodes. 1 for voronoi, 1 for dreiding
#SBATCH --mem-per-cpu=2000              # mb. 3800 for voronoi, 1000 for dreiding


echo `date`
echo hi
