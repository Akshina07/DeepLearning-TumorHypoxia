#!/bin/bash
#SBATCH --job-name=hyperp_tune
#SBATCH --ntasks=1
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --partition=pi_gerstein
#SBATCH --mem=100000
#SBATCH --time=7-00:00:00


/gpfs/ysm/project/aj557/conda_envs/py36/bin/python hyper_param_tuning.py
