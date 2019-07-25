#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00


/gpfs/ysm/project/aj557/conda_envs/py36/bin/python evaluate_inception.py
