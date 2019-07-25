#!/bin/bash
#SBATCH --job-name=m_2l_bal
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
##SBATCH --mem=50000


/gpfs/ysm/project/aj557/conda_envs/py36/bin/python model_2layer_balanced.py
