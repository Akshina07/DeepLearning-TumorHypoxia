#!/bin/bash
#SBATCH --job-name=m_multiLayer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=pi_gerstein
#SBATCH --time=10-00:00:00
#SBATCH --mem=100000



/gpfs/ysm/project/aj557/conda_envs/py36/bin/python model_multilayer.py
