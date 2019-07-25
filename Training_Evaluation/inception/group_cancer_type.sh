#!/bin/bash
#SBATCH --partition=pi_gerstein
#SBATCH --job-name=g-cancer
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=10000

/gpfs/ysm/project/aj557/conda_envs/py36/bin/python group_cancer_type.py --n1=0 --n2=262
