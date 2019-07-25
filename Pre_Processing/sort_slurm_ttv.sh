#!/bin/bash
#SBATCH --partition=pi_gerstein
#SBATCH --job-name=sort_ttv
#SBATCH --ntasks=1 --nodes=1
#SBACTCH --mem=50000
#SBATCH --time=24:00:00


/gpfs/ysm/project/aj557/conda_envs/openslide-python/bin/python submit_sorting_ttv_jobs.py


