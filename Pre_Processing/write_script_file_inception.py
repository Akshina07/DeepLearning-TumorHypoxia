'''
-------CREATE_SLURM_SCRIPTS_FOR_GENERATION_OF_TRANSFER_VALUES--------------------------
CREATE BASH SCRIPTS FOR EACH DIRECTORY CONTAINING 50 IMAGES EACH TO PROCESS THROUGH INCEPTION.PY TO GENERATE TRANSFER VALUES
->MAKE SURE YOU SPECIFY ENOUGH MEMORY FOR EACH CSV ( NOT MORE THAN 3 GB ON AVERAGE)
->TIME LIMIT UPPER BOUND: 2 DAYS 
'''

from optparse import OptionParser
import numpy as np
import scipy.misc
import subprocess
from glob import glob #filename path expansion
import time
import os
import sys

if __name__ == '__main__':
	for i in range(1):
        	filepath="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/data_slides_resnet/transfer_full.sh"  # Transfer values saved in a CSV for each directory
		f_tile=open(filepath,"w+")
        	f_tile.write("#!/bin/bash\n#SBATCH --partition=pi_gerstein\n")
        	f_tile.write("#SBATCH --job-name=ttfull\n")
		f_tile.write("#SBATCH --ntasks=1 --nodes=1\n#SBATCH --cpus-per-task=2\n#SBATCH --gres=gpu:k80:2\n#SBATCH --time=2-00:00:00\n#SBATCH --mem=20000")
		#f_tile.write("#srun --pty -p gpu -c 2 -t 5 --gres=gpu:1 bash\n")
		f_tile.write("/gpfs/ysm/project/aj557/conda_envs/py36/bin/python /ysm-gpfs/pi/gerstein/aj557/data_deeppath/inception_extract_transfer_values.py  \n")
