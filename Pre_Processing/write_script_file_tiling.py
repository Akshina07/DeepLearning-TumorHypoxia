from optparse import OptionParser
import numpy as np
import scipy.misc
import subprocess
from glob import glob #filename path expansion
import time
import os
import sys


if __name__ == '__main__':
    for i in range(262): #number of directories
        slidepath="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/data_slices/dir_{}/*/*.svs".format(i+1) #Path to all the svs images in a particular directory
        filepath="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/data_slices/dir_{}/tiling_{}.sh".format(i+1,i+1) #Path to the tiling script
        out_path="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/data_slices/dir_{}/tile_out{}".format(i+1,i+1) #Path to the output folder which will contain all the tiles
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        files=glob(slidepath)
        j=0
        for image in files:
            j=j+1
            if(j==1 or j%10==0):  #This is to write one script per 10 images ( can be modified to run one script for all 50 images in one go.
                filepath="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/data_slices/dir_{}/tiling_{}-{}.sh".format(i+1,i+1,j+1)
                f_tile=open(filepath,"w+")
                f_tile.write("#!/bin/bash\n#SBATCH --partition=pi_gerstein\n")
                f_tile.write("#SBATCH --job-name=tt_{}-{}\n".format(i+1,j+1))
                f_tile.write("#SBATCH --ntasks=1 --nodes=1\n#SBATCH --time=48:00:00\n#SBATCH --mem=20000\n") #20 GB in case of 10 images. The memory specification depends on the over directory number as they are sorted based on tile number. For 50 images -> dir 1-50 : 30GG, dir 50-150 : 40 GB, dir 150-262 : 80GB 
            f_tile.write("python deepzoom_tile.py  -s 299 -e 0 -j 32 -B 25  --output=\"/ysm-gpfs/pi/gerstein/aj557/data_deeppath/data_slices/dir_{}/tile_out{}\" ".format(i+1,i+1)+image+"\n\n")
           

