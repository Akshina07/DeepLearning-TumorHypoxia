'''
 OBJECTIVE: SORT THE DIRECTORY OF IMAGES IN ORDER OF INCREASING NUMBER OF TILES
 '''

from __future__ import print_function
import json
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
from optparse import OptionParser
import re
import shutil
from unicodedata import normalize
import numpy as np
import scipy.misc
import subprocess
from glob import glob #filename path expansion
from multiprocessing import Process, JoinableQueue
import time
import os
import sys

from xml.dom import minidom
from PIL import Image, ImageDraw #Pillow module for reading images and handling them


if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] <slide>') # parser is to read command line arguments and also ssetup arguments or options for a command line script
    
    parser.add_option('-L', '--ignore-bounds', dest='limit_bounds',default=True, action='store_false',help='display entire scan area')
    parser.add_option('-e', '--overlap', metavar='PIXELS', dest='overlap',type='int', default=1,help='overlap of adjacent tiles [1]')
    parser.add_option('-s', '--size', metavar='PIXELS', dest='tile_size',type='int', default=254,help='tile size [254]')
    parser.add_option('-B', '--Background', metavar='PIXELS', dest='Bkg',type='float', default=50,help='Max background threshold [50]; percentager of background allowed')
    #parser.add_option('
                      
    (opts, args) = parser.parse_args()
                          
    try:
        slidepath = args[0]
    except IndexError:
        parser.error('Missing slide argument')
    '''
    if opts.basename is None:
        opts.basename = os.path.splitext(os.path.basename(slidepath))[0] # possibly extracting the base directory
    '''
    print(opts.basename)
    print(args)
    print(args[0])
    #print(slidepath)
    files = glob(slidepath)
    print(files)
    print("***********************")
    array_count=[]
    for imgNb in range(len(files)):
        filename = files[imgNb]
        opts.basenameJPG = os.path.splitext(os.path.basename(filename))[0]
        print("processing: " + opts.basenameJPG)
        image = open_slide(filename)
        dz=DeepZoomGenerator(image, opts.tile_size, opts.overlap, limit_bounds=opts.limit_bounds)
        array_count.append([dz.tile_count,filename])
    array_count=sorted(array_count, key=lambda array_count: array_count[0])
    #print(max(array_count[0][:]))
    i=0
    dirc=1
    while(i<len(array_count)):
    	count=0
	dir_name="dir_{}".format(dirc)
	command_1="mkdir /ysm-gpfs/pi/gerstein/aj557/data_deeppath/data_slices/{}".format(dir_name)
	os.system(command_1)
        while(count<50 and i<len(array_count)):
            # print(i)
	    content=array_count[i][1]
	    name_1=os.path.splitext(content)[0]
            name_2=os.path.splitext(name_1)[0]
            name_3=os.path.split(name_2)[0]
	    name=os.path.split(name_3)[1]
            #print(name)
    	    command_2="mv /ysm-gpfs/pi/gerstein/aj557/data_deeppath/DATA_SORTED2/dir_*/{}/  /ysm-gpfs/pi/gerstein/aj557/data_deeppath/data_slices/{}/".format(name,dir_name)
            #print(command_2)
	    os.system(command_2)
            count=count+1
            i=i+1
        dirc=dirc+1
    print("End")
