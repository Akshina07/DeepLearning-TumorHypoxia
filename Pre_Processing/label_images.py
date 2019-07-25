'''
LABEL IMAGES. MOVE IMAGES INTO N SUBDIRECTORIES IE ONE FOR EACH LABEL
FOR EG: IF YOU NEED TO CLASSIFY INTO 2 CLASSES 0/1 DATA WILL BE SAVED AS :
	/DATA/LABEL_0
	/DATA/LABEL_1
'''
import numpy as np
#import openslide
import os
import sys
from glob import glob
from argparse import ArgumentParser
import random
import numpy as np
from shutil import copyfile
import pandas as pd
from pandas import DataFrame


if __name__ == '__main__':
    descr= """
        enter path name to folder containing tiled images
    """
    parser = ArgumentParser(description=descr)
    parser.add_argument("--SourceFolder", help="path to sorted images", dest='SourceFolder') #FOLDER CONTAINING ALL THE IMAGES
    args = parser.parse_args()
    df=pd.read_excel("/ysm-gpfs/pi/gerstein/aj557/data_deeppath/labels.xlsx") #the excel file contains the patient ID's mapped to the labels. Labels were generated using z values and median scores. refer the paper and supplementary texts to define thresholds for hypoxic and non-hypoxic classification.
    df=np.asarray(df)
    #print(df)
    
    SourceFolder = os.path.abspath(args.SourceFolder)
    os.system("mkdir {}/label_0".format(SourceFolder))
    os.system("mkdir {}/label_1".format(SourceFolder))
    #print(df.shape)

    SourceFolder = os.path.abspath(args.SourceFolder)
    image_files=glob(os.path.join(SourceFolder, "*.jpeg"))
    for file in image_files:
        file_root = os.path.basename(file)
        #file_root = file_root.replace('_files', '')
        if(file_root[0]=='v'): # valid    #EXTRACT THE PATIENT ID'S FROM THE FILE NAMES ( PATIENT ID IS A 12 CHARACTER SEQUENCE IN THE SLIDE NAME )
            patient_id=file_root[6:18]
        elif(file_root[1]=='r'): #train
            patient_id=file_root[6:18]
        else:#test
            patient_id=file_root[5:17]
        #print(patient_id)
        index=np.where(df==patient_id)[0]   #FIND THE INDICES WHERE THE PATIENT ID OF THE SLIDE MATCHES THE LABELS LIST. THIS IS GENERALLY A ONE TO ONE MAPPING 
        for i in index:
            #print(index)
            #print(file)
            if(df[i][5]==1):
                os.system("mv {} {}/label_1".format(file,SourceFolder))
            else:
                os.system("mv {} {}/label_0".format(file,SourceFolder))
    print("End")

        
#print(image_file)
