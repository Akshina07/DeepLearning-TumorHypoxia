import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
import csv 
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
path_data_train="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_TRAIN.csv"
path_data_test="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_TEST.csv"
path_labels_train="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABELS_TRAIN.csv"
path_labels_test="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABELS_TEST.csv"
# PATH TO BALANCED SETS 	

valid_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_VALID_B.csv"
train_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_ONLYTRAIN_B.csv"
valid_label_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABEL_VALID_B.csv"
train_label_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABEL_ONLYTRAIN_B.csv"
count_1_train=0
count_0_train=0
count_1_valid=0
count_0_valid=0

def sep_valid_train(label_chunk,data_chunk):
    valid_d=[]
    valid_l=[]
    train_d=[]
    train_l=[]
    #test_d=[]
    test_l=[]
    #global count_1_train,count_0_train,count_1_valid,count_valid
    for i in range(len(label_chunk)):
       file=label_chunk[i][0]
       label=label_chunk[i][1]
       #print(file)
       if(file[0]=="v"):
           global count_1_valid
           global count_0_valid
           if(label==1):
               if(count_1_valid==75000): 
                   continue
               count_1_valid+=1
           else:
               if(count_0_valid==75000):
                   continue
               count_0_valid+=1
           valid_d.append(data_chunk[i][1:2050])
           valid_l.append(label_chunk[i,0:2])
       else:
           global count_1_train
           global count_0_train
           if(label==1):
               if(count_1_train==350000):
                   continue
               count_1_train+=1
           else:
               if(count_0_train==350000):
                   continue
               count_0_train+=1
           train_d.append(data_chunk[i,1:2050])
           train_l.append(label_chunk[i,0:2])
    if(np.array(valid_d).shape[0]!=[0]):
        with open(valid_set,"a") as f:
            pd.DataFrame(valid_d).to_csv(f,sep=",",header=False)
    if(np.array(valid_l).shape[0]!=[0]):
        with open(valid_label_set,"a") as f:
            pd.DataFrame(valid_l).to_csv(f,sep=",",header=False)
    if(np.array(train_d).shape[0]!=[0]):
        with open(train_set,"a") as f:
            pd.DataFrame(train_d).to_csv(f,sep=",",header=False)
    if(np.array(train_l).shape[0]!=[0]):
         with open(train_label_set,"a") as f:
            pd.DataFrame(train_l).to_csv(f,sep=",",header=False)


    
if __name__=="__main__":

    labels=pd.read_csv(path_labels_train,sep=",",chunksize=15000)
    i=0
    for chunk in  pd.read_csv(path_data_train,sep=",",chunksize=15000):
        #if(i==2):
        #    break
        c_label=np.array(next(iter(labels)))[:,1:3]
        print("Done"+str(i+1))
        sep_valid_train(c_label,np.array(chunk))
        i+=1
    
