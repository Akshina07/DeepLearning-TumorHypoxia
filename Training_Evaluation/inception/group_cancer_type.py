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

data_complete=[]
cancer_type=np.array(pd.read_csv("/ysm-gpfs/pi/gerstein/aj557/data_deeppath/cancer_type.csv",sep=","))
def group_tumor_type(tumor,data):
    data_train=[] #training and validation dataset for BRCA cancer type
    data_test=[]  # test set
    labels_train=[] #labels for traning and validation dataset for BRCA cancer type
    labels_test=[]  # true labels for test set
    #print(tumor)
    #print(cancer_type.shape)
    for i in range(data.shape[0]):
        file=data[i][2049]
        if(file[0]=='v' or file[1]=='r'): # train and valid data
            patient_id=file[6:18]
        else:#test data
            patient_id=file[5:17]
        #print(patient_id)
        index=np.where(cancer_type==patient_id)[0]
        #print(index)
        #print(cancer_type[index[0]][0])
        for j in index:
            if(cancer_type[j][1]==tumor):
                if(file[0]=='v' or file[1]=='r'):
                    data_train.append(data[i][1:2050])
                    labels_train.append(data[i][2049:2051])
                else:
                    data_test.append(data[i][1:2050])
                    labels_test.append(data[i][2049:2051])
            else:
                #print("Hello")
                continue
    if(np.array(data_train).shape[0]!=[0]):
        with open('/ysm-gpfs/pi/gerstein/aj557/data_deeppath/{}_DATA_TRAIN.csv'.format(tumor),'a') as f:
            pd.DataFrame(data_train).to_csv(f,sep=',',header=False)
    if(np.array(data_test).shape[0]!=[0]):
        with open('/ysm-gpfs/pi/gerstein/aj557/data_deeppath/{}_DATA_TEST.csv'.format(tumor),'a') as f:
            pd.DataFrame(data_test).to_csv(f,sep=',',header=False)
    if(np.array(labels_train).shape[0]!=[0]):
        with open('/ysm-gpfs/pi/gerstein/aj557/data_deeppath/{}_LABELS_TRAIN.csv'.format(tumor),'a') as f:
            pd.DataFrame(labels_train).to_csv(f,sep=',',header=False)
    if(np.array(labels_test).shape[0]!=[0]):
        with open('/ysm-gpfs/pi/gerstein/aj557/data_deeppath/{}_LABELS_TEST.csv'.format(tumor),'a') as f:
            pd.DataFrame(labels_test).to_csv(f,sep=',',header=False)  

if __name__=="__main__":
    #LOADING DATA AND GROUPING DATA ON THE BASIS OF TUMOR TYPE 
    #data_complete=pd.DataFrame([])
    chunks=[]
    parser = ArgumentParser()
    parser.add_argument("--n1", help="directory number", dest='num1')
    parser.add_argument("--n2", help="directory number", dest='num2')
    parser.add_argument("--type",help=" type of tumor ", dest='tumor')
    args = parser.parse_args()
    num1=int(args.num1)
    num2=int(args.num2)
    tumor=str(args.type)
    for i in range(num1,num2):
        path="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/CSV/transfer_values{}.csv".format(i+1) # path to the transfer value csv 
        for chunk in pd.read_csv(path,sep=",",chunksize=10000): # process the data in chunk as the entire data set cannot fit in memory at one go
            #print(i+1)
            group_tumor_type(tumor,np.array(chunk))
        #data_complete=pd.concat([data_complete,chunk],sort=False)
       	print("DONE")
        print(i+1)
    #data_complete=np.array(pd.concat(chunks,axis=0,sort=False))
    '''
