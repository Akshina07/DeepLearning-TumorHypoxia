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
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras import regularizers
import math
import json
from keras.models import load_model,model_from_json
from keras import regularizers
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve, auc,accuracy_score,f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#DATA PATHS 
path_data_train="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_TRAIN.csv"
test_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_TEST.csv"
path_labels_train="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABELS_TRAIN.csv"
test_label_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABELS_TEST.csv"
valid_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_VALID_B.csv"
train_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_ONLYTRAIN_B.csv"
valid_label_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABEL_VALID_B.csv"
train_label_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABEL_ONLYTRAIN_B.csv"
#DATA AND MODEL SPECIFICATIONS
n_train_samples=700000
n_valid_samples=150000
n_test_samples=369895
batch_size_train=500
batch_size_valid=3000
batch_size_test=3000
n_batch_train=math.ceil(n_train_samples/batch_size_train)
n_batch_valid=math.ceil(n_valid_samples/batch_size_valid)
n_batch_test=math.ceil(n_test_samples/batch_size_test)

#IN CASE OF MULTICLASS LABELLING
def conv_one_hot(data):  #assuming data is a numpy array for with labels (binary) 
    y_train_=[]
    for i in range(data.shape[0]):
        if(data[i]==0):
            y_train_.append([1,0])
        else:
            y_train_.append([0,1])
    return y_train_

#CUSTOMIZED GENERATORS FOR READING DATA IN BATCHES
def generator_train():
    #y=pd.read_csv(train_label_set,sep=",",chunksize=500)
    x=pd.read_csv(train_set,sep=",",chunksize=500)
    nb_calls=0
    while True:
        if(nb_calls==n_batch_train):
            #y=pd.read_csv(train_label_set,sep=",",chunksize=500)
            x=pd.read_csv(train_set,sep=",",chunksize=500)
            nb_calls=0
            # create numpy arrays of input data
            # and labels, from each line in the file
        x_=np.array(next(iter(x)))[:,1:2049]
        #y_=np.array(next(iter(y)))[:,2:3]
        nb_calls+=1
            #img = load_images(x)
        yield x_

def generator_valid():
    while True:
        #y=pd.read_csv(valid_label_set,sep=",",chunksize=3000)
        #print("done 1")
        x=pd.read_csv(valid_set,sep=",",chunksize=3000)
        for i in range(n_batch_valid):
            # create numpy arrays of input data
            # and labels, from each line in the file
            x_=np.array(next(iter(x)))[:,1:2049]
            #y_=np.array(next(iter(y)))[:,2:3]
            #img = load_images(x)
            yield x_

def generator_test():
    while True:
        x=pd.read_csv(test_set,sep=",",chunksize=3000)
        for i in range(n_batch_test):
            # create numpy arrays of input data
            # and labels, from each line in the file
            x_=np.array(next(iter(x)))[:,1:2049]
            #y_=to_categorical(np.array(next(iter(y)))[:,2:3])
            #img = load_images(x)
            yield x_

if __name__=="__main__":
   #open a model. load the model architechure
    with open('Model_supervised_12.json') as f:
        model=model_from_json(f.read())
   #load weights into the model 
    model.load_weights('Model_weights_supervised_12.h5')
    print("Saved model to disk")
   
    # Calculate predictions on train data 
    predictions_train=model.predict_generator(generator_train(),steps=n_batch_train)
   
     # Load the predicitons for test data if already ca;culated. Or use the predict_generator to calculate the predicitons.
    predictions=np.array(pd.read_csv("/ysm-gpfs/pi/gerstein/aj557/data_deeppath/Predictions_Model12.csv",sep=',',header=None))[:,1]
   
    # Used for converting predictions to labels. Required for calculating F1 scores.
    predict=[]
    for i in range(len(predictions)):
        if(predictions[i]>=0.5):
            predict.append(1)
        else:
            predict.append(0)
    predict=np.array(predict)
    
    #y_train=pd.to_numeric(np.array(pd.read_csv(train_label_set,sep=',',header=0))[:,2])
    #Load the true labels
    y=pd.to_numeric(np.array(pd.read_csv(test_label_set,sep=',',nrows=150000,header=0))[:,2])
  
    #print(y.shape)
    #print(predictions.shape)
    print("F1 Scores for Test data")
    print(f1_score(y,predict,average=None))
    print(f1_score(y,predict,average='micro'))
    print(f1_score(y,predict,average='macro'))
   
    print("ROC CURVE AND SCORE FOR TRAIN DATA")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(1): 
        fpr[i], tpr[i], _ = roc_curve(y,predictions)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(1):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig("Model_12_test_supervised_roc.jpeg")
   
    # EVALUATION ON TRAINING SET
    predict=[]
    for i in range(len(predictions_train)):
        if(predictions_train[i]>=0.5):
            predict.append(1)
        else:
            predict.append(0)
    predict=np.array(predict)
    
    #load the training set.
    y_train=pd.to_numeric(np.array(pd.read_csv(train_label_set,sep=',',header=0))[:,2])
    #print(y.shape)
    #print(predictions.shape)
    print("F1 Scores for Train data")
    print(f1_score(y_train,predict,average=None))
    print(f1_score(y_train,predict,average='micro'))
    print(f1_score(y_train,predict,average='macro'))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(y_train,predictions_train)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(1):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig("Model_12_train_supervised_roc.jpeg")
        




    
