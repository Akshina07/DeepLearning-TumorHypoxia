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
from keras.layers import Dense,Activation,Dropout
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras import regularizers
import math
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.normalization import BatchNormalization
path_data_train="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_TRAIN.csv"
test_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_TEST.csv"
path_labels_train="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABELS_TRAIN.csv"
test_label_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABELS_TEST.csv"
valid_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_VALID.csv"
train_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_ONLYTRAIN.csv"
valid_label_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABEL_VALID.csv"
train_label_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABEL_ONLYTRAIN.csv"
n_train_samples=1555713
n_valid_samples=395338
n_test_samples=369895
batch_size_train=14000
batch_size_valid=3000
batch_size_test=3000
n_batch_train=math.ceil(n_train_samples/batch_size_train)
n_batch_valid=math.ceil(n_valid_samples/batch_size_valid)
n_batch_test=math.ceil(n_test_samples/batch_size_test)
def create_model(optimizer='adam',loss='mean_squared_error',activation1='relu',activation2='sigmoid'):
    # create model
    n_cols_2=2048
    model=Sequential()
    model.add(Dense(2048, input_shape=(n_cols_2,),kernel_initializer="uniform")) # fully connected layer
    model.add(BatchNormalization())
    model.add(Activation(activation1))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation=activation2)) #classification layer
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model
def conv_one_hot(data):  #assuming data is a numpy array for with labels (binary) 
    y_train_=[]
    for i in range(data.shape[0]):
        if(data[i]==0):
            y_train_.append([1,0])
        else:
            y_train_.append([0,1])
    return y_train_

def generator_train():
    y=pd.read_csv(train_label_set,sep=",",chunksize=14000)
    x=pd.read_csv(train_set,sep=",",chunksize=14000)
    nb_calls=0
    while True:
        if(nb_calls==n_batch_train):
            y=pd.read_csv(train_label_set,sep=",",chunksize=14000)
            x=pd.read_csv(train_set,sep=",",chunksize=14000)
            nb_calls=0
            # create numpy arrays of input data
            # and labels, from each line in the file
        x_=np.array(next(iter(x)))[:,1:2049]
        y_=np.array(next(iter(y)))[:,2:3]
        nb_calls+=1
            #img = load_images(x)
        yield x_,y_

def generator_valid():
    y=pd.read_csv(valid_label_set,sep=",",chunksize=3000)
    x=pd.read_csv(valid_set,sep=",",chunksize=3000)
    nb_calls=0
    while True:
        if(nb_calls==n_batch_valid):
            y=pd.read_csv(valid_label_set,sep=",",chunksize=3000)
            x=pd.read_csv(valid_set,sep=",",chunksize=3000)
            nb_calls=0
        
            # create numpy arrays of input data
            # and labels, from each line in the file
        x_=np.array(next(iter(x)))[:,1:2049]
        y_=np.array(next(iter(y)))[:,2:3]
        nb_calls+=1
            #img = load_images(x)
        yield x_,y_

def generator_test():
    x=pd.read_csv(test_set,sep=",",chunksize=3000)
    nb_calls=0
    while True:
        if(nb_calls==n_batch_test):
            x=pd.read_csv(test_set,sep=",",chunksize=3000)
            nb_calls=0
            # create numpy arrays of input data
            # and labels, from each line in the file
        x_=np.array(next(iter(x)))[:,1:2049]
            #y_=to_categorical(np.array(next(iter(y)))[:,2:3])
            #img = load_images(x)
        nb_calls+=1
        yield x_

if __name__=="__main__":
   

    #param_grid_1 = dict(epoch=epoch,batchsize=batchsize)
    x_train=np.array(pd.read_csv(train_set,sep=",",nrows=20000))[:,1:2049]
    y_train=np.array(pd.read_csv(train_label_set,sep=",",nrows=20000))[:,2:3]
    model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=2000, verbose=0)
    # define the grid search parameters
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    loss=['binary_crossentropy','mean_squared_error','logcosh']
    epoch=[50,100,150,200]
    batchsize=[50,100,500,1000,1500,2000]
    activation1=['relu','tanh','sigmoid','elu']
    activation2=['relu','tanh','sigmoid','elu']
    param_grid = dict(optimizer=optimizer,loss=loss,activation1=activation1,activation2=activation2,epochs=epoch,batch_size=batchsize)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(x_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    

    #label=pd.read_csv(train_label_set,sep=",",chunksize=14000)
    #data=pd.read_csv(train_set,sep=",",chunksize=14000)
    #label_valid=pd.read_csv(valid_label_set,sep=",",chunksize=3000)
    #print("done 1")
    #data_valid=pd.read_csv(valid_set,sep=",",chunksize=3000)
    #print("done 2")
    #label_test=np.array(pd.read_csv(test_label_set,sep=",",chunksize=10000))
    #print("done 3")
    #data_test=pd.read_csv(test_set,sep=",",chunksize=30i
    
    #print("Hello1") 
    #model = Sequential()
    #n_cols=2048
    #model.add(Batch
    #model.add(Dense(2048, input_shape=(n_cols,),init='uniform'))
        




    
