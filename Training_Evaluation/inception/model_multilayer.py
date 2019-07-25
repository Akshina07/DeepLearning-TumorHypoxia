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
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
# PATH TO TRAINING AND TESTING DATA SETS
path_data_train="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_TRAIN.csv"
test_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_TEST.csv"
path_labels_train="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABELS_TRAIN.csv"
test_label_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABELS_TEST.csv"
valid_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_VALID.csv"
train_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_ONLYTRAIN.csv"
valid_label_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABEL_VALID.csv"
train_label_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABEL_ONLYTRAIN.csv"

#NUMBER OF TILES IN EACH SET AND BATCH SIZE SPECIFICATIONS
n_train_samples=1555713
n_valid_samples=395338
n_test_samples=369895
batch_size_train=14000
batch_size_valid=3000
batch_size_test=3000
n_batch_train=math.ceil(n_train_samples/batch_size_train)
n_batch_valid=math.ceil(n_valid_samples/batch_size_valid)
n_batch_test=math.ceil(n_test_samples/batch_size_test)

def conv_one_hot(data):  #assuming data is a numpy array for with labels (binary) 
    y_train_=[]
    for i in range(data.shape[0]):
        if(data[i]==0):
            y_train_.append([1,0])
        else:
            y_train_.append([0,1])
    return y_train_

# CUSTOMIZED GENERATOR FUNCTION. 
# The generator functions yields infinitely therefore it is important to 
# specify breaks so that for one epoch all the batches are yielded and for the next 
# epoch the generator should yield from initial batch
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
    label_test=np.array(pd.read_csv(test_label_set,sep=",",chunksize=10000)) #array containing the labels.
    #print("Hello1") 
    model = Sequential()
    n_cols=2048 # number of features extracted from transfer learning
    #model.add(Batch
    model.add(Dense(2048, input_shape=(n_cols,),kernel_initializer="uniform"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) # to avoid overfitting
    model.add(Dense(1024, kernel_initializer="uniform"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, kernel_initializer="uniform"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, kernel_initializer="uniform"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
    #model_2.fit(x_train_, y_train, epochs=10, validation_split=0.2, callbacks=[early_stopping_monitor])
    #early_stopping_monitor = EarlyStopping(patience=3)
    #train model
    print("Model complied")
    model.fit_generator(generator_train(),steps_per_epoch=n_batch_train,epochs=100,validation_data=generator_valid(),validation_steps=n_batch_valid,shuffle=False,workers=False)
    
    # serialize model to JSON. save the model architecture
    model_json = model.to_json()
    with open("Model_supervised_1-2.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5. save the model weigths.
    #model.save_model("Model_supervised_1-2.h5")
    model.save_weights("Model_weights_supervised_1-2.h5")
    print("Saved model to disk")

    predictions=model.predict_generator(generator_test(),steps=n_batch_test)
    #print("Overall accuracy={}".format(1-sum(abs(label_test-predictions))*0.01))
    predictions_model="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/Predictions_Model1-2.csv"  # save the predicted probabilities on the test data into a csv 




    
