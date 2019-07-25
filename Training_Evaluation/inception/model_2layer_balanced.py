import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,auc,roc_curve,accuracy_score
from argparse import ArgumentParser
import csv 
from collections import deque
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,LeakyReLU
from keras.callbacks import EarlyStopping,LearningRateScheduler
from keras.utils import to_categorical
from keras import regularizers
import math
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

# PATH TO THE DATA SETS 
path_data_train="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_TRAIN.csv"
test_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_TEST.csv"
path_labels_train="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABELS_TRAIN.csv"
test_label_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABELS_TEST.csv"
valid_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_VALID_BALANCED.csv"
train_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_DATA_ONLYTRAIN_BALANCED.csv"
valid_label_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABEL_VALID_BALANCED.csv"
train_label_set="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/BRCA_LABEL_ONLYTRAIN_BALANCED.csv"

# DATA SPECIFICATION AND CORRESPONDING BATCH SIZE

n_train_samples=700000
n_valid_samples=150000
n_test_samples=150000
batch_size_train=500
batch_size_valid=3000
batch_size_test=15000
n_batch_train=math.ceil(n_train_samples/batch_size_train)
n_batch_valid=math.ceil(n_valid_samples/batch_size_valid)
n_batch_test=math.ceil(n_test_samples/batch_size_test)

# the Y ie the labels are one hot encoded when using softmax and categorical cross entropy
def conv_one_hot(data):  #assuming data is a numpy array for with labels (binary) 
    y_train_=[]
    for i in range(data.shape[0]):
        if(data[i]==0):
            y_train_.append([1,0])
        else:
            y_train_.append([0,1])
    return y_train_

# Customized data generators. The data generators yield tuples in an infinite loop. Make surethe entire data set is yielded for one epoch and the generator yields from the first tuple for each new epoch 


def generator_train():
    y=pd.read_csv(train_label_set,sep=",",chunksize=500,header=None)
    x=pd.read_csv(train_set,sep=",",chunksize=500,header=None)
    nb_calls=0
    while True:
        if(nb_calls==n_batch_train):
            y=pd.read_csv(train_label_set,sep=",",chunksize=500,header=None)
            x=pd.read_csv(train_set,sep=",",chunksize=500,header=None)
            nb_calls=0
            # create numpy arrays of input data
            # and labels, from each line in the file
        x_=np.array(next(iter(x)))[:,1:2049]
        y_=np.array(next(iter(y)))[:,2:3]
        nb_calls+=1
            #img = load_images(x)
        yield x_,y_

def generator_valid():
    y=pd.read_csv(valid_label_set,sep=",",chunksize=3000,header=None)
    x=pd.read_csv(valid_set,sep=",",chunksize=3000,header=None)
    nb_calls=0
    while True:
        if(nb_calls==n_batch_valid):
            y=pd.read_csv(valid_label_set,sep=",",chunksize=3000,header=None)
            x=pd.read_csv(valid_set,sep=",",chunksize=3000,header=None)
            nb_calls=0
        
            # create numpy arrays of input data
            # and labels, from each line in the file
        x_=np.array(next(iter(x)))[:,1:2049]
        y_=np.array(next(iter(y)))[:,2:3]
        nb_calls+=1
            #img = load_images(x)
        yield x_,y_

def generator_test():
    x=pd.read_csv(test_set,sep=",",chunksize=15000,header=None)
    nb_calls=0
    while True:
        if(nb_calls==n_batch_test):
            x=pd.read_csv(test_set,sep=",",chunksize=15000,header=None)
            nb_calls=0
            # create numpy arrays of input data
            # and labels, from each line in the file
        x_=np.array(next(iter(x)))[:,1:2049]
            #y_=to_categorical(np.array(next(iter(y)))[:,2:3])
            #img = load_images(x)
        nb_calls+=1
        yield x_

# This is used in case we want to change the learning rate as we proceed with training. Will have to add epoch checkpoints and correspnding callbacks. Changing learning rates can also be achieved with varying mometum of the sgd optimizer function.

def learning_rate_scheduler(epoch):
    if epoch <10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))
    
if __name__=="__main__":
    label_test=np.array(pd.read_csv(test_label_set,sep=",")) #Test labels 
    model = Sequential()
    n_cols=2048
    model.add(Dense(2048, input_shape=(n_cols,),kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    sgd = SGD(lr=0.001,momentum=0.8)
    model.compile(optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=['accuracy']) # in case of sparse categorical cross entropy loss function there is no need to one hot enode the labels i.e y
    #early_stopping_monitor = EarlyStopping(patience=3)
    print("Model complied")
    #callback = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)
    model.fit_generator(generator_train(),steps_per_epoch=n_batch_train,epochs=75,validation_data=generator_valid(),validation_steps=n_batch_valid,shuffle=False,workers=False)
    
    # serialize model to JSON
    model_json = model.to_json()i
    with open("Model_2layer_balanced.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    #model.save_model("Model_supervised_5.h5")
    model.save_weights("Model_2layer_balanced.h5")
    print("Saved model to disk")

    predictions=model.predict_generator(generator_test(),steps=n_batch_test)
    #print("Overall accuracy={}".format(1-sum(abs(label_test-predictions))*0.01))
    predictions_model="/ysm-gpfs/pi/gerstein/aj557/data_deeppath/Predictions_2layer_balanced.csv"
    with open(predictions_model,"a") as f:
        pd.DataFrame(predictions).to_csv(f,header=False)
        f.close()


    # VISUALIZING DATA AND RESULTS. PLOT AUC CURVE AND PRINT MODEL OVERALL ACCURACY ON TEST DATA SET
    '''
    predict=[]
    for i in range(len(predictions)):
        if(predictions[i]>=0.5):
            predict.append(1)
        else:
            predict.append(0)
    predict=np.array(predict)
    y=pd.to_numeric(np.array(pd.read_csv(test_label_set,sep=',',header=None))[:,2])
    predictions=predictions[:,0]
    print(y.shape)
    print(predictions.shape)

    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(y, predictions)
        roc_auc[i] = auc(fpr[i], tpr[i])

    #plot the auc curve 
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
        plt.savefig("Model_2layer_balanced.jpeg")

    print(accuracy_score(y,predict))   #predict over accuracy. 
    ''' 




    
