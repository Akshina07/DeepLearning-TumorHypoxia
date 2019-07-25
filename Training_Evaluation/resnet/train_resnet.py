from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential,model_from_json
from tensorflow.python.keras.layers import Dense
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline 
import cv2
import os
from tensorflow.python.keras import optimizers
import math
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve, auc,accuracy_score,f1_score

# data initialization and processing 
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input) #data generator for processing the incoming images 
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = data_generator.flow_from_directory('/ysm-gpfs/pi/gerstein/aj557/data_deeppath/data_slides_resnet/train_slides5x',target_size=(224, 224),batch_size=200,class_mode='categorical')
#print("Done1")
validation_generator = data_generator.flow_from_directory('/ysm-gpfs/pi/gerstein/aj557/data_deeppath/data_slides_resnet/valid_slides5x',target_size=(224, 224),batch_size=200,class_mode='categorical')

# DATA SPECIFICATIONS AND TOP LAYER PARAMETERS
NUM_TRAINING_TILES=633367
NUM_VALID_TILES=150712
NUM_TEST_TILES=147678
NUM_CLASSES = 2
CHANNELS = 3
IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']
NUM_EPOCHS = 4 # For max of 4 epoch run on GPU else run on pi_gerstein for more number of epochs.
EARLY_STOP_PATIENCE = 3
BATCH_SIZE_TRAINING = 200
BATCH_SIZE_VALIDATION = 200
STEPS_PER_EPOCH_TRAINING = math.ceil(NUM_TRAINING_TILES/BATCH_SIZE_TRAINING)
STEPS_PER_EPOCH_VALIDATION = math.ceil(NUM_VALID_TILES/BATCH_SIZE_VALIDATION)

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1
image_size = IMAGE_RESIZE
if __name__=='__main__':
    model = Sequential()
    # 1st layer is the resnet base model (transfer learning)
    model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights ='imagenet'))
    # 2nd layer as Dense for 2-class classification
    model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))
    model.layers[0].trainable = False # Not training the resnet on the new data set. Using the pre-trained weigths 
    sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)  #compile the model

    #uncomment to add checkpoints and stop tranining when for 3 consecutive epoch the losses don't decrease any further .
    #cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
    #cb_checkpointer = ModelCheckpoint(filepath = '../working/best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')


    fit_history = model.fit_generator(train_generator,steps_per_epoch=STEPS_PER_EPOCH_TRAINING,epochs = NUM_EPOCHS,validation_data=validation_generator,validation_steps=STEPS_PER_EPOCH_VALIDATION) #using a generator to run the model.

    model_json = model.to_json()
    with open("Model_resnet5x.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    #model.save_model("Model_supervised_5.h5")
    model.save_weights("Model_resnet5x_weights.h5")
    print("Saved model to disk")
    #model.load_weights("/ysm-gpfs/pi/gerstein/aj557/data_deeppath/data_slides_resnet/best.hdf5")

    # PLOTS THE MODEL STATISTICS ( TRAINING VS VALIDATION LOSS AND ACCURACY )
    plt.figure(1, figsize = (15,8)) 
    plt.subplot(221)  
    plt.plot(fit_history.history['acc'])  
    plt.plot(fit_history.history['val_acc'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'valid']) 
    
    plt.subplot(222)  
    plt.plot(fit_history.history['loss'])  
    plt.plot(fit_history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'valid']) 

    plt.show()
    plt.savefig("Resnet_stats_tiles1_18July19.jpg")

    with open('Model_resnet5x.json') as f:
       model=model_from_json(f.read())   #load a pre-existing model architechure
    model.compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)
    model.load_weights('Model_resnet_weights.h5')  # load weights of the model

    test_generator = data_generator.flow_from_directory( directory = '/ysm-gpfs/pi/gerstein/aj557/data_deeppath/data_slides_resnet/test_slides5x/',target_size = (image_size, image_size),batch_size = BATCH_SIZE_TESTING,class_mode = 'categorical')

    pred = model.predict_generator(test_generator, steps = len(test_generator)) #returns probalities for classs corresponding to each test tile
    predicted_class_indices = np.argmax(pred, axis = 1) #converts probabilties into the labels 
    #results_df contains tile id's and label predicted for each tile 
    results_df = pd.DataFrame({'id': pd.Series(test_generator.filenames), 'label': pd.Series(predicted_class_indices)})
    results_df['id'] = results_df.id.str.extract('(\d+)')
    results_df['id'] = pd.to_numeric(results_df['id'], errors = 'coerce')
    results_df.sort_values(by='id', inplace = True)
 
    predict=[]
    for i in range(len(pred)):
        if(pred[i][0]>=0.5):
            predict.append([0,1])
        else:
            predict.append([1,0])
    predict=np.array(predict)  
    y_test=np.array(results_df['id'])
    y_label=np.array(results_df['label'])
    y_test_b=[]
    for i in range(len(y_test)):
        if(y_test[i]==1):
            y_test_b.append([0,1])
        else:
            y_test_b.append([1,0])
    y_test_b=np.array(y_test_b)  
    print(f1_score(y_test,y_label,average=None))
    print(f1_score(y_test,y_label,average='micro'))
    print(f1_score(y_test,y_label,average='macro'))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test_b[:,i],pred[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(roc_auc[i])
    
    print(results_df.shape)
     # CALCULATING THE PREDICTED LABELS PER SLIDE RATHER THAN PER TILE 
     # AGGREGATION USING THE AVERAGE PROBABILTY FOR A PARTICULAR LABEL PER TILE 
    '''
     y_train_labels=results_df['id']
     y_unique=[]
     y_unique_label=[]
     i=0
     for name in y_train_labels:
         file=name[6:18]
     y_train[i]=file
         #print(file)
          if(i==0):
              y_unique.append(file)
              #y_unique_label.append(y_train[i])
          else:
              index=np.where(np.array(y_unique)==file)[0]
              if(index.shape[0]==0):
                  y_unique.append(file)
            #y_unique_label.append(y_train[i])
    
           i+=1 
    for slide in y_unique:
    index=np.where(np.array(y_train)==slide)[0] 
    ''' 





    

