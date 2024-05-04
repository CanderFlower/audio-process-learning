import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from genre_classify import load_data,save_history,plot_history

DATA_PATH = r"D:\Code\Tests\Audio\Chap2\mfcc.json"
MODEL_PATH = r"D:\Code\Tests\Audio\Chap2\model_cnn.h5"
HISTORY_PATH = r"D:\Code\Tests\Audio\Chap2\history_cnn.json"

def split_datasets(inputs,targets,test_size,validation_size):
    
    inputs_train,inputs_test,targets_train,targets_test = train_test_split(
        inputs,targets,
        test_size=test_size
    )
    inputs_train,inputs_validation,targets_train,targets_validation = train_test_split(
        inputs_train,targets_train,
        test_size=validation_size
    )
    return inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test

if __name__ == "__main__":
    
    #Prepare data
    print("Loading Data...")
    genres,inputs,targets = load_data(DATA_PATH)
    print("Data Loaded.")
    
    #Reshape data for CNN
    inputs = inputs[...,np.newaxis]
    input_shape = (inputs.shape[1],inputs.shape[2],inputs.shape[3])
    
    #Split data
    inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = split_datasets(
        inputs,targets,0.25,0.2
    )
    print("Data Splitted.")
    
    #Build Model
    model = tf.keras.Sequential()
    
    for _ in range(2):
        model.add(tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=input_shape))
        model.add(tf.keras.layers.MaxPool2D((3,3), strides=(2,2), padding="same"))
        model.add(tf.keras.layers.BatchNormalization())
    for _ in range(1):
        model.add(tf.keras.layers.Conv2D(32,(2,2),activation="relu",input_shape=input_shape))
        model.add(tf.keras.layers.MaxPool2D((2,2), strides=(2,2), padding="same"))
        model.add(tf.keras.layers.BatchNormalization())
        
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64,activation="relu",kernel_regularizer = tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(0.4))
    
    model.add(tf.keras.layers.Dense(len(genres),activation="softmax"))
    print("Model built.")
    
    #Compile & Train Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00008)
    model.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    
    history=model.fit(inputs_train,targets_train,validation_data=(inputs_validation,targets_validation),batch_size=32,epochs=100)
    print("Model trained.")
    
    #Evaluate Model
    print("Evaluating...")
    test_error,test_accuracy = model.evaluate(inputs_test,targets_test,verbose=1)
    print("Evaluation:")
    print("Error: {}".format(test_error))
    print("Accuracy: {}".format(test_accuracy))
    
    #Save & Show History
    save_history(history,HISTORY_PATH)
    
    plot_history(history)