import numpy as np
import tensorflow as tf
from random import random
from sklearn.model_selection import train_test_split

def get_dataset(data_size,test_rate):
    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(data_size)])
    targets = np.array([[input[0]+input[1]] for input in inputs])
    x_train,x_test,y_train,y_test=train_test_split(inputs,targets,test_size=test_rate)
    return x_train,x_test,y_train,y_test

if __name__ == "__main__":
    
    x_train,x_test,y_train,y_test = get_dataset(100000, 0.1)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5,activation="relu",input_dim=2),
        tf.keras.layers.Dense(1,activation="relu")
    ])
    
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)
    model.compile(optimizer=optimizer,loss="mse")
    
    model.fit(x_train,y_train,epochs=100)
    
    print("[Evaluate!]")
    model.evaluate(x_test,y_test,verbose=1)
    
    test_data = np.array([[0.1,0.2],[0.3,0.7]])
    predictions = model.predict(test_data)
    for data,prediction in zip(test_data,predictions):
        print("{} + {} = {}".format(data[0],data[1],prediction))