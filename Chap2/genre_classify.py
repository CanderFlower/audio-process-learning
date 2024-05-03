import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_PATH = r"D:\Code\Tests\Audio\Chap2\mfcc.json"
MODEL_PATH = r"D:\Code\Tests\Audio\Chap2\model.h5"
HISTORY_PATH = r"D:\Code\Tests\Audio\Chap2\history.json"

def load_data(data_path):
    with open(data_path,mode="r") as f:
        data = json.load(f)
    genres = np.array(data["mapping"])
    inputs = np.array(data["mfcc"])
    targets = np.array(data["label"])
    return genres,inputs,targets

def save_history(history, history_path):
    with open(history_path, "w") as f:
        json.dump(history.history, f)


def plot_history(history):
    fig,axs = plt.subplots(2)
    
    axs[0].plot(history.history["accuracy"],label="train accuracy")
    axs[0].plot(history.history["val_accuracy"],label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy")
    
    axs[1].plot(history.history["loss"],label="train loss")
    axs[1].plot(history.history["val_loss"],label="test loss")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss eval")
    
    plt.show()

if __name__ == "__main__":
    
    print("Loading Data...")
    genres,inputs,targets = load_data(DATA_PATH)
    print("Data Loaded.")
    
    inputs_train,inputs_test,targets_train,targets_test = train_test_split(
        inputs,targets,
        test_size=0.3
    )
    print("Data Splitted.")
    
    kernel_regularizer = tf.keras.regularizers.l2(0.001)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(inputs.shape[1],inputs.shape[2])),
        
        tf.keras.layers.Dense(512,activation="relu",kernel_regularizer = kernel_regularizer),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(256,activation="relu",kernel_regularizer = kernel_regularizer),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64,activation="relu",kernel_regularizer = kernel_regularizer),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(len(genres),activation="softmax")
    ])  
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    history = model.fit(inputs_train,targets_train,validation_data=(inputs_test,targets_test),batch_size=32,epochs=150)
    
    model.save(MODEL_PATH)
    
    save_history(history,HISTORY_PATH)
    
    plot_history(history)