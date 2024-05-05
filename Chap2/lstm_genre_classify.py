import tensorflow as tf
from genre_classify import load_data,save_history,plot_history
from cnn_genre_classify import split_datasets

DATA_PATH = r"D:\Code\Tests\Audio\Chap2\mfcc.json"
MODEL_PATH = r"D:\Code\Tests\Audio\Chap2\model_lstm.h5"
HISTORY_PATH = r"D:\Code\Tests\Audio\Chap2\history_lstm.json"

if __name__ == "__main__":
    
    #Prepare data
    print("Loading Data...")
    genres,inputs,targets = load_data(DATA_PATH)
    print("Data Loaded.")
    
    #Get the shape of data
    input_shape = (inputs.shape[1],inputs.shape[2])
    
    #Split data
    inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = split_datasets(
        inputs,targets,0.25,0.2
    )
    print("Data Splitted.")
    
    #Build Model
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.LSTM(64,input_shape=input_shape,return_sequences=True))
    model.add(tf.keras.layers.LSTM(64))
    
    model.add(tf.keras.layers.Dense(64,activation="relu"))
    model.add(tf.keras.layers.Dropout(0.4))
    
    model.add(tf.keras.layers.Dense(len(genres),activation="softmax"))
    
    print("Model built.")
    
    #Compile & Train Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00008)
    model.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    
    history=model.fit(inputs_train,targets_train,validation_data=(inputs_validation,targets_validation),batch_size=32,epochs=150)
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