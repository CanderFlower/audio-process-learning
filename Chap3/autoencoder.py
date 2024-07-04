import pickle
from numpy import prod
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Flatten, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import os

class Autoencoder:
    
    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
       self.input_shape = input_shape 
       self.conv_filters = conv_filters
       self.conv_kernels = conv_kernels
       self.conv_strides = conv_strides
       self.latent_space_dim = latent_space_dim
       self.conv_layer_num = len(conv_filters)
       
       self.encoder = self._build_encoder()
       self.decoder = self._build_decoder()
       self.model = self._build_autoencoder()
       
       self.history = None
       
    def _build_encoder(self):
        input = Input(shape=self.input_shape, name="encoder_input")
        self.input = input
        x = input
        for i,(filter,kernel,stride) in enumerate(zip(self.conv_filters, self.conv_kernels, self.conv_strides)):
            x = Conv2D(filter, kernel, stride, padding="same", activation="relu", name=f"encoder_conv_{i}")(x)
            x = BatchNormalization(name=f"encoder_bn_{i}")(x)
        self.shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        output = Dense(self.latent_space_dim, name="encoder_output")(x)
        return Model(input, output, name="encoder")
    
    def _build_decoder(self):
        input = Input(shape=self.latent_space_dim, name="decoder_input")
        x = input
        x = Dense(prod(self.shape_before_bottleneck), name="decoder_dense")(x)
        x = Reshape(self.shape_before_bottleneck, name="decoder_reshape")(x)
        for i, (filter, kernel, stride) in enumerate(zip(reversed(self.conv_filters), reversed(self.conv_kernels), reversed(self.conv_strides))):
            x = Conv2DTranspose(filter, kernel, stride, padding="same", activation="relu", name=f"decoder_conv_trans_{i}")(x)
            x = BatchNormalization(name=f"decoder_bn_{i}")(x)
        output = Conv2D(1, self.conv_kernels[-1], self.conv_strides[-1],
                        padding="same", activation="sigmoid", name="decoder_output")(x)
        return Model(input, output, name="decoder")
    
    def _build_autoencoder(self):
        input = self.input
        output = self.decoder(self.encoder(input))
        return Model(input, output, name="autoencoder")
    
    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=loss)
    
    def train(self, x_train, batch_size, epochs):
        self.history = self.model.fit(x_train, x_train, batch_size, epochs)
        
    def predict(self, input):
        latent_presentation = self.encoder.predict(input)
        reconstructed_image = self.decoder.predict(latent_presentation)
        return latent_presentation, reconstructed_image
    
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()
    
    def plot_history(self):
        history = self.history
        plt.plot(history.history["loss"], label="train loss")
        if 'val_loss' in history.history:
            plt.plot(history.history["val_loss"], label="test loss")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.title("Loss Evaluation")
        plt.show()
        
    def save(self, path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if not os.path.exists(path):
            os.makedirs(path)
            
        para_path = os.path.join(path, "parameters.pkl")
        parameters = [self.input_shape,
                      self.conv_filters,
                      self.conv_kernels,
                      self.conv_strides,
                      self.latent_space_dim]
        with open(para_path, "wb") as f:
            pickle.dump(parameters, f)
            
        model_path = os.path.join(path, "weights.h5")
        self.model.save_weights(model_path)
        
    @classmethod
    def load(cls, path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        para_path = os.path.join(path, "parameters.pkl")
        with open(para_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        
        model_path = os.path.join(path, "weights.h5")
        autoencoder.model.load_weights(model_path)
        return autoencoder
        
if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()