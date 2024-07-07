import os
import pickle
from autoencoder import Autoencoder
from tensorflow.keras.layers import Dense, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

class VAE(Autoencoder):
    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        super().__init__(input_shape,
                       conv_filters,
                       conv_kernels,
                       conv_strides,
                       latent_space_dim)
        self.reconstruct_loss_weight = 1200000
    
    def _add_bottleneck(self, x):
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x)
        
        def sample_point(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(mu))
            return mu + K.exp(log_variance/2) * epsilon
        
        return Lambda(sample_point, name="encoder_output")([self.mu, self.log_variance])
    
    def vae_loss(self, y_test, y_predict):
        error = y_test - y_predict
        mse = K.mean(K.square(error), axis=[1, 2, 3])
        kl = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis=-1)
        return mse*self.reconstruct_loss_weight + kl
    
    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        loss = self.vae_loss
        self.model.compile(optimizer=optimizer, loss=loss)
        
    @classmethod
    def load(cls, path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        para_path = os.path.join(path, "parameters.pkl")
        with open(para_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        
        model_path = os.path.join(path, "weights.h5")
        autoencoder.model.load_weights(model_path)
        return autoencoder
    
        
if __name__ == "__main__":
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()