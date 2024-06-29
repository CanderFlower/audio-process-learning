from numpy import prod
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Flatten, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

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
       
    def _build_encoder(self):
        input = Input(shape=self.input_shape, name="encoder_input")
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
        output = Conv2D(1, self.conv_kernels[self.conv_layer_num-1], self.conv_strides[self.conv_layer_num-1],
                        padding="same", activation="sigmoid", name="decoder_output")(x)
        return Model(input, output, name="decoder")
    
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        
if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()