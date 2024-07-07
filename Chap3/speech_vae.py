from VAE import VAE
import numpy as np
import os

BATCH_SIZE = 16
EPOCHS = 150
MODEL_PATH = "speech_model_vae"
X_PATH = r"D:\Code\Tests\Audio\Chap3\recordings_feature"

def load_fsdd(path):
    x_train = []
    for _, _, files in os.walk(path):
        for file in files:
            filepath = os.path.join(X_PATH, file)
            x_train.append(np.load(filepath))
    
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]
    return x_train

if __name__ == "__main__":
    
    x_train = load_fsdd(X_PATH)
    autoencoder = VAE(
        input_shape=x_train.shape[1:],
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, 1),
        latent_space_dim=512
    )
    autoencoder.summary()
    autoencoder.compile()
    autoencoder.train(x_train, BATCH_SIZE, EPOCHS)
    autoencoder.save(MODEL_PATH)
    autoencoder.plot_history()
    