from VAE import VAE
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mnist import plot_latent_position_3d, plot_reconstructed_image

BATCH_SIZE = 512
EPOCHS = 50
MODEL_PATH = "mnist_model_vae"

def plot_latent_position_2d(latent_presentation, y_test):
    plt.figure(figsize=(15, 8))
    plt.title("Latent space representation")
    
    scatter = plt.scatter(latent_presentation[:, 0], latent_presentation[:, 1],
                          cmap="rainbow", c=y_test,
                          alpha=0.5, s=2)
    plt.colorbar(scatter)
    plt.show()

def test_model():
    autoencoder = VAE.load(MODEL_PATH)
    (_,_), (x_test, y_test) = mnist.load_data()
    
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))
    
    selected_idx = np.random.choice(x_test.shape[0], 10)
    test_image = x_test[selected_idx]
    
    
    latent_presentation, _ = autoencoder.predict(x_test)
    plot_latent_position_3d(latent_presentation, y_test)
    
    _, reconstructed_image = autoencoder.predict(test_image)
    plot_reconstructed_image(test_image, reconstructed_image)
    

if __name__ == "__main__":
    '''
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    autoencoder = VAE(
        input_shape=x_train.shape[1:],
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=3
    )
    autoencoder.summary()
    autoencoder.compile()
    autoencoder.train(x_train, BATCH_SIZE, EPOCHS)
    autoencoder.save(MODEL_PATH)
    autoencoder.plot_history()
    '''
    test_model()