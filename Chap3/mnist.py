from autoencoder import Autoencoder
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BATCH_SIZE = 128
EPOCHS = 50
MODEL_PATH = "mnist_model"

def plot_latent_position(latent_presentation, y_test):
    fig = plt.figure(figsize=(15, 8))
    plt.title("Latent space representation")
    
    ax = fig.add_subplot(projection="3d")
    scatter = ax.scatter(latent_presentation[:,0], latent_presentation[:,1], latent_presentation[:, 2],
                cmap="rainbow", c=y_test,
                alpha=0.5, s=2)
    plt.colorbar(scatter)
    plt.show()

def plot_reconstructed_image(test_image, reconstructed_image):
    fig = plt.figure(figsize=(15, 8))
    plt.title("Reconstructed images")
    num = len(test_image)
    for i, (image, re_image) in enumerate(zip(test_image, reconstructed_image)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num, i+1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        
        re_image = re_image.squeeze()
        ax = fig.add_subplot(2, num, num+i+1)
        ax.axis("off")
        ax.imshow(re_image, cmap="gray_r")  
    plt.show()

def test_model():
    autoencoder = Autoencoder.load(MODEL_PATH)
    (_,_), (x_test, y_test) = mnist.load_data()
    
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))
    
    selected_idx = np.random.choice(x_test.shape[0], 10)
    test_image = x_test[selected_idx]
    
    
    latent_presentation, _ = autoencoder.predict(x_test)
    plot_latent_position(latent_presentation, y_test)
    
    _, reconstructed_image = autoencoder.predict(test_image)
    plot_reconstructed_image(test_image, reconstructed_image)
    

if __name__ == "__main__":
    '''
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    autoencoder = Autoencoder(
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