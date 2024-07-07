import pickle
from VAE import VAE
import numpy as np
import os
import librosa
import soundfile as sf
import shutil
import matplotlib.pyplot as plt
from speech_vae import load_fsdd

MODEL_PATH = "speech_model_vae"
AUDIO_PATH = r"D:\Code\Tests\Audio\Chap3\recordings"
X_PATH = r"D:\Code\Tests\Audio\Chap3\recordings_feature"
OUTPUT_PATH = r"D:\Code\Tests\Audio\Chap3\generated"
MIN_MAX_PATH = r"D:\Code\Tests\Audio\Chap3\min_max.pkl"
NUM_SAMPLES = 10
HOP_LENGTH = 256
FRAME_LENGTH = 512
SAMPLE_RATE = 22050

def choose_speech(path):
    files = os.listdir(AUDIO_PATH)
    chosen_files = np.random.choice(files, NUM_SAMPLES)
    return chosen_files

def denormalize(spectrogram, file, min_max):
    min = min_max[file]["min"]
    max = min_max[file]["max"]
    spectrogram = min + spectrogram * (max-min)
    return spectrogram

def plot_latent_position_2d(latent_presentation):
    plt.figure(figsize=(8, 8))
    plt.title("Latent space representation")
    plt.scatter(latent_presentation[:, 0], latent_presentation[:, 1],
                          alpha=0.5, s=2)
    plt.show()

def save_comparision(path, origin, generated):
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    librosa.display.specshow(origin, hop_length=HOP_LENGTH, x_axis="time", y_axis="log")
    plt.colorbar(format='%+2.0f dB')
    plt.title("Original Normalized Spectrogram")
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(generated, hop_length=HOP_LENGTH, x_axis="time", y_axis="log")
    plt.colorbar(format='%+2.0f dB')
    plt.title("Generated Normalized Spectrogram")
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def process(autoencoder, file, min_max):
    audio_path = os.path.join(AUDIO_PATH, file)
    signal, sr = librosa.load(audio_path)
    
    spectrogram_path = os.path.join(X_PATH, file+".npy")
    spectrogram = np.load(spectrogram_path)
    
    _, generated_log_spectrogram = autoencoder.predict(spectrogram[np.newaxis, ..., np.newaxis])
    generated_log_spectrogram = generated_log_spectrogram[0,:,:,0]
    
    comparision_path = os.path.join(OUTPUT_PATH, f"comparison_{os.path.splitext(file)[0]}.png")
    save_comparision(comparision_path, spectrogram, generated_log_spectrogram)
    
    generated_log_spectrogram = denormalize(generated_log_spectrogram, file, min_max)
    generated_spectrogram = librosa.db_to_amplitude(generated_log_spectrogram)
    
    #generated_signal = librosa.istft(generated_spectrogram, hop_length=HOP_LENGTH)
    generated_signal = librosa.griffinlim(generated_spectrogram, hop_length=HOP_LENGTH)

    output_original_path = os.path.join(OUTPUT_PATH, f"origin_{file}")
    shutil.copy(audio_path, output_original_path)
    
    output_generated_path = os.path.join(OUTPUT_PATH, f"generated_{file}")
    sf.write(output_generated_path, generated_signal, SAMPLE_RATE)

def show_latent_space(autoencoder):
    x_train = load_fsdd(X_PATH)
    latent_space,_ = autoencoder.predict(x_train)
    plot_latent_position_2d(latent_space)

def test_speech_vae():
    autoencoder = VAE.load(MODEL_PATH)
    
    #show_latent_space(autoencoder)
    
    filenames = choose_speech(AUDIO_PATH)
    
    with open(MIN_MAX_PATH, "rb") as f:
        min_max = pickle.load(f)
        
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)
    
    for file in filenames:
        process(autoencoder, file, min_max)

if __name__ == "__main__":
    test_speech_vae()