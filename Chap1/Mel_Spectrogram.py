import librosa
import numpy as np
from matplotlib import pyplot as plt

def show_banks():
    filter_banks = librosa.filters.mel(sr=sample_rate,n_fft=2048,n_mels=10)
    plt.figure(figsize=(25,10))
    librosa.display.specshow(filter_banks,sr=sample_rate,x_axis="linear")
    plt.colorbar(format="%+2.f")
    plt.show()

if __name__ == "__main__":
    wave_path = r"D:\Code\Tests\Audio\audio_data\music_piano.wav"
    waveform, sample_rate = librosa.load(wave_path,sr=None)
    
    #show_banks()
    
    mel_spec = librosa.feature.melspectrogram(y=waveform,sr=sample_rate,n_fft=2048,hop_length=512,n_mels=40)
    log_mel_spec = librosa.power_to_db(mel_spec)
    plt.figure(figsize=(25,10))
    librosa.display.specshow(log_mel_spec,sr=sample_rate,x_axis="time",y_axis="mel")
    plt.colorbar(format="%+2.f")
    plt.show()