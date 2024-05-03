import librosa
import numpy as np
from matplotlib import pyplot as plt

from Zero_Crossing_Rate import Calc_ZCR


if __name__ == "__main__":
    wave_path = r"D:\Code\Tests\Audio\audio_data\jazz.wav"
    waveform, sample_rate = librosa.load(wave_path,sr=None)
    
    mfcc = librosa.feature.mfcc(y=waveform,n_mfcc=13,sr=sample_rate)
    
    plt.figure(figsize=(25,10))
    librosa.display.specshow(mfcc,sr=sample_rate,x_axis="time")
    plt.colorbar(format="%+2.f")
    
    frame_length = 1024
    hop_length = 512
    waveform_ZCR = Calc_ZCR(waveform,frame_length,hop_length)
    frames_num = np.arange(len(waveform_ZCR))
    frames_at = librosa.frames_to_time(frames_num,sr=sample_rate,hop_length=hop_length)
    plt.plot(frames_at,waveform_ZCR*40,"g")
    plt.show()
    
    mfcc_delta = librosa.feature.delta(mfcc,order=1)
    plt.figure(figsize=(25,10))
    librosa.display.specshow(mfcc_delta,sr=sample_rate,x_axis="time")
    plt.colorbar(format="%+2.f")
    plt.show()
    
    mfcc_delta2 = librosa.feature.delta(mfcc,order=2)
    plt.figure(figsize=(25,10))
    librosa.display.specshow(mfcc_delta2,sr=sample_rate,x_axis="time")
    plt.colorbar(format="%+2.f")
    plt.show()