import librosa
import numpy as np
from matplotlib import pyplot as plt
from Amplitude_Envelope import Pad_Wave
from Root_Mean_Square import Calc_Wave_RMS
from Zero_Crossing_Rate import Calc_ZCR

if __name__ == "__main__":
    wave_path = r"D:\Code\Tests\Audio\audio_data\jazz.wav"
    waveform, sample_rate = librosa.load(wave_path,sr=None)
    
    frame_length = 1024
    hop_rate = 0.5
    hop_length = int(hop_rate * frame_length)
    
    waveform = Pad_Wave(waveform,frame_length,hop_length)[1]
    
    s_scale = librosa.stft(waveform,hop_length=hop_length,n_fft=frame_length)
    
    y_scale = np.abs(s_scale) ** 2
    '''
    print(y_scale.shape)
    print(type(y_scale[0][0]))
    '''
    y_log_scale = librosa.power_to_db(y_scale)
    plt.figure(figsize=(25,10))
    #librosa.display.specshow(y_scale,sr=sample_rate,hop_length=hop_length,n_fft=frame_length,x_axis="time",y_axis="log")
    librosa.display.specshow(y_log_scale,sr=sample_rate,hop_length=hop_length,n_fft=frame_length,x_axis="time",y_axis="log")
    
    waveform_RMS = Calc_Wave_RMS(waveform,frame_length,hop_length)
    frames_num = np.arange(len(waveform_RMS))
    frames_at = librosa.frames_to_time(frames_num,sr=sample_rate,hop_length=hop_length)
    plt.plot(frames_at,waveform_RMS*800,"b")
    
    waveform_ZCR = Calc_ZCR(waveform,frame_length,hop_length)
    plt.plot(frames_at,waveform_ZCR*1200,"g")
    
    plt.colorbar(format="%+2.f")
    plt.title("Short-Time Fourier Transform")
    plt.show()
