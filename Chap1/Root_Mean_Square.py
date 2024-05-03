import librosa
import numpy as np
from matplotlib import pyplot as plt
from Amplitude_Envelope import Pad_Wave

def RMS(array:np.ndarray):
    return (np.sum(array**2)/len(array))**0.5


def Calc_Wave_RMS(waveform,frame_length:int,hop_length:int):
    frame_num,waveform = Pad_Wave(waveform,frame_length,hop_length)
    waveform_rms = []
    for i in range(frame_num):
        current_frame = waveform[i*hop_length:i*hop_length+frame_length]
        waveform_rms.append(RMS(current_frame))
    return np.array(waveform_rms)

if __name__ == "__main__":
    wave_path = r"D:\Code\Tests\Audio\audio_data\music_piano.wav"
    waveform, sample_rate = librosa.load(wave_path,sr=None)

    frame_length = 1024
    hop_rate = 0.5
    hop_length = int(hop_rate * frame_length)
    
    waveform_RMS_librosa = librosa.feature.rms(y=waveform,frame_length=frame_length,hop_length=hop_length).T[:,0]
    waveform_RMS_librosa = np.pad(waveform_RMS_librosa,(0,1),mode="edge")
    #print(len(waveform_RMS_librosa))
    
    #waveform_RMS = Calc_Wave_RMS(waveform,frame_length,hop_length)
    #print(len(waveform_RMS))
    
    waveform = np.pad(waveform,(int(frame_length//2),int(frame_length//2)),mode="constant")
    
    waveform_RMS = Calc_Wave_RMS(waveform,frame_length,hop_length)
    #print(len(waveform_RMS))
    

    frames_num = np.arange(len(waveform_RMS))
    frames_at = librosa.frames_to_time(frames_num,sr=sample_rate,hop_length=hop_length)
    plt.plot(frames_at,waveform_RMS,"r")
    plt.plot(frames_at,waveform_RMS_librosa,"g")

    plt.title("Root Mean Square")

    librosa.display.waveshow(waveform,sr=sample_rate,max_points=len(waveform)+100,color="blue",offset=0)
  
    print("biasRMS:",RMS(waveform_RMS_librosa-waveform_RMS))
    plt.show()  
    
    print("Success!")