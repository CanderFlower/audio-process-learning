import librosa
import numpy as np
from matplotlib import pyplot as plt
from Amplitude_Envelope import Pad_Wave

def Calc_ZCR(waveform,frame_length:int,hop_length:int):
    frame_num,waveform = Pad_Wave(waveform,frame_length,hop_length)
    wave_ZCR = []
    for i in range(frame_num):
        curr_frame = waveform[i*hop_length:i*hop_length+frame_length]
        frame_s = curr_frame[0:frame_length-1]
        frame_e = curr_frame[1:frame_length]
        curr_ZCR = np.sum(np.abs(np.sign(frame_s)-np.sign(frame_e)))/2.0/frame_length
        wave_ZCR.append(curr_ZCR)
    return np.array(wave_ZCR)
    
if __name__ == "__main__":
    wave_path = r"D:\Code\Tests\Audio\audio_data\music_piano.wav"
    waveform, sample_rate = librosa.load(wave_path,sr=None)

    frame_length = 1024
    hop_rate = 0.5
    hop_length = int(hop_rate * frame_length)
    
    wave_ZCR = Calc_ZCR(waveform,frame_length,hop_length)
    wave_ZCR_librosa = librosa.feature.zero_crossing_rate(waveform,frame_length=frame_length,hop_length=hop_length)[0][1:]
    
    
    frames_num = np.arange(len(wave_ZCR))
    frames_at = librosa.frames_to_time(frames_num,sr=sample_rate,hop_length=hop_length)
    plt.plot(frames_at,wave_ZCR,"r")
    plt.plot(frames_at,wave_ZCR_librosa,"g")

    plt.title("Zero Crossing Rate")

    librosa.display.waveshow(waveform,sr=sample_rate,max_points=len(waveform)+100,color="blue",offset=0)
  
    plt.show()  
    
    print("Success!")
    