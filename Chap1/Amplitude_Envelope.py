import math
import librosa
import numpy as np
from matplotlib import pyplot as plt


def Pad_Wave(waveform,frame_length:int,hop_length:int):
    if (len(waveform)-frame_length)%hop_length != 0:
        frame_num = int((len(waveform)-frame_length)/hop_length) + 2
        new_len = frame_num*hop_length+(frame_length-hop_length)
        pad_len = new_len - len(waveform)
        waveform = np.pad(waveform,(0,pad_len),mode="wrap")
    frame_num = int((len(waveform)-frame_length)/hop_length) + 1
    return frame_num,waveform


def Calc_Amplitude_Envelope(waveform,frame_length:int,hop_length:int):
    frame_num,waveform = Pad_Wave(waveform,frame_length,hop_length)
    waveform_ae = []
    for i in range(frame_num):
        current_frame = waveform[i*hop_length:i*hop_length+frame_length]
        waveform_ae.append(max(current_frame))
    return np.array(waveform_ae)

if __name__ == "__main__":
    wave_path = r"D:\Code\Tests\Audio\audio_data\music_piano.wav"
    waveform, sample_rate = librosa.load(wave_path,sr=None)
    frame_length = 1024
    hop_rate = 0.5
    hop_length = int(hop_rate * frame_length)
    waveform_AE = Calc_Amplitude_Envelope(waveform,frame_length,hop_length)

    frames_num = np.arange(len(waveform_AE))
    frames_at = librosa.frames_to_time(frames_num,sr=sample_rate,hop_length=hop_length)
    '''
    print("Sample rate:",sample_rate)
    print(frames_at[0],frames_at[1],frames_at[2631])
    print(len(frames_at))
    '''
    #plt.figure(10,20)
    '''
    len = len(frames_at)
    wav_len = librosa.get_duration(y=waveform,sr=sample_rate)
    for i in range(len):
        frames_at.append(frames_at[i]+wav_len)
        waveform_AE.append(waveform_AE[i])
    '''
    wav_len = librosa.get_duration(y=waveform,sr=sample_rate)
    frames_at_after = frames_at + wav_len
    frames_at_doubled = np.append(frames_at,frames_at_after)
    waveform_AE_doubled = np.append(waveform_AE,waveform_AE)
    #plt.plot(frames_at,waveform_AE,"r")
    plt.plot(frames_at,waveform_AE,"r")

    plt.title("Piano Amplitude Envelope")

    all_frame = []
    hop_duration = 1.0/sample_rate
    for i in range(len(waveform)):
        all_frame.append(hop_duration*i)
    wave_rev = -waveform
    plt.plot(all_frame,waveform,"g")

    librosa.display.waveshow(waveform,sr=sample_rate,max_points=len(waveform)+100,color="blue",offset=0)
    #librosa.display.waveshow(waveform,sr=sample_rate,color="blue",offset=wav_len)

    print("sample rate:",sample_rate)
    print("point num:",len(waveform))
    plt.show()
    print("Success!")
