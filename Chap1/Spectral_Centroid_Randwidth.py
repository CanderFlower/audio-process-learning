import librosa
import numpy as np
from matplotlib import pyplot as plt
from Amplitude_Envelope import Pad_Wave

def Calc_SC_RW(name:str):
    path = "D:\\Code\\Tests\\Audio\\audio_data\\"+name+".wav"
    waveform,sr = librosa.load(path,sr=None)
    sc = librosa.feature.spectral_centroid(y=waveform,sr=sr,n_fft=1024)
    rw = librosa.feature.spectral_bandwidth(y=waveform,sr=sr,n_fft=1024)
    return sc[0],rw[0]

if __name__ == "__main__":
    blues_sc,blues_rw = Calc_SC_RW("blues")
    jazz_sc,jazz_rw = Calc_SC_RW("jazz")
    rock_sc,rock_rw = Calc_SC_RW("rock")
    orchestra_sc,orchestra_rw = Calc_SC_RW("orchestra")
    
    fig,aix = plt.subplots(2,2)
    aix[0,0].plot(np.arange(len(blues_sc)),blues_sc)
    aix[0,0].set_title("blues")
    aix[0,1].plot(np.arange(len(jazz_sc)),jazz_sc)
    aix[0,1].set_title("jazz")
    aix[1,0].plot(np.arange(len(rock_sc)),rock_sc)
    aix[1,0].set_title("rock")
    aix[1,1].plot(np.arange(len(orchestra_sc)),orchestra_sc)
    aix[1,1].set_title("orchestra")
    
    #plt.show()
    
    aix[0,0].plot(np.arange(len(blues_rw)),blues_rw)
    aix[0,0].set_title("blues")
    aix[0,1].plot(np.arange(len(jazz_rw)),jazz_rw)
    aix[0,1].set_title("jazz")
    aix[1,0].plot(np.arange(len(rock_rw)),rock_rw)
    aix[1,0].set_title("rock")
    aix[1,1].plot(np.arange(len(orchestra_rw)),orchestra_rw)
    aix[1,1].set_title("orchestra")
    
    plt.show()