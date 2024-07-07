import librosa
import os
import pickle
import numpy as np

AUDIO_PATH = r"D:\Code\Tests\Audio\Chap3\recordings"
SAVE_PATH = r"D:\Code\Tests\Audio\Chap3\recordings_feature"
MINMAX_PATH = r"D:\Code\Tests\Audio\Chap3\min_max.pkl"
FRAME_LENGTH = 512
HOP_LENGTH = 256
DURATION = 0.74
SAMPLE_RATE = 22050

class Preprocessor:
    def __init__(self, audio_path, save_path, frame_length, hop_length, duration, sample_rate):
        self.audio_path = audio_path
        self.save_path = save_path
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.duration = duration
        self.sample_rate = sample_rate
        self.expected_length = int(self.duration * self.sample_rate)
        self.min_max = {}
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        
    def normalize(self, array):
        min, max = array.min(), array.max()
        array = (array-min)/(max-min)
        return array, min, max
    
    def pad_and_cut_if_needed(self, signal):
        pad_length = self.expected_length - len(signal)
        if pad_length > 0:
            signal = np.pad(signal, (0, pad_length), mode="constant")
        return signal[:self.expected_length]
    
    def process_audio(self, filename):
        filepath = os.path.join(AUDIO_PATH, filename)
        signal, _ = librosa.load(filepath)
        
        signal = self.pad_and_cut_if_needed(signal)
        
        stft = librosa.stft(signal, n_fft=self.frame_length, hop_length=self.hop_length)[:-1]
        log_spectrogram = librosa.amplitude_to_db(stft)
        log_spectrogram, min, max = self.normalize(log_spectrogram)
        
        self.min_max[filename] = {
            "min": min, 
            "max": max
        }
        
        savepath = os.path.join(SAVE_PATH, filename)
        np.save(savepath+".npy", log_spectrogram)
        
    def preprocess(self):
        for _, _, files in os.walk(AUDIO_PATH):
            for file in files:
                self.process_audio(file)
        
        with open(MINMAX_PATH, "wb") as f:
            pickle.dump(self.min_max, f)

if __name__ == "__main__":
    preprocessor = Preprocessor(AUDIO_PATH, SAVE_PATH, FRAME_LENGTH, HOP_LENGTH, DURATION, SAMPLE_RATE)
    preprocessor.preprocess()