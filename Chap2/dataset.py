import torchaudio
from torch.utils.data import Dataset
import os
import librosa
from matplotlib import pyplot as plt

def download_dataset():
    torchaudio.datasets.YESNO(root=r"D:\Code\Tests\Audio\audio_data\Chap2\yesno",download=True)
    print("Downloaded")
    
class MyDataset(Dataset):
    def __init__(self,path_name,label_name):
        super().__init__()
        self.path_name = path_name
        self.label_name = label_name
        self.path = os.path.join(path_name,label_name)
        self.audio = os.listdir(self.path)
    def __getitem__(self, index):
        audio_name = self.audio[index]
        audio_path = os.path.join(self.path,audio_name)
        waveform,sr = librosa.load(audio_path)
        return waveform,sr,audio_name
    
if __name__ == "__main__":
    path_name = r"D:\Code\Tests\Audio\Chap2\yesno"
    label_name = "waves_yesno"
    dataset = MyDataset(path_name,label_name)
    waveform, sample_rate, audio_name = dataset[3]
    plt.figure(figsize = (25,10))
    librosa.display.waveshow(waveform,sr=sample_rate,color="blue")
    plt.title(audio_name+" waveform")
    plt.show()