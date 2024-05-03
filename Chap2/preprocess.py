import json
import os
import math
import librosa
import numpy as np

DATASET_PATH = r"D:\Code\Tests\Audio\Chap2\Genre\genres_original"
JSON_PATH = r"D:\Code\Tests\Audio\Chap2\mfcc_delta2.json"
SAMPLES_PER_TRACK = 22050 * 30

def save_mfcc(dataset_path,json_path,n_mfcc=13,n_fft=2048,hop_length=512,num_segment=10):
    
    data = {
        "mapping" : [],
        "mfcc" : [],
        "label" : []
    }
    
    samples_per_segment = int(SAMPLES_PER_TRACK/num_segment)
    expected_mfcc_length = math.ceil(samples_per_segment / hop_length)
    
    for i,(root,dirs,files) in enumerate(os.walk(dataset_path)):
        if root is dataset_path:
            continue
        
        genre_name = root.split("\\")[-1]
        data["mapping"].append(genre_name)
        genre_num = i-1
        
        print("Processing [{}]".format(genre_name))
        
        for file in files:
            file_path = os.path.join(root,file)
            waveform,sr = librosa.load(file_path)
            
            for seg_num in range(num_segment):
                #print("File[{}], Seg#[{}]".format(file,seg_num))
                start_pos = seg_num * samples_per_segment
                end_pos = start_pos + samples_per_segment
                curr_mfcc = librosa.feature.mfcc(
                    y=waveform[start_pos:end_pos],
                    sr=sr,
                    n_mfcc=n_mfcc,
                    n_fft=n_fft,
                    hop_length=hop_length
                )
                if curr_mfcc.shape[1]!=expected_mfcc_length:
                    continue
                mfcc_delta = librosa.feature.delta(curr_mfcc,order=1)
                mfcc_delta2 = librosa.feature.delta(curr_mfcc,order=2)
                curr_mfcc=curr_mfcc.T
                mfcc_delta = mfcc_delta.T
                mfcc_delta2 = mfcc_delta2.T
                curr_mfcc = np.stack([curr_mfcc,mfcc_delta,mfcc_delta2],axis=2)
                data["mfcc"].append(curr_mfcc.tolist())
                data["label"].append(genre_num)
                
    with open(json_path,mode="w") as f:
        json.dump(data,f,indent=4)
    
if __name__ == "__main__":
    save_mfcc(DATASET_PATH,JSON_PATH)