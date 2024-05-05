# Audio Processing Learning

This repository contains my code and projects related to learning audio processing and deep learning based on audio.

The database used in the projects is not included to avoid copyright issues.

## Chapters

### Chapter 1
- Signal Processing Techniques:
  - Amplitude Envelope Calculation
    - Note: Try different np.pad modes and choose "wrap"
  - Root Mean Square (RMS) Calculation
    - Note: Compare RMS, ZCR, AE
  - Spectral Centroid and Bandwidth Calculation
  - Zero Crossing Rate Computation
  - Short-Time Fourier Transform (STFT) Implementation
    - Note: Compare with ZCR, since ZCR reflects frequency
  - Mel-frequency Cepstral Coefficients (MFCC) Computation
    - Note: Compare with ZCR

### Chapter 2
- Deep Learning for Audio with Python:
  - Artificial Neuron Implementation
  - Multi-Layer Perceptron (MLP) Implementation
    - Note1: The derivative of the sigmoid can be calculated with the activations saved
    - Note2: Calculation of matrix derivative
    - Note3: Simply use shape to ensure the priority of dot multiplication
    - Note4: Sigmoid can only be used to models that outputs in $[0,1]$
  - Data Preprocessing Utilities
    - Note: Use .json to store data
  - Music Genre Classification based on MLP
    - Note1: More layers and nodes do not necessarily mean better performance. It predominantly depends on the structure of the model.
    - Note2: Attempts were made to train the MLP using MFCC delta features (mfcc_delta2), but the results were found to be unsatisfactory.
    - Note3: Dropout is useful.
    - Note4: With ploting the history, it can be known that too much epochs does not improve the performance.
  - Music Genre Classification based on CNN
    - Note1: CNN performs much better than simply MLP, since CNN can learn the relationships and features from spectrum.
    - Note2: Under the same condition(100 epochs), just using MFCC has an accuracy of 75.35%, while using MFCC_delta2 as 3 channels has an accuracy of 75.51%, indicating there is still no significant difference between them. When using 200 epochs for MFCC_delta2(also changed learning_rate, dropout and regularizer), the accuracy will only become 75.83%, and the history plot shows that the accuracy will not change significantly after about the 90-th epoch.
  - Music Genre Classification based on RNN-LSTM
    - Note1: Pay attention to the input shapes for different models. LSTM requires 2-dimensional data (for each input), while CNN requires 3-dimensional data since it has an extra dimension for channels (which will not be used for MFCC).
    - Note2: RNN-LSTM has an accuracy of 67.75% (after 150 eopchs), which is lower than CNN. It may can be improved through some ways like add the number of cells.


## References

- [音频信号处理及深度学习教程](https://space.bilibili.com/550180844/channel/collectiondetail?sid=1034039&ctype=0)
- [Audio Signal Processing for Machine Learning](https://www.youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0)
- [Deep Learning (for Audio) with Python](https://www.youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf)