import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm 
import h5py

from utils.util import read_audio

EPS = 1E-8
def LogMelExtractor(data, sr, 
                    mel_bins = 128, 
                    hoplen = 15,
                    winlen = 25,
                    log = True, 
                    snv = False):
    
    # define parameters, 
    MEL_ARGS = {
        'n_mels': mel_bins,
        'sr' : sr,
        'hop_length': int(sr * hoplen / 1000),
        'win_length': int(sr * winlen / 1000)
        }
    mel_spectrogram = librosa.feature.melspectrogram(data, **MEL_ARGS) 
    if log:
        mel_spectrogram = np.log(mel_spectrogram + EPS)
    if snv:
        mel_spectrogram = standard_normal_variate(mel_spectrogram)
    
    return mel_spectrogram

def MfccExtractor(data, sr, 
                  n_mfcc = 13,
                  window = 'hamming',
                  hoplen = 15,
                  winlen = 25,
                  snv = False):
        
    # define parameters
    MFCC_ARGS = {
        'n_mfcc' : n_mfcc,
        'sr': sr,
        'window': window,
        'hop_length': int(sr * hoplen / 1000),
        'win_length': int(sr * winlen / 1000)
        }

    EPS = np.spacing(1)
    mfcc = librosa.feature.mfcc(data, **MFCC_ARGS)

    if snv:
        mfcc = standard_normal_variate(mfcc)
    
    return mfcc


def standard_normal_variate(data):
    mean = np.mean(data)
    std = np.std(data)

    return (data - mean) / std

if __name__ == "__main__":
    df = pd.read_csv('data/label.csv')
    with h5py.File('lms_128.h5','w') as store:
        for filename in tqdm(df['filename']):
            audio, fs = read_audio(filename, filter=True)
            feature = LogMelExtractor(audio, fs, log=True, snv=False)
            #feature = MfccExtractor(audio, fs, snv=False)
            basename = filename
            store[filename] = feature


