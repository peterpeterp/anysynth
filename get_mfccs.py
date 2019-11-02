# Code source: Brian McFee
# License: ISC
# sphinx_gallery_thumbnail_number = 6
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt

import librosa
import librosa.display

os.chdir('/home/peterp/synthi')

n_mfcc = 30

for filename in ['air_horn_1.mp3']:
    y, sr = librosa.load(filename)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), gridspec_kw={'width_ratios':[3,1]})

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, dct_type=2)
    librosa.display.specshow(mfccs, x_axis='time', ax=axes[0,0])

    axes[0,1].plot(np.mean(mfccs,axis=1),range(1,n_mfcc+1))

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mfcc,fmax=5000, hop_length=1000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr,fmax=5000, ax=axes[1,0])

    axes[1,1].plot(np.mean(S_dB,axis=1),range(1,n_mfcc+1))



    plt.savefig(filename.replace('.mp3','.pdf'))
    plt.close('all')




    # librosa.display.specshow(mfcc, x_axis='time',y_axis='mel', sr=sr)
    #
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel-frequency spectrogram')
    # plt.tight_layout()
    # plt.savefig(filename.replace('.mp3','.pdf'))
    # plt.close('all')
