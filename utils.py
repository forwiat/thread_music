import librosa
import numpy as np
from hparams import hparams
hp = hparams()
def get_spectrogram(fpath: str):
    x, _ = librosa.load(fpath, hp.SR)
    x, _ = librosa.effects.trim(x)
    fft = librosa.stft(x, n_fft=hp.N_FFT, hop_length=hp.HOP_LENGTH, win_length=hp.WIN_LENGTH)
    mag = np.abs(fft)
    mag = 20 * np.log10(np.maximum(1e-5, mag)) # to db
    mag = np.clip((mag - hp.REF_DB + hp.MAX_DB) / hp.MAX_DB, 1e-8, 1)
    mag = np.transpose(mag, (1, 0))
    n_frames, feature_size = mag.shape
    if n_frames >= hp.SEGMENT:
        start_index = np.random.randint(low=0, high=n_frames - hp.SEGMENT + 1)
        segmented_mag = mag[start_index: start_index + hp.SEGMENT, :]
    else:
        segmented_mag = np.concatenate((mag, np.zeros((hp.SEGMENT - n_frames, feature_size))), axis=0)
    return segmented_mag
