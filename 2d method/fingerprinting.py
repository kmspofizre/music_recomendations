import os
import librosa
import hashlib
import numpy as np
from scipy.ndimage import maximum_filter, label, find_objects
import pickle


def extract_spectrogram(y, sr, n_fft=2048, hop_length=512):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def find_peaks(spectrogram, neighborhood_size=20, threshold=10):
    local_max = maximum_filter(spectrogram, size=neighborhood_size) == spectrogram
    detected_peaks = local_max & (spectrogram >= threshold)
    labeled, num_features = label(detected_peaks)
    slices = find_objects(labeled)
    peaks = []
    for dy, dx in slices:
        freq = (dy.start + dy.stop - 1) // 2
        time = (dx.start + dx.stop - 1) // 2
        peaks.append((freq, time))
    return peaks


def generate_hashes(peaks, fan_value=15, delta_freq=200, delta_time=200):
    hashes = []
    peaks = sorted(peaks, key=lambda x: x[1])
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):
                freq1, time1 = peaks[i]
                freq2, time2 = peaks[i + j]
                if (abs(freq1 - freq2) < delta_freq) and (abs(time1 - time2) < delta_time):
                    hash_str = f"{freq1}|{freq2}|{time2 - time1}"
                    hash_val = hashlib.sha1(hash_str.encode('utf-8')).hexdigest()[0:20]
                    hashes.append((hash_val, time1))
    return hashes
