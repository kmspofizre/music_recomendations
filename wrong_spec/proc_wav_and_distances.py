# proc_wav_and_distances.py

import torch
import numpy as np
from wrong_spec.get_spec import extract_spectrogram
from wrong_spec.max_polling import apply_max_pooling
from thershhold_counter import normalize_spectrogram, count_above_threshold


def process_wav_and_find_distances(waveform_np, sample_rate, n_fft=2048, hop_length=512, kernel_size=5, stride=None,
                                   K=5, N=10):
    spectrogram = extract_spectrogram(waveform_np, sample_rate, n_fft, hop_length)

    pooled_spectrogram = apply_max_pooling(spectrogram, kernel_size, stride)

    normalized_spectrogram = normalize_spectrogram(pooled_spectrogram)

    low = 0.0
    high = 1.0
    best_threshold = 0.0

    while high - low > 1e-5:
        mid = (low + high) / 2.0
        count = count_above_threshold(normalized_spectrogram, mid)

        if count < K:
            low = mid
        elif count > N:
            high = mid
        else:
            best_threshold = mid
            break

    if best_threshold == 0.0:
        best_threshold = (low + high) / 2.0
    print(best_threshold)

    spectrogram_tensor = torch.tensor(normalized_spectrogram, dtype=torch.float32)
    column_sums = torch.sum(spectrogram_tensor, dim=1)
    indices_above_threshold = torch.nonzero(column_sums > best_threshold).squeeze().numpy()

    if indices_above_threshold.ndim == 0:
        indices_above_threshold = np.array([indices_above_threshold.item()])
    indices_above_threshold = np.sort(indices_above_threshold)

    if len(indices_above_threshold) < 2:
        distances = np.array([])
    else:
        distances = np.diff(indices_above_threshold)

    return distances

