# get_spec.py

import torchaudio
import torch
import numpy as np


def extract_spectrogram(waveform_np, sample_rate, n_fft=2048, hop_length=512, win_length=None, window_fn=torch.hann_window):
    waveform_tensor = torch.tensor(waveform_np, dtype=torch.float32)

    if waveform_tensor.ndim == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)

    window = window_fn(n_fft) if window_fn else None

    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window_fn=window_fn,
        power=None
    )
    spectrogram = spectrogram_transform(waveform_tensor)

    if spectrogram.is_complex():
        spectrogram = torch.abs(spectrogram)

    return spectrogram.numpy()

