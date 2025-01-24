import torch
import numpy as np


def normalize_spectrogram(spectrogram):
    spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32)

    min_val = torch.min(spectrogram_tensor)
    max_val = torch.max(spectrogram_tensor)
    normalized_spectrogram = (spectrogram_tensor - min_val) / (max_val - min_val + 1e-8)

    return normalized_spectrogram.numpy()


def count_above_threshold(spectrogram, threshold):
    spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32)
    column_sums = torch.sum(spectrogram_tensor, dim=1)
    count_above = (column_sums > threshold).sum().item()

    return count_above
