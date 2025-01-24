import torch
import torchaudio
import numpy as np


def apply_max_pooling(spectrogram, kernel_size=5, stride=None):
    spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32)

    if spectrogram_tensor.ndim == 2:
        spectrogram_tensor = spectrogram_tensor.unsqueeze(0).unsqueeze(0)  # Добавляем batch и channel размеры

    elif spectrogram_tensor.ndim == 3:
        spectrogram_tensor = spectrogram_tensor.unsqueeze(1)  # Добавляем channel размер

    if stride is None:
        stride = kernel_size

    pooled_spectrogram = torch.nn.functional.max_pool2d(
        spectrogram_tensor,
        kernel_size=kernel_size,
        stride=stride
    )


    return pooled_spectrogram.squeeze().numpy()
