from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import librosa
import matplotlib.pyplot as plt
import numpy as np


def load_audio(file_path, target_sr=16000):
    """
    Загружает аудиофайл и преобразует его к целевой частоте дискретизации.

    Args:
        file_path (str): Путь к аудиофайлу.
        target_sr (int): Целевая частота дискретизации.

    Returns:
        y (numpy.ndarray): Аудиосигнал.
        sr (int): Частота дискретизации.
    """
    y, sr = librosa.load(file_path, sr=target_sr)
    return y, sr


audio_path = "ZOLOTO_-_Ulicy_zhdali.wav"


audio, sr = load_audio(audio_path)
print(f"Частота дискретизации: {sr} Гц")
print(f"Длительность аудио: {len(audio)/sr:.2f} секунд")


plt.figure(figsize=(14, 5))
librosa.display.waveshow(audio, sr=sr)
plt.title("Waveform")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.show()


inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)


with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states


pre_logits_vector = hidden_states[-2]


embeddings = pre_logits_vector.mean(dim=1)
print("Размерность вектора предпоследнего слоя:", pre_logits_vector.shape)

embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Размерность: [batch_size, hidden_dim]


vector = pre_logits_vector.squeeze().numpy()

torch.save(embeddings_norm, "pre_logits_vector.pt")
print("Вектор сохранён в 'pre_logits_vector.pt'")