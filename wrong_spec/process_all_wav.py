import os
import numpy as np
import torchaudio
from wrong_spec.get_spec import extract_spectrogram
from wrong_spec.max_polling import apply_max_pooling
from thershhold_counter import normalize_spectrogram


def process_wav_and_find_distances(waveform_np, sample_rate, n_fft=2048, hop_length=512, kernel_size=5, stride=None,
                                   threshold=0.9):
    spectrogram = extract_spectrogram(waveform_np, sample_rate, n_fft, hop_length)

    pooled_spectrogram = apply_max_pooling(spectrogram, kernel_size, stride)

    normalized_spectrogram = normalize_spectrogram(pooled_spectrogram)

    column_sums = np.sum(normalized_spectrogram, axis=0)

    indices_above_threshold = np.where(column_sums > threshold)[0]
    print(f"Number of moments above threshold {threshold}: {len(indices_above_threshold)}")

    if len(indices_above_threshold) < 2:
        distances = np.array([])
    else:
        distances = np.diff(indices_above_threshold)

    return distances


def build_feature_library(
        folder_path,
        n_fft=2048,
        hop_length=512,
        kernel_size=5,
        stride=None,
        threshold=0.9
):

    feature_library = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            waveform, sample_rate = torchaudio.load(file_path)
            waveform_np = waveform.numpy()

            distances = process_wav_and_find_distances(
                waveform_np, sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                kernel_size=kernel_size,
                stride=stride,
                threshold=threshold
            )

            feature_library.append((filename, distances))

    return feature_library


def find_top_k_similar_songs(
        query_wav_path,
        feature_library,
        k=5,
        n_fft=2048,
        hop_length=512,
        kernel_size=5,
        stride=None,
        threshold=0.9
):
    query_waveform, query_sample_rate = torchaudio.load(query_wav_path)
    query_waveform_np = query_waveform.numpy()
    query_distances = process_wav_and_find_distances(
        query_waveform_np, query_sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        kernel_size=kernel_size,
        stride=stride,
        threshold=threshold
    )
    print(f"Query file distances: {len(query_distances)}")

    similarities = []

    for filename, dist_vector in feature_library:
        if len(query_distances) == len(dist_vector):
            distance = np.linalg.norm(query_distances - dist_vector)
        else:
            min_length = min(len(query_distances), len(dist_vector))
            if min_length == 0:
                distance = float('inf')
            else:
                distance = np.linalg.norm(query_distances[:min_length] - dist_vector[:min_length])

        similarities.append((filename, distance))
        print(f"Compared with {filename}: distance = {distance}")

    similarities.sort(key=lambda x: x[1])

    top_k = similarities[:k]
    return top_k


if __name__ == "__main__":
    library = build_feature_library("../wav_folder")

    query_file = "../wav_folder/Eminem_-_Without_Me.wav"
    top_5 = find_top_k_similar_songs(query_file, library, k=6)

    print("\nСамые похожие песни:")
    for filename, dist_val in top_5:
        print(f"{filename}, distance = {dist_val}")


test_songs = [
        {"query_filepath": "../wav_folder/Aarne_NLE_Choppa_Imanbek_-_ice_ice_ice.wav", "true_song_id": 1},
        {"query_filepath": "../wav_folder/AnnenMayKantereit_-_Come_Together.wav", "true_song_id": 2},
        {"query_filepath": "../wav_folder/AnnenMayKantereit_-_Ich_geh_heut_nicht_mehr_tanzen.wav", "true_song_id": 3},
        {"query_filepath": "../wav_folder/Big_Baby_Tape_Aarne_-_Supersonic.wav", "true_song_id": 4},
        {"query_filepath": "../wav_folder/Boyce_Avenue_-_Careless_Whisper.wav", "true_song_id": 5},
        {"query_filepath": "../wav_folder/Coldplay_-_The_Scientist.wav", "true_song_id": 6},
        {"query_filepath": "../wav_folder/Coldplay_-_Yellow.wav", "true_song_id": 7},
        {"query_filepath": "../wav_folder/Eminem_-_Lose_Yourself.wav", "true_song_id": 8},
        {"query_filepath": "../wav_folder/Eminem_-_Not_Afraid.wav", "true_song_id": 9},
        {"query_filepath": "../wav_folder/Eminem_-_Without_Me.wav", "true_song_id": 10},
        {"query_filepath": "../wav_folder/Fort_Minor_feat_Styles_Of_Beyond_Bobo_-_Believe_Me_feat_Bobo_Styles_Of_Beyond.wav", "true_song_id": 11},
        {"query_filepath": "../wav_folder/FRIENDLY_THUG_52_NGG_Big_Baby_Tape_-_Legit_Check.wav", "true_song_id": 12},
        {"query_filepath": "../wav_folder/Linkin_Park_-_Bleed_It_Out.wav", "true_song_id": 13},
        {"query_filepath": "../wav_folder/Linkin_Park_-_Breaking_The_Habit.wav", "true_song_id": 14},
        {"query_filepath": "../wav_folder/Linkin_Park_-_Faint.wav", "true_song_id": 15},
        {"query_filepath": "../wav_folder/Linkin_Park_-_In_The_End.wav", "true_song_id": 16},
        {"query_filepath": "../wav_folder/Linkin_Park_-_Numb.wav", "true_song_id": 17},
        {"query_filepath": "../wav_folder/Linkin_Park_-_One_More_Light.wav", "true_song_id": 18},
        {"query_filepath": "../wav_folder/Linkin_Park_-_What_Ive_Done.wav", "true_song_id": 19},
        {"query_filepath": "../wav_folder/Loc-Dog_-_V_tojj_vesne.wav", "true_song_id": 20},
        # ...
    ]