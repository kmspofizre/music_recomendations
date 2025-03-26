from fingerprinting import extract_spectrogram, find_peaks, generate_hashes
from storage import load_fingerprint_database
import pickle
import librosa
from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, label, find_objects 


def load_song_id_map(filepath="song_id_map.pkl"):
    if not os.path.exists(filepath):
        print(f"Song ID map file {filepath} does not exist.")
        return {}
    with open(filepath, 'rb') as f:
        song_id_map = pickle.load(f)
    print(f"Song ID map loaded from {filepath}.")
    return song_id_map


def visualize_spectrogram_with_peaks(spectrogram, peaks, title="Spectrogram with Peaks"):
    plt.figure(figsize=(12, 6))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Время')
    plt.ylabel('Частота')
    for peak in peaks:
        freq, time = peak
        plt.scatter(time, freq, marker='x', color='red')
    plt.title(title)
    plt.show()


def find_peaks_adaptive(spectrogram, neighborhood_size=20, percentile=90, visualize=False, title="Spectrogram"):
    threshold = np.percentile(spectrogram, percentile)
    print(f"Computed threshold based on {percentile}th percentile: {threshold:.2f} dB")

    local_max = maximum_filter(spectrogram, size=neighborhood_size) == spectrogram
    detected_peaks = local_max & (spectrogram >= threshold)
    labeled, num_features = label(detected_peaks)
    slices = find_objects(labeled)
    peaks = []
    for dy, dx in slices:
        if dy is None or dx is None:
            continue  # Пропуск пустых срезов
        freq = (dy.start + dy.stop - 1) // 2
        time = (dx.start + dx.stop - 1) // 2
        peaks.append((freq, time))

    if visualize:
        visualize_spectrogram_with_peaks(spectrogram, peaks, title)

    return peaks


def recognize_song(query_filepath, db_path="fingerprint_db.pkl", song_id_map_path="song_id_map.pkl",
                   n_fft=2048, hop_length=512, neighborhood_size=20, percentile=90,
                   fan_value=15, delta_freq=200, delta_time=200):

    fingerprint_db = load_fingerprint_database(db_path)
    song_id_map = load_song_id_map(song_id_map_path)

    y, sr = librosa.load(query_filepath, sr=None, mono=True)
    print(f"Loaded query file: {query_filepath}, sample rate={sr}, duration={len(y) / sr:.2f} seconds")
    spectrogram = extract_spectrogram(y, sr, n_fft, hop_length)
    print(f"Spectrogram shape: {spectrogram.shape}, min: {spectrogram.min()}, max: {spectrogram.max()} dB")

    # Поиск пиков с адаптивным порогом и визуализацией
    peaks = find_peaks_adaptive(spectrogram, neighborhood_size, percentile, visualize=True,
                                title=f"Spectrogram with Peaks for Query: {os.path.basename(query_filepath)}")
    print(f"Found {len(peaks)} peaks in query file.")

    # Генерация хешей
    hashes = generate_hashes(peaks, fan_value, delta_freq, delta_time)
    print(f"Generated {len(hashes)} hashes for query file.")

    if not hashes:
        print("No hashes generated for the query file.")
        return []

    matches = []
    for hash_val, offset in hashes:
        if hash_val in fingerprint_db:
            for song_id, song_offset in fingerprint_db[hash_val]:
                difference = offset - song_offset
                matches.append((song_id, difference))

    if not matches:
        print("No matches found.")
        return []

    song_match_counts = defaultdict(lambda: defaultdict(int))
    for song_id, diff in matches:
        song_match_counts[song_id][diff] += 1

    song_scores = {}
    for song_id, diffs in song_match_counts.items():
        max_count = max(diffs.values())
        song_scores[song_id] = max_count

    sorted_songs = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)

    top_results = sorted_songs[:5]

    top_results_named = [
        (song_id_map[song_id]["Name"], score)
        for song_id, score in top_results
    ]

    return top_results_named


if __name__ == "__main__":
    query_file = "../wav_folder/Linkin_Park_-_Breaking_The_Habit.wav"
    results = recognize_song(query_file)
    print(results)
    if results:
        print("\nПохожие песни:")
        for filename, score in results:
            print(f"{filename}, совпадений: {score}")
    else:
        print("Никаких совпадений не найдено.")
