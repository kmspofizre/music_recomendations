
import os
import librosa
import hashlib
import numpy as np
from scipy.ndimage import maximum_filter, label, find_objects
import pickle
import matplotlib.pyplot as plt


def extract_spectrogram(y, sr, n_fft=2048, hop_length=512):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def visualize_spectrogram(spectrogram, title="Spectrogram"):
    plt.figure(figsize=(12, 6))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Время')
    plt.ylabel('Частота')
    plt.title(title)
    plt.show()


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


def find_peaks(spectrogram, neighborhood_size=20, threshold=-50, visualize=False, title="Spectrogram"):
    local_max = maximum_filter(spectrogram, size=neighborhood_size) == spectrogram
    detected_peaks = local_max & (spectrogram >= threshold)
    labeled, num_features = label(detected_peaks)
    slices = find_objects(labeled)
    peaks = []
    for dy, dx in slices:
        if dy is None or dx is None:
            continue
        freq = (dy.start + dy.stop - 1) // 2
        time = (dx.start + dx.stop - 1) // 2
        peaks.append((freq, time))

    if visualize:
        visualize_spectrogram_with_peaks(spectrogram, peaks, title)

    return peaks


def generate_hashes(peaks, fan_value=15, delta_freq=200, delta_time=200):
    hashes = []
    peaks = sorted(peaks, key=lambda x: x[1])  # Sort by time
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


def create_database(filepath="fingerprint_db.pkl"):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            fingerprint_db = pickle.load(f)
        print(f"Fingerprint database loaded from {filepath}. Number of unique hashes: {len(fingerprint_db)}")
    else:
        fingerprint_db = {}
        print(f"Created new fingerprint database.")
    return fingerprint_db


def save_fingerprint_database(fingerprint_db, filepath="fingerprint_db.pkl"):
    with open(filepath, 'wb') as f:
        pickle.dump(fingerprint_db, f)
    print(f"Fingerprint database saved to {filepath}.")


def parse_artist_from_filename(filename):
    if "_-_" in filename:
        artist = filename.split("_-_")[0]
    else:
        artist = "Unknown"
    return artist


def fingerprint_song(fingerprint_db, song_id, filepath, n_fft=2048, hop_length=512, neighborhood_size=20, threshold=-50,
                    fan_value=15, delta_freq=200, delta_time=200):
    y, sr = librosa.load(filepath, sr=None)
    print(f"Loaded {filepath}: sample rate={sr}, duration={len(y) / sr:.2f} seconds")
    spectrogram = extract_spectrogram(y, sr, n_fft, hop_length)
    print(f"Spectrogram shape: {spectrogram.shape}, min: {spectrogram.min()}, max: {spectrogram.max()} dB")
    peaks = find_peaks(spectrogram, neighborhood_size, threshold, visualize=False,
                       title=f"Spectrogram with Peaks for {os.path.basename(filepath)}")
    print(f"Found {len(peaks)} peaks in {filepath}")
    hashes = generate_hashes(peaks, fan_value, delta_freq, delta_time)
    print(f"Generated {len(hashes)} hashes for {filepath}")
    if not hashes:
        print(f"No hashes generated for {filepath}")
    for hash_val, offset in hashes:
        if hash_val in fingerprint_db:
            fingerprint_db[hash_val].append((song_id, offset))
        else:
            fingerprint_db[hash_val] = [(song_id, offset)]
    print(f"Fingerprinted {filepath}: {len(hashes)} hashes added.")


def index_songs(folder_path, db_path="fingerprint_db.pkl"):
    fingerprint_db = create_database(db_path)
    song_id = 1
    song_id_map = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".wav"):
            filepath = os.path.join(folder_path, filename)
            artist = parse_artist_from_filename(filename)
            print(f"Processing song: {filename}")
            genre = input(f"Введите жанр для '{filename}': ")
            fingerprint_song(fingerprint_db, song_id, filepath)

            song_id_map[song_id] = {
                "Name": filename,
                "Artist": artist,
                "Genre": genre
            }
            print(f"Indexed {filename} with song_id={song_id}, artist='{artist}', genre='{genre}'")
            song_id += 1
    save_fingerprint_database(fingerprint_db, db_path)
    with open("song_id_map.pkl", "wb") as f:
        pickle.dump(song_id_map, f)
    print("Song ID map saved to song_id_map.pkl.")


if __name__ == "__main__":
    songs_folder = "../wav_folder"
    index_songs(songs_folder)
