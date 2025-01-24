from fingerprinting import extract_spectrogram, find_peaks, generate_hashes
from storage import save_fingerprint_database, load_fingerprint_database
import librosa
import pickle
import os


def index_songs(folder_path, db_path="fingerprint_db.pkl", n_fft=2048, hop_length=512,
                neighborhood_size=20, threshold=10, fan_value=15, delta_freq=200, delta_time=200):
    fingerprint_db = load_fingerprint_database(db_path)
    song_id = 1
    song_id_map = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".wav"):
            filepath = os.path.join(folder_path, filename)
            y, sr = librosa.load(filepath, sr=None)
            spectrogram = extract_spectrogram(y, sr, n_fft, hop_length)
            peaks = find_peaks(spectrogram, neighborhood_size, threshold)
            hashes = generate_hashes(peaks, fan_value, delta_freq, delta_time)

            for hash_val, offset in hashes:
                if hash_val in fingerprint_db:
                    fingerprint_db[hash_val].append((song_id, offset))
                else:
                    fingerprint_db[hash_val] = [(song_id, offset)]

            song_id_map[song_id] = filename
            print(f"Indexed {filename}: {len(hashes)} hashes.")
            song_id += 1

    save_fingerprint_database(fingerprint_db, db_path)

    with open("song_id_map.pkl", "wb") as f:
        pickle.dump(song_id_map, f)
    print("Song ID map saved to song_id_map.pkl.")


if __name__ == "__main__":
    index_songs("../wav_folder")
