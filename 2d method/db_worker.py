import pickle


def print_song_id_map(filepath="song_id_map.pkl"):
    try:
        with open(filepath, 'rb') as f:
            song_id_map = pickle.load(f)
        print(f"Loaded song ID map from {filepath}.")
        for song_id, metadata in song_id_map.items():
            print(f"Song ID: {song_id}")
            print(f"  Name: {metadata.get('Name', 'Unknown')}")
            print(f"  Artist: {metadata.get('Artist', 'Unknown')}")
            print(f"  Genre: {metadata.get('Genre', 'Unknown')}")
    except FileNotFoundError:
        print(f"File {filepath} not found.")
    except Exception as e:
        print(f"Error loading song ID map: {e}")


print_song_id_map()
