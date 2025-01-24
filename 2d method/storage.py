# storage.py

import pickle
import os

def save_fingerprint_database(fingerprint_db, filepath="fingerprint_db.pkl"):
    with open(filepath, 'wb') as f:
        pickle.dump(fingerprint_db, f)
    print(f"Fingerprint database saved to {filepath}.")

def load_fingerprint_database(filepath="fingerprint_db.pkl"):
    if not os.path.exists(filepath):
        print(f"Fingerprint database file {filepath} does not exist.")
        return {}
    with open(filepath, 'rb') as f:
        fingerprint_db = pickle.load(f)
    print(f"Fingerprint database loaded from {filepath}.")
    return fingerprint_db
