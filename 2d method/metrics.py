import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.special import softmax
from collections import defaultdict


from recognizer import load_fingerprint_database, load_song_id_map, recognize_song


def getprob(arr: np.ndarray):
  s = np.sum(arr)
  return arr/s


def evaluate_roc_with_softmax(test_songs, db_path, song_id_map_path):
    fingerprint_db = load_fingerprint_database(db_path)
    song_id_map = load_song_id_map(song_id_map_path)
    all_y_true = []
    all_scores = []

    for item in test_songs:
        query_filepath = item["query_filepath"]
        true_song_id = item["true_song_id"]

        true_song_name = song_id_map[true_song_id]["Name"]

        results = recognize_song(query_filepath, db_path, song_id_map_path)

        if not results:

            all_y_true.append(0)
            all_scores.append(0.0)
            continue

        top5 = results[:5]
        similarities = [sim for (_, sim) in top5]

        similarities_arr = np.array(similarities, dtype=float)

        print(f"Similarities: {similarities_arr}")
        prob = getprob(similarities_arr)

        p_top = prob[0]
        top_song_name = top5[0][0]
        y_true = 1 if (top_song_name == true_song_name) else 0

        all_y_true.append(y_true)
        all_scores.append(p_top)

    all_y_true = np.array(all_y_true)
    all_scores = np.array(all_scores)
    print(all_y_true)
    print(all_scores)
    if len(set(all_y_true)) < 2:
        roc_auc = 0.5
        print("Warning: all_y_true has only one class => returning 0.5 as neutral.")
    else:
        roc_auc = roc_auc_score(all_y_true, all_scores)

    print(f"Global ROC AUC (top-1 with softmax) = {roc_auc:.3f}")

    fpr, tpr, thresholds = roc_curve(all_y_true, all_scores)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}", color="blue")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Top-1 predictions (softmax probability)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return roc_auc


if __name__ == "__main__":
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
        {
            "query_filepath": "../wav_folder/Fort_Minor_feat_Styles_Of_Beyond_Bobo_-_Believe_Me_feat_Bobo_Styles_Of_Beyond.wav",
            "true_song_id": 11},
        {"query_filepath": "../wav_folder/FRIENDLY_THUG_52_NGG_Big_Baby_Tape_-_Legit_Check.wav", "true_song_id": 12},
        {"query_filepath": "../wav_folder/Linkin_Park_-_Bleed_It_Out.wav", "true_song_id": 13},
        {"query_filepath": "../wav_folder/Linkin_Park_-_Breaking_The_Habit.wav", "true_song_id": 14},
        {"query_filepath": "../wav_folder/Linkin_Park_-_Faint.wav", "true_song_id": 15},
        {"query_filepath": "../wav_folder/Linkin_Park_-_In_The_End.wav", "true_song_id": 16},
        {"query_filepath": "../wav_folder/Linkin_Park_-_Numb.wav", "true_song_id": 17},
        {"query_filepath": "../wav_folder/Linkin_Park_-_One_More_Light.wav", "true_song_id": 18},
        {"query_filepath": "../wav_folder/Linkin_Park_-_What_Ive_Done.wav", "true_song_id": 19},
        {"query_filepath": "../wav_folder/Loc-Dog_-_V_tojj_vesne.wav", "true_song_id": 20},

    ]

    db_path = "fingerprint_db.pkl"
    song_id_map_path = "song_id_map.pkl"

    evaluate_roc_auc_global = evaluate_roc_with_softmax(test_songs, db_path, song_id_map_path)
