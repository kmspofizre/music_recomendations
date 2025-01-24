import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import pickle
from sklearn.metrics import roc_curve, auc

import re


def clean_name(name):
    name = name.lower()
    name = re.sub(r'[_\-]', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def convert_pt_to_npy(vectors_dir, expected_size=32):
    pt_files = [f for f in os.listdir(vectors_dir) if f.endswith('.pt')]

    if not pt_files:
        print("В директории нет .pt файлов для преобразования.")
        return

    for file in pt_files:
        pt_path = os.path.join(vectors_dir, file)
        npy_filename = os.path.splitext(file)[0] + '.npy'
        npy_path = os.path.join(vectors_dir, npy_filename)

        try:
            tensor = torch.load(pt_path)
            if tensor.dim() == 1 and tensor.size(0) == expected_size:
                array = tensor.detach().numpy()
                np.save(npy_path, array)
                print(f"Преобразовано: {file} -> {npy_filename}")
            else:
                print(f"Файл {file} имеет неподдерживаемую форму: {tensor.shape}. Ожидалось [{expected_size}].")
        except Exception as e:
            print(f"Ошибка при обработке файла {file}: {e}")


vectors_dir = "vectors32/"


convert_pt_to_npy(vectors_dir)


def extract_song_info(filename):
    name = os.path.splitext(filename)[0]
    delimiter = " - "

    if delimiter in name:
        artist, title = name.split(delimiter, 1)
        return title.strip(), artist.strip()
    else:
        return name.strip(), "Unknown"

npy_files = [f for f in os.listdir(vectors_dir) if f.endswith('.npy')]
data = []

for file in npy_files:
    song_id = os.path.splitext(file)[0]
    song_path = os.path.join(vectors_dir, file)
    vector = np.load(song_path).astype('float32')
    title, artist = extract_song_info(file)
    data.append({
        'song_id': song_id,
        'title': title,
        'artist': artist,
        'vector': vector
    })

print(f"Загружено {len(data)} песен.")


metadata = pd.DataFrame(data)
print(metadata.head())


with open('song_id_map.pkl', 'rb') as f:
    song_id_map = pickle.load(f)


print("Первые несколько записей song_id_map:")
for key in list(song_id_map.keys())[:3]:
    print(f"Song ID: {key}")
    print(f"Информация: {song_id_map[key]}")
    print("-" * 40)


song_id_df = pd.DataFrame.from_dict(song_id_map, orient='index').reset_index().rename(columns={'index': 'song_id_numeric'})
print("Колонки song_id_df:")
print(song_id_df.columns.tolist())

print(song_id_df.head())


required_columns = ['Name', 'Artist', 'Genre']
for col in required_columns:
    if col not in song_id_df.columns:
        print(f"Ошибка: Колонка '{col}' отсутствует в song_id_df.")
        import sys
        sys.exit("Прекращаем выполнение из-за отсутствия необходимых колонок.")

song_id_df['Name_no_ext'] = song_id_df['Name'].str.replace('.wav', '', regex=False)

# Стандартизация имен
song_id_df['Name_clean'] = song_id_df['Name_no_ext'].apply(clean_name)
metadata['song_id_clean'] = metadata['song_id'].apply(clean_name)

# Объединение на основе очищенных имен
metadata = metadata.merge(song_id_df, left_on='song_id_clean', right_on='Name_clean', how='left')

# Проверка наличия пропущенных значений в 'Genre'
missing_genre = metadata['Genre'].isnull().sum()


X = np.vstack(metadata['vector'].values)
print(f"Форма матрицы векторов: {X.shape}")
n_neighbors = 5
metric = 'cosine'
knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm='brute')
knn.fit(X)
print("Модель k-NN обучена.")


joblib.dump(knn, "knn_model32cp.joblib")
print("Модель k-NN сохранена в 'knn_model32cp.joblib'.")


def find_similar_songs_with_softmax(song_id, knn_model, metadata, top_n=20):
    if song_id not in metadata['song_id'].values:
        print(f"Песня с song_id '{song_id}' не найдена.")
        return []

    idx = metadata.index[metadata['song_id'] == song_id][0]

    song_vector = X[idx].reshape(1, -1)

    distances, indices = knn_model.kneighbors(song_vector, n_neighbors=top_n + 1)

    similarities = 1 - distances[0]

    probabilities = softmax(similarities)

    input_genre = metadata.loc[idx, 'Genre']
    input_artist = metadata.loc[idx, 'artist']

    similar_songs = []

    for i, index in enumerate(indices[0]):
        similar_song_id = metadata.iloc[index]['song_id']
        if similar_song_id == song_id:
            continue

        title = metadata.iloc[index]['title']
        artist = metadata.iloc[index]['artist']
        similarity = probabilities[i]

        similar_genre = metadata.iloc[index]['Genre']
        similar_artist = metadata.iloc[index]['artist']
        is_correct = int((similar_genre == input_genre) or (similar_artist == input_artist))

        similar_songs.append((title, artist, similarity, is_correct))

    return similar_songs[:top_n]


query_song_id = "Maks_Korzh_-_Gory_po_koleno"

similar = find_similar_songs_with_softmax(query_song_id, knn, metadata, top_n=5)

print(f"\nПохожие песни для '{query_song_id}':")
for i, (sim_title, sim_artist, similarity, is_correct) in enumerate(similar, 1):
    status = "Подходит" if is_correct else "Не подходит"
    print(f"{i}. {sim_title} (Сходство: {similarity:.4f})")


print(f"Количество строк в metadata: {metadata.shape[0]}")
print(f"Размерность X_pca: {X_pca.shape[0]}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

metadata['pca_x'] = X_pca[:, 0]
metadata['pca_y'] = X_pca[:, 1]

plt.figure(figsize=(12, 10))
plt.scatter(metadata['pca_x'], metadata['pca_y'], alpha=0.7)

for i, row in metadata.iterrows():
    plt.annotate(row['title'], (row['pca_x'], row['pca_y']), fontsize=8)

plt.title("PCA Снижение Размерности Векторов Эмбеддингов Песен")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

y_true = []
y_scores = []

for idx, row in metadata.iterrows():
    print(f"{idx}. Обработка песни {row['song_id']} - {row['title']} от {row['artist']}")
    input_song_id = row['song_id']
    similar_songs = find_similar_songs_with_softmax(input_song_id, knn, metadata, top_n=20)

    for sim_title, sim_artist, similarity, is_correct in similar_songs:
        y_true.append(is_correct)
        y_scores.append(similarity)

print(f"Количество пар для ROC: {len(y_true)}")


fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Доля ложных положительных (False Positive Rate)')
plt.ylabel('Доля истинных положительных (True Positive Rate)')
plt.title('ROC-кривая для модели поиска похожих песен')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"ROC AUC: {roc_auc:.2f}")

# Расчет точности
accuracy = sum(y_true) / len(y_true) if len(y_true) > 0 else 0
print(f"Точность модели: {accuracy*100:.2f}%")
