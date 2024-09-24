import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your data
df = pd.read_csv('../data.csv')

# Fill missing values with the median
median = df['popularity'].median()
df['popularity'] = df['popularity'].fillna(median)
df['genres'] = df['genres'].fillna('')

# Standardize relevant features
predicted_cols = ['popularity', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'valence',
                  'tempo']
df[predicted_cols] = StandardScaler().fit_transform(df[predicted_cols])

# Function to split genres and assign main genre
def get_main_genre(song_genre_str, genre_list):
    song_genres = str(song_genre_str).strip().split('.')
    genre_dict = {}
    for song_genre in song_genres:
        for genre in genre_list:
            if genre in song_genre:
                genre_dict[genre] = genre_dict.get(genre, 0) + 1
    if genre_dict:
        return max(genre_dict, key=genre_dict.get)  # Return most frequent genre
    return None


# Example genre list from the Spotify API (could be replaced with real genres)
genres = ['indie', 'rock', 'pop', 'hip hop', 'classical', 'jazz']

# Assign main genre to each song
df['main_genre'] = df['genres'].apply(lambda x: get_main_genre(x, genres))


# Create song pairs and labels for training
def create_pairs_and_labels(df, predicted_cols):
    genre_feature_data = {}
    genre_song_dict = {}

    # Group songs by their main genre
    for index, row in df.iterrows():
        genre = row['main_genre']
        if genre:
            if genre not in genre_song_dict:
                genre_song_dict[genre] = []
            genre_song_dict[genre].append(index)

    # Collect song features by genre
    for genre in genre_song_dict:
        genre_feature_data[genre] = df.loc[genre_song_dict[genre], predicted_cols].values.tolist()

    pairs = []
    labels = []

    # Create similar pairs (same genre, label = 1)
    for genre, genre_data in genre_feature_data.items():
        for i in range(len(genre_data)):
            for j in range(i + 1, len(genre_data)):
                pairs.append([genre_data[i], genre_data[j]])
                labels.append(1)

    # Create dissimilar pairs (different genre, label = 0)
    used_keys = list(genre_feature_data.keys())
    for idx, genre in enumerate(used_keys):
        genre_data = genre_feature_data[genre]
        for other_genre in used_keys[idx + 1:]:
            other_genre_data = genre_feature_data[other_genre]
            for i in range(len(genre_data)):
                for j in range(len(other_genre_data)):
                    pairs.append([genre_data[i], other_genre_data[j]])
                    labels.append(0)

    return np.array(pairs), np.array(labels)


# Get pairs and labels
pairs, labels = create_pairs_and_labels(df, predicted_cols)

# Split data into train and test
train_pairs, test_pairs, train_labels, test_labels = train_test_split(pairs, labels, test_size=0.2, random_state=42)


# Define the model class as a subclass of nn.Module
class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim):
        super(EmbeddingNetwork, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Instantiate the model
input_dim = len(predicted_cols)
model = EmbeddingNetwork(input_dim)


# Define a forward function for the Siamese Network
def siamese_network(model, pair1, pair2):
    embedding1 = model(pair1)
    embedding2 = model(pair2)
    return embedding1, embedding2


# Training loop
def train_siamese(model, train_pairs, train_labels, learning_rate=0.001, epochs=10, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()  # Binary classification (similar or dissimilar)

    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(0, len(train_pairs), batch_size):
            batch_pairs = train_pairs[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            # Convert list of arrays to numpy arrays first
            batch1 = np.array([pair[0] for pair in batch_pairs])
            batch2 = np.array([pair[1] for pair in batch_pairs])
            labels = torch.tensor(batch_labels, dtype=torch.float32)

            # Convert numpy arrays to tensors
            batch1 = torch.tensor(batch1, dtype=torch.float32)
            batch2 = torch.tensor(batch2, dtype=torch.float32)

            optimizer.zero_grad()

            # Forward pass
            output1, output2 = siamese_network(model, batch1, batch2)

            # Calculate distance between embeddings
            distance = torch.norm(output1 - output2, p=2, dim=1)

            # Calculate similarity score and loss
            similarity_score = torch.sigmoid(-distance)
            loss = criterion(similarity_score, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_pairs)}')


# Train the model
train_siamese(model, train_pairs, train_labels)

model.eval()

# Example data preparation for prediction
def prepare_data_for_prediction(df, predicted_cols):
    # Extract features and standardize them
    features = df[predicted_cols].values
    features = StandardScaler().fit_transform(features)
    return features

dbscan_features = prepare_data_for_prediction(df, predicted_cols)


def predict_similarity(model, pair1, pair2):
    model.eval()

    # Convert inputs to tensors
    pair1 = torch.tensor(pair1, dtype=torch.float32)
    pair2 = torch.tensor(pair2, dtype=torch.float32)

    # Forward pass through the model to get embeddings
    with torch.no_grad():
        embedding1 = model(pair1)
        embedding2 = model(pair2)

    # Ensure embeddings are 2-dimensional
    if len(embedding1.shape) == 1:
        embedding1 = embedding1.unsqueeze(0)
    if len(embedding2.shape) == 1:
        embedding2 = embedding2.unsqueeze(0)

    # Compute the Euclidean distance between embeddings
    distance = torch.norm(embedding1 - embedding2, p=2, dim=1)
    return distance.item()


def cluster_songs(features, model):
    n = len(features)
    clustering_labels = []

    # Create a pairwise distance matrix
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = predict_similarity(model, features[i], features[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric matrix

    # Example clustering: You can use DBSCAN or another clustering method
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=15, min_samples=2).fit(distance_matrix)
    clustering_labels = clustering.labels_

    return clustering_labels


clustering_labels = cluster_songs(dbscan_features, model)

dbscan_indices = df[df['genres'] == ''].index
df.loc[dbscan_indices, 'cluster_label'] = clustering_labels

print(df['songname'][df['cluster_label'] == 10])


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_clusters(features, labels):
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
    plt.title('Song Clusters')
    plt.show()

visualize_clusters(dbscan_features, clustering_labels)