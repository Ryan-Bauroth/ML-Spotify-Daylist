import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from src.python_files.spotify_helper import get_genres

df = pd.read_csv('../data.csv')

median = df['popularity'].median()
df['popularity'] = df['popularity'].fillna(median)
df['genres'] = df['genres'].fillna('')

def cluster_songs_by_genre(df):
    # genres = get_genres()['genres']
    genres = get_clustering_genres()

    genre_song_dict = {}

    for song_idx in range(len(df['songname'])):
        genre_dict = {}
        song_genres = str(df['genres'][song_idx]).strip().split('.')
        for song_genre in song_genres:
            for genre in genres:
                if genre in song_genre:
                    if genre_dict.get(genre) is None:
                        genre_dict[genre] = 1
        for saved_genre in genre_dict:
            if genre_song_dict.get(saved_genre) is None:
                genre_song_dict[saved_genre] = []
            genre_song_dict[saved_genre].append(song_idx)
    return genre_song_dict

def get_avg_distance(genre, cleaned_np, genre_song_dict, weight=None):
    """
    Calculate the average song difference for a given genre.

    :param genre: The genre of the songs.
    :param cleaned_np: The cleaned numpy array.
    :return: The average song difference if there are multiple songs in the genre, else None.
    """
    average_song_diff = 0
    if len(genre_song_dict[genre]) < 2:
        return None
    runs = 0
    for idx in range(len(genre_song_dict[genre])):
        genre_one = genre_song_dict[genre][idx]
        for other_idx in range(idx, len(genre_song_dict[genre])):
            genre_two = genre_song_dict[genre][other_idx]
            song_1 = df['songname'][genre_one]
            song_2 = df['songname'][genre_two]
            genre_one_vals = np.copy(cleaned_np[genre_one])
            genre_two_vals = np.copy(cleaned_np[genre_two])

            # Skip if song names are identical
            if song_1 == song_2:
                continue

            # Vectorized Euclidean distance calculation
            if weight is not None:
                for col in range(len(weight)):
                    genre_one_vals[col] *= weight[col]
                    genre_two_vals[col] *= weight[col]

            idx_diff = np.linalg.norm(genre_one_vals - genre_two_vals)
            runs += 1
            average_song_diff += idx_diff
    if runs != 0:
        average_song_diff /= runs
    else:
        return None

    return average_song_diff


def weight_genres(genre_song_dict, cleaned_np, cleaned_df):

    genre_weight_dict = {}

    for genre in genre_song_dict:
        const_score = get_avg_distance(genre, cleaned_np, genre_song_dict)

        if const_score is None:
            continue

        col_scores = []

        for col in cleaned_df:
            new_np_arr = np.delete(cleaned_np, col, axis=1)
            col_scores.append(const_score - get_avg_distance(genre, new_np_arr, genre_song_dict))

        total_impact = np.sum(col_scores)
        feature_importance_scores = (col_scores / total_impact)

        # Calculate opposite scores
        max_possible_score = 1
        opposite_scores = max_possible_score - feature_importance_scores

        # Normalize opposite scores to ensure they sum to 1
        total_opposite_impact = np.sum(opposite_scores)
        genre_weight_dict[genre] = opposite_scores / total_opposite_impact

    return genre_weight_dict

def get_clustering_genres():
    dataset = pd.read_csv('../dataset.csv')

    genre_name_arr = get_genres()['genres']

    for genre_name in dataset['genres']:
        if genre_name not in genre_name_arr:
            genre_name_arr.append(genre_name)

    return genre_name_arr

if __name__ == "__main__":
    genre_song_dict = cluster_songs_by_genre(df)
    print("done")

    cleaned_df = df.drop(
        ['songname', 'artist', 'id', 'genres', 'temp', 'time', 'dayofweek', 'month', 'duration_ms', 'popularity'],
        axis=1)

    # Assuming cleaned_df is your DataFrame
    cleaned_df = MinMaxScaler().fit_transform(cleaned_df)

    cleaned_df = pd.DataFrame(cleaned_df)

    # Convert cleaned_df to NumPy array for faster operations
    cleaned_np = np.array(cleaned_df)

    genre_weights = weight_genres(genre_song_dict, cleaned_np, cleaned_df)
    print(genre_weights)



# print({key: [round(v,10) for v in val] for key, val in genre_weight_dict.items()})
