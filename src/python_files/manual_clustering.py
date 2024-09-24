import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from src.python_files.spotify_helper import get_genres

df = pd.read_csv('../data.csv')

median = df['popularity'].median()
df['popularity'] = df['popularity'].fillna(median)
df['genres'] = df['genres'].fillna('')

def cluster_songs_by_genre(df):
    """
    This function clusters songs by genre based on the provided dataframe.

    :param df: A pandas dataframe containing songs and their respective genres.
    :return: A dictionary containing the genres as keys and a list of song indices as values.
    """
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
    """
    Calculates the weight of genres based on their importance scores.

    :param genre_song_dict: A dictionary mapping genre names to song lists.
    :param cleaned_np: A numpy array containing the cleaned data.
    :param cleaned_df: A pandas dataframe containing the cleaned data.
    :return: A dictionary mapping genre names to their weight values.

    """
    genre_weight_dict = {}

    for genre in genre_song_dict:
        const_score = get_avg_distance(genre, cleaned_np, genre_song_dict)

        if const_score is None:
            col_count = len(cleaned_df.T[0])
            genre_weight_dict[genre] = np.full(col_count, 1/col_count)
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
    """
    Returns a list of genres found in the dataset.

    :return: A list of genres.
    """
    dataset = pd.read_csv('../dataset.csv')
    other_dataset = pd.read_csv('../spotify_songs.csv')

    genre_name_arr = get_genres()['genres']

    for genre_name in dataset['genres']:
        if genre_name not in genre_name_arr:
            genre_name_arr.append(genre_name)

    for genre_name in other_dataset['genres']:
        if genre_name not in genre_name_arr:
            genre_name_arr.append(genre_name)

    return genre_name_arr

def sort_song_without_genre(genre_song_dict, cleaned_np, cleaned_df, genre_weights, song_idx):
    """
    Sorts songs without a specified genre based on their similarity to genres using a weighted distance metric.

    :param genre_song_dict: A dictionary mapping genres to a list of song indices belonging to that genre.
    :param cleaned_np: A numpy array representing the cleaned song data.
    :param cleaned_df: A pandas DataFrame representing the cleaned song data.
    :param genre_weights: A dictionary mapping genres to their respective weights.
    :param song_idx: The index of the input song to compare with genres.

    :return: A list of up to three genres that are most similar to the input song.
    """
    genre_song_data_dict = {}

    for genre in genre_song_dict:
        genre_songs = np.zeros((len(genre_song_dict[genre]), cleaned_np.shape[1]))
        for i, song in enumerate(genre_song_dict[genre]):
            genre_songs[i, :] = cleaned_np[song] * genre_weights[genre]

        genre_song_data_dict[genre] = genre_songs

    genre_diff_dict = {}

    for genre in genre_song_dict:
        genre_medians = np.median(genre_song_data_dict[genre], axis=0)
        input_song_weighted = cleaned_df.T[song_idx] * genre_weights[genre]
        genre_diff_dict[genre] = np.linalg.norm(genre_medians - input_song_weighted)

    genre_diff_dict = {genre: distance for genre, distance in genre_diff_dict.items() if distance != 0}
    max_distance = max(genre_diff_dict.values())
    normalized_distances = {genre: max_distance - distance for genre, distance in genre_diff_dict.items()}

    sorted_vals = sorted(normalized_distances.items(), key=lambda item: item[1], reverse=True)[:3]

    return [genre for genre, _ in sorted_vals]

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

    print(sort_song_without_genre(genre_song_dict, cleaned_np, cleaned_df, genre_weights, 5))



    # print({key: [round(v,10) for v in val] for key, val in genre_weight_dict.items()})
