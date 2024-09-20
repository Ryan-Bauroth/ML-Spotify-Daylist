import os
import spotipy
from dotenv import load_dotenv
from spotipy import SpotifyOAuth

load_dotenv()

clientID = os.getenv('CLIENT_ID')
clientSecret = os.getenv('CLIENT_SECRET')
redirect_uri = os.getenv('REDIRECT_URI')

scope = "user-library-read playlist-read-private user-read-playback-state user-modify-playback-state user-read-currently-playing playlist-modify-public playlist-modify-private"


# initialize spotipy
sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(client_id=clientID, client_secret=clientSecret, scope=scope,
                              redirect_uri=redirect_uri))



def update_playlist(songs, artists, playlist_id=None, song_ids=[]):
    if song_ids is None:
        song_ids = []
    id = sp.me()["id"]

    song_uris = []

    for song_idx in range(len(songs)):
        if len(song_ids) > 0 and str(song_ids[song_idx]) != "":
            song_data = [sp.track(str(song_ids[song_idx]))]
        else:
            song_data =  sp.search("track:\"" + songs[song_idx] + "\" artist:\"" + artists[songs.index(songs[song_idx])].split(".")[0] + "\"", 10, 0,
                      "track")["tracks"]["items"]
        if len(song_data) > 0:
            song_uris.append(song_data[0]["uri"])

    if not playlist_id:
        playlist_id = sp.user_playlist_create(id, "ML_Daylist", False, False, "crazy")["id"]
        print(playlist_id)
    else:
        songs_to_replace = []
        items = sp.playlist_items(playlist_id)['items']
        for item in items:
            songs_to_replace.append(item['track']['uri'])
        if len(songs_to_replace) > 0:
            sp.playlist_remove_all_occurrences_of_items(playlist_id, songs_to_replace)

    sp.playlist_add_items(playlist_id, song_uris, position=0)

def get_recs(songname, artists):
    song_uris = []
    song_data = sp.search("track:\"" + songname + "\" artist:\"" + artists + "\"", 10, 0,"track")["tracks"]["items"]
    if len(song_data) > 0:
        song_uris.append(song_data[0]["uri"])
    recs = sp.recommendations(
        seed_tracks=song_uris,
        limit=50,
        market="US",
    )
    return recs["tracks"]

def get_genres():
    return sp.recommendation_genre_seeds()

if __name__ == "__main__":
    print(sp.recommendation_genre_seeds()['genres'])
    # song_uris = ['spotify:track:6gQrm0rwg6hok8IxzysD8m', 'spotify:track:4f8Mh5wuWHOsfXtzjrJB3t',
    #  'spotify:track:36FZL9SzRh5BhtG1cUyGWr', 'spotify:track:3KZ5nrQ9jzquvFl5c9c45d',
    #  'spotify:track:1YZmYp9jZ4Veq6lZpiuV69', 'spotify:track:7ARveOiD31w2Nq0n5FsSf8',
    #  'spotify:track:6VLjdregMReJhKp2r32IWo', 'spotify:track:7ByxizhA4GgEf7Sxomxhze',
    #  'spotify:track:44U5sw3AvxKI0Sy0tYakll', 'spotify:track:4KVSdwwJ67JHu5s9vIA0zi',
    #  'spotify:track:0PLhwCmQ7cC3ThRGPn3HxF', 'spotify:track:7uGYWMwRy24dm7RUDDhUlD',
    #  'spotify:track:5LyKocU0lhUBlXrFKDxbBO', 'spotify:track:4mTtWe59jnFfM949udKXuE',
    #  'spotify:track:2k3IJR9hf34ZfEnTdlcoSK', 'spotify:track:52NGJPcLUzQq5w7uv4e5gf',
    #  'spotify:track:1PxKsGzQcmiwDHvA9ig5gv', 'spotify:track:58k32my5lKofeZRtIvBDg9']
    #
    # playlist_id = '3NKOueAcxMhburUy6pxjnX'
    # songs_to_replace = []
    # items = sp.playlist_items(playlist_id)['items']
    # for item in items:
    #     songs_to_replace.append(item['track']['uri'])
    # print(songs_to_replace)
    # sp.playlist_remove_all_occurrences_of_items(playlist_id, songs_to_replace)
    # sp.playlist_replace_items(playlist_id, song_uris)
