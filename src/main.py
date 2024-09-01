import os
import spotipy
from spotipy import SpotifyOAuth
from dotenv import load_dotenv
from pprint import pprint


load_dotenv()

scope = "user-library-read playlist-read-private"
clientID = os.getenv('CLIENT_ID')
clientSecret = os.getenv('CLIENT_SECRET')
redirect_uri = os.getenv('REDIRECT_URI')

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=clientID, client_secret=clientSecret, scope=scope, redirect_uri=redirect_uri))
 
playlists = sp.current_user_playlists()


"""
    Get all tracks in a playlist.
    
    Written by AI

    :param playlist_id: The ID of the playlist.
    :return: A list of tracks in the playlist.
"""
def get_playlist_tracks(playlist_id):
    results = sp.playlist_items(playlist_id, additional_types="track")
    tracks = results['items']
    # Loop through the next pages (if any)
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

for playlist in playlists['items']:
    print(f"Playlist: {playlist['name']}")
    playlist_id = playlist['id']
    tracks = get_playlist_tracks(playlist_id)
    for idx, item in enumerate(tracks):
        track = item['track']
        if track and 'name' in track and 'artists' in track and track['artists']:
            print(f"{idx + 1}. {track['name']} by {track['artists'][0]['name']}")
        else:
            print(f"{idx + 1}. Track information is incomplete and cannot be displayed.")