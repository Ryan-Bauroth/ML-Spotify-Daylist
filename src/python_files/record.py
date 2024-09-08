import os
from dotenv import load_dotenv
import time
from datetime import datetime
import openmeteo_requests
import asyncio
import requests_cache
from retry_requests import retry
import spotipy
from spotipy import SpotifyOAuth

# get environment variables from .env file & set scopes const
load_dotenv()
clientID = os.getenv('CLIENT_ID')
clientSecret = os.getenv('CLIENT_SECRET')
redirect_uri = os.getenv('REDIRECT_URI')
longitude = os.getenv('LONGITUDE')
latitude = os.getenv('LATITUDE')
scope = "user-library-read playlist-read-private user-read-playback-state user-modify-playback-state user-read-currently-playing"


# initialize spotipy
sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(client_id=clientID, client_secret=clientSecret, scope=scope,
                              redirect_uri=redirect_uri))

async def get_weather_info():
    """
    This function retrieves the current weather temperature in Fahrenheit for a given latitude and longitude using the Open-Meteo API.

    :return: The current temperature in Fahrenheit.
    """
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('../.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "models": "gfs_seamless"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Current values. The order of variables needs to be the same as requested.
    current = response.Current()
    current_temperature_2m = current.Variables(0).Value()

    return current_temperature_2m


def get_spotify_playing():
    """
    This function is used to retrieve information about the current playback of a user

    :return: The current playback information, or None if an error occurs.
    """
    try:
        return sp.current_playback(market=None, additional_types="track")
    except spotipy.exceptions.SpotifyException as error:
        print("error getting playback spotify exception")
        print(error)
        return None
    except Exception as error:
        print("error getting playback not spotify exception")
        print(error)
        return None



def get_spotify_audio_features(playback):
    """
    :param playback: A dictionary object representing the current playback state in the Spotify API.
    :return: A list of audio features for the currently playing track.

    If the user is listening to music (thus playback exists), uses Spotipy call to get audio features
    """
    if playback:
        return sp.audio_features([playback["item"]["id"]])


def get_spotify_genre(playback):
    """
    :param playback: The playback information containing the currently playing track and artist(s)
    :return: A list of genres associated with the currently playing track and artist(s)

    Retrieves an array of genres to describe the currently playing track.
    """
    genre_arr = []
    for artist in playback["item"]["artists"]:
        artist_id = artist["id"]
        artist_genres = sp.artist(artist_id)["genres"]
        for genre in artist_genres:
            if genre not in genre_arr:
                genre_arr.append(genre.replace(".", ""))
    return genre_arr


def get_spotify_popularity(playback):
    """
    :param playback: The current playback context, including the currently playing track.
    :return: The popularity value of the current track.

    Retrieves the popularity of the currently playing track from the Spotify API.
    """
    return playback["item"]["popularity"]


def get_hour_info():
    """
    Calculates the number of seconds passed since midnight.

    :return: The number of seconds passed since midnight.
    """
    now = datetime.now()
    arr = now.strftime("%H:%M:%S").split(":")
    return str(int(arr[2]) + int(arr[1]) * 60 + int(arr[0]) * 3600)


def get_weekday_info():
    """
    Get the current weekday information.

    :return: An integer representing the current weekday. Monday is 0 and Sunday is 6.
    """
    return datetime.today().weekday()


def get_month_info():
    """
    Returns the current month.

    :return: Current month as an integer.
    """
    return datetime.today().month


def main():
    """
    :return: the amount of time until the main function should be run again

    Main function of record.py file
    """

    # sets how long until the main function should be run again
    r_time = 10

    # gets information about the users current playback
    playback = get_spotify_playing()

    # gets the most recent item in the datatable
    f = open('../data.csv', 'a+')
    f.seek(0)
    lines = f.readlines()
    lastline = [""]
    if lines:
        lastline = lines[-1].strip().split(",")

    # sets the minimum listening time before a song gets 'counted' (s) * (1000) = ms
    min_listen_time = 15 * 1000

    # this try catch is included just in case something goes wrong with spotify (likely a Spotify AI DJ issue)
    try:
        # only updates datatable if user is listening to music, the current song being played is not already stored in the
        # datatable, and the song has been playing for the minimum listening time (ms)
        if playback and str(playback["item"]["name"].strip()).replace(",", "") not in lastline and time.time() * 1000 - playback[
            "timestamp"] >= min_listen_time and playback["progress_ms"] >= min_listen_time and playback["currently_playing_type"] == "track":

            # gets datatable information and formats it as str
            artists = ""
            for artist in playback["item"]["artists"]:
                artists += str(artist["name"]).replace(".", "").replace(",","").strip() + "."
            artists = artists.rstrip(".")
            audio_features = get_spotify_audio_features(playback)
            genres = get_spotify_genre(playback)
            popularity = str(get_spotify_popularity(playback))
            day_of_week = str(get_weekday_info())
            month = str(get_month_info())
            temp = str(asyncio.run(get_weather_info()))

            """
            Song attributes (all stored as strings):
            
            - **Song Name**: The name of the song
            - **Artist Name(s)**: The artist(s) of the song
            - **Genre(s)**: The genre(s) of the artist(s)
            - **Popularity**: Rating (0-100) of the song's popularity
            - **Duration**: Duration of the track in milliseconds
            
            Spotify attributes (0.0 least / 1.0 most, unless noted):
            
            - **Danceability**: How suitable a track is for dancing (tempo, beat)
            - **Energy**: Intensity and activity (e.g. death metal high, Bach low)
            - **Loudness**: Overall loudness in dB (typically -60 to 0 dB)
            - **Speechiness**: Presence of spoken words (values >0.66 mostly spoken, 0.33-0.66 speech+music, <0.33 mostly music)
            - **Acousticness**: Confidence measure of a track being acoustic
            - **Instrumentalness**: No vocals probability (values >0.5 likely instrumental)
            - **Liveness**: Audience presence (values >0.8 likely live)
            - **Valence**: Musical positiveness (high positive, low negative)
            - **Tempo**: Beats per minute (BPM)
            
            Additional attributes:
            - **Temp**: Temperature in Fahrenheit
            - **hour_info**: Current hour in seconds
            - **day_of_week**: Day song played (0=Monday, 6=Sunday)
            - **month**: Month song played (1=January, 12=December)
            """

            data = [
                str(playback["item"]["name"]).replace(",", ""),
                artists,
                ".".join(genres),
                popularity,
                str(audio_features[0]["danceability"]),
                str(audio_features[0]["energy"]),
                str(audio_features[0]["loudness"]),
                str(audio_features[0]["speechiness"]),
                str(audio_features[0]["acousticness"]),
                str(audio_features[0]["instrumentalness"]),
                str(audio_features[0]["liveness"]),
                str(audio_features[0]["valence"]),
                str(audio_features[0]["tempo"]),
                str(audio_features[0]["duration_ms"]),
                temp,
                get_hour_info(),
                day_of_week,
                month,
            ]
            print(data)
            f.write(", ".join(data) + "\n")
        elif playback and time.time() * 1000 - playback["timestamp"] < 15000:
            # calculates the amount of time before the song passes the 15-second threshold
            # i have no idea why but every time it is 3 seconds too early so this is a quick fix
            r_time = (15000 - (time.time() * 1000 - playback["timestamp"]))/1000 + 3
    except TypeError as error:
        print(error)
    finally:
        # closes file and returns next wait time
        f.close()
        return r_time



if __name__ == "__main__":
    """
    Runs the Main function every ~10 seconds, adjusting its wait time based on how far the user is into the song.
    """
    while True:
        time.sleep(main())
