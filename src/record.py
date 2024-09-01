import asyncio
import os
import time
from datetime import datetime
import openmeteo_requests
import requests_cache
import spotipy
from dotenv import load_dotenv
from retry_requests import retry
from spotipy import SpotifyOAuth

load_dotenv()

scope = "user-library-read playlist-read-private user-read-playback-state user-modify-playback-state user-read-currently-playing"
clientID = os.getenv('CLIENT_ID')
clientSecret = os.getenv('CLIENT_SECRET')
redirect_uri = os.getenv('REDIRECT_URI')

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(client_id=clientID, client_secret=clientSecret, scope=scope,
                              redirect_uri=redirect_uri))


async def get_weather_info():
    """
    This function retrieves the current weather temperature in Fahrenheit for a given latitude and longitude using the Open-Meteo API.

    :return: The current temperature in Fahrenheit.
    """
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 35.797459,
        "longitude": -78.921082,
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
    return sp.current_playback(market=None, additional_types="track")


def get_spotify_audio_features(playback):
    if playback:
        return sp.audio_features([playback["item"]["id"]])

def get_spotify_genre(playback):
    genre_arr = []
    for artist in playback["item"]["artists"]:
        artist_id = artist["id"]
        artist_genres = sp.artist(artist_id)["genres"]
        for genre in artist_genres:
            if genre not in genre_arr:
                genre_arr.append(genre.replace(".", ""))
    return genre_arr

def get_spotify_popularity(playback):
    return playback["item"]["popularity"]


def get_hour_info():
    now = datetime.now()
    arr = now.strftime("%H:%M:%S").split(":")
    return str(int(arr[2]) + int(arr[1]) * 60 + int(arr[0]) * 3600)

def get_weekday_info():
    return datetime.today().weekday()

def get_month_info():
    return datetime.today().month

def main():
    playback = get_spotify_playing()
    f = open('data.csv', 'a+')
    f.seek(0)
    lines = f.readlines()
    lastline = [""]
    if lines:
        lastline = lines[-1].strip().split(",")

    if playback and str(playback["item"]["name"].strip()).replace(",", "") not in lastline and playback["progress_ms"] >= 15000 and playback["currently_playing_type"] == "track":
        artists = ""
        for artist in playback["item"]["artists"]:
            artists += str(artist["name"]).replace(".", "") + "."
        artists = artists.rstrip(".")
        audio_features = get_spotify_audio_features(playback)
        genres = get_spotify_genre(playback)
        popularity = str(get_spotify_popularity(playback))
        day_of_week = str(get_weekday_info())
        month = str(get_month_info())
        temp = str(asyncio.run(get_weather_info()))
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
        f.close()

"""
Runs the main function every 10 seconds.
"""
while True:
    main()
    time.sleep(10)