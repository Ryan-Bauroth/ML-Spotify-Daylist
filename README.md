# Spotify Machine Learning Project (name tbd)

## Description

The goal of this project is to:<br/>
a) **Gather data** on a user's Spotify listening automatically<br/>
b) **Create a prediction algorithm** based on this data to predict what music that user would want given the temperate, time of day, month, and day of the week<br/>
c) **Generate a playlist** based on these predictions and upload it to Spotify

## Installation & Use

For either option below, the first step is to clone the repository by copying the HTTPS link into the IDE of your choice

Then, create a .env file with the following format:

`CLIENT_ID=YOUR_SPOTIFY_CLIENT_ID`<br/>
`CLIENT_SECRET=YOUR_SPOTIFY_CLIENT_SECRET`<br/>
`REDIRECT_URI=http://localhost:8080/callback`<br/>

`PLAYLIST_ID=A_SPOTIFY_PLAYLIST_ID`<br/>
*(if you don't have one, you can generate a new playlist by deleting this paramter)*<br/>

`LONGITUDE=YOUR_LONGITUDE`<br/>
`LATITUDE=YOUR_LATITUDE`

### How to Generate a Dataset

1) Run the record.py file. This will redirect you to localhost on your device. Allow my app to access your data.
2) Listen to music & it will automatically be stored in the data.csv file.

**Here is an example dataset graphed:**
![time:valence daily chart d3](https://github.com/user-attachments/assets/2fed2504-bdd2-4902-accb-e5821f035b47)
_X-axis is time (seconds passed since midnight). Y-axis is valance (positivity of the music. 1 is the most positive). Colors represent different days of the week._


### How to Run the Prediction Algorithm

1) Run the most recent song prediction Jupiter Notebook to generate your playlist.


## Functionality

### Generating a Dataset

The record.py file is in charge of generating a dataset. This file uses the Spotipy library to connect to the Spotify API and get your current playback state. It then compiles this data, in addition to your current weather data and date information and uploads it to the dataset. This functionality is run every ~15 seconds.

### Cleaning Data

While the data retreived from Spotify is mostly in a good format, the genre names are rather inconsistent. Unforunately, as songs on Spotify's API don't have their own genre, genre data is retrieved from each artist instead. This means that each song has many different genres that may even conflict. Furthermore, the genres listed under each artist are very specific. For example, instead of the genre 'afrobeats', an artists genre description might include 'psychedelic funk afrobeat'. In my limited listening history, there might only be one song listed as this. This means that predicting using these specific genre names is not very helpful. Instead, I transform these genres into more broad categories using the *get_clustering_genres()* function. These are then transformed into an array of ints (whether each genre exists on a song or not). This is then used to train one my models.
