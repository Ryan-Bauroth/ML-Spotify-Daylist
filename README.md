# Spotify Machine Learning Project (name tbd)

## Description

The goal of this project is to:<br/>
a) **Gather data** on a user's Spotify listening automatically<br/>
b) **Create a prediction algorithm** based on this data to predict what music that user would want given the temperate, time of day, month, and day of the week<br/>
c) **Generate a playlist** based on these predictions and upload it to Spotify

## Installation & Use

For either option below, the first step is to clone the repository by copying the HTTPS link into the IDE of your choice

Then, create a .env file with the following format:

CLIENT_ID=YOUR_SPOTIFY_CLIENT_ID
CLIENT_SECRET=YOUR_SPOTIFY_CLIENT_SECRET
REDIRECT_URI=http://localhost:8080/callback

PLAYLIST_ID=A_SPOTIFY_PLAYLIST_ID
(if you don't have one, you can generate a new playlist by deleting this paramter)

LONGITUDE=YOUR_LONGITUDE
LATITUDE=YOUR_LATITUDE

### How to Generate a Dataset

1) Run the record.py file. This will redirect you to localhost on your device. Allow my app to access your data.
2) Listen to music & it will automatically be stored in the data.csv file.

**Here is an example dataset graphed:**
![time:valence daily chart d3](https://github.com/user-attachments/assets/2fed2504-bdd2-4902-accb-e5821f035b47)
_X-axis is time (seconds passed since midnight). Y-axis is valance (positivity of the music. 1 is the most positive). Colors represent different days of the week._


### How to Run the Prediction Algorithm

1) Run the most recent song prediction Jupiter Notebook to generate your playlist.
