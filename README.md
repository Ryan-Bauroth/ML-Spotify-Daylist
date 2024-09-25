# ML Daylist

[![Screenshot_2024-09-24_at_9 38 36_PM-removebg-3](https://github.com/user-attachments/assets/53e81554-9794-42ba-bb7f-5f3fb850d2f3)](https://open.spotify.com/playlist/3NKOueAcxMhburUy6pxjnX?si=dd6b36f7df8f466b)

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
*(if you don't have one, you can generate a new playlist by deleting this parameter)*<br/>

`LONGITUDE=YOUR_LONGITUDE`<br/>
`LATITUDE=YOUR_LATITUDE`

### How to Generate a Dataset

1) Run the record.py file. This will redirect you to localhost on your device. Allow my app to access your data.
2) Listen to music & it will automatically be stored in the data.csv file.

**Here is an example dataset graphed:**
![time:valence daily chart d3](https://github.com/user-attachments/assets/2fed2504-bdd2-4902-accb-e5821f035b47)
_X-axis is time (seconds passed since midnight). Y-axis is valence (positivity of the music. 1 is the most positive). Colors represent different days of the week._


### How to Run the Prediction Algorithm

1) Run the most recent song prediction Jupyter Notebook to generate your playlist.


## Functionality

### Generating a Dataset

The record.py file is in charge of generating a dataset. This file uses the Spotipy library to connect to the Spotify API and get your current playback state. It then compiles this data, in addition to your current weather data and date information and uploads it to the dataset. This functionality is run every ~15 seconds.

### Cleaning Data

While the data retrieved from Spotify is mostly in a good format, the genre names are rather inconsistent. Unfortunately, as songs on Spotify's API don't have their own genre, genre data is retrieved from each artist instead. This means that each song has many different genres that may even conflict. Furthermore, the genres listed under each artist are very specific. For example, instead of the genre 'afrobeats', an artist's genre description might include 'psychedelic funk afrobeat'. In my limited listening history, there might only be one song listed as this. This means that predicting using these specific genre names is not very helpful. Instead, I transform these genres into more broad categories using the *get_clustering_genres()* function. 

### Training Models and Predicting

This prediction algorithm primarily uses two models to predict songs:

1) First, I train a random forest classifier model which allows me to predict a list of genres based on my features (which are time of day, day of week, day of month, and temperature).<br/>

`model = train_random_forest_classifier(features, classifier_target)[0]`<br/>

2) I then use this prediction to create a list of genre probabilities, which I add to my previous features in order to help predict songs to add to a playlist. The predictions are done separately in an effort to make song predictions more clustered in a way that makes sense rather than being random songs that all match a similar vibe coincidentally.<br/>

`features = pd.concat([features, classifier_target], axis=1)`<br/>

3) For the song predictions, I utilize a gradient boosted regression model because of its ability to take each training feature individually. A lot of models I found other than this one would average all the seperate features together, which was not something I wanted as each feature represents a different aspect of the song (such as valence/positivity of the song or energy). These attributes are the same attributes that I gathered from Spotify's API as part of the data collection.<br/>

`multi_output_gbr = train_gradient_boosting_regressor(features, targets)[0]`<br/>

4) The algorithm then compares each feature of each song in the data collection and adds the squared absolute value of the differences together to calculate an accuracy score. This accuracy score is then multiplied by certain weights (which are calculated in a separate process detailed later). The lower the accuracy score, the better the song fits the current prediction.<br/>

`accuracy_score += abs(y_pred[0][col_idx] - data[predicted_cols[col_idx]][i]) * prediction_weighting[col_idx]`

### Generating Genre Weights

In an effort to further cluster songs by genre, an algorithm calculates weights to multiply with predicted accuracy scores. This is done as some features are more important for some genres rather than others (for example, the acousticness feature might be really important for country music, but less so for pop music). This process, detailed below, is found in the manual_clustering.py file.

1) To organize clusters of genres, the *cluster_songs_by_genre()* function is run.

2) Using the organized genres, a separate function called *get_avg_distance()* calculates the average distance between each multidimensional array of Spotify features

3) Finally, the weights are calculated in the aptly named *weight_genres()* function. This works by one by one removing each feature from a specific genre and running the *get_avg_distance()* function. This calculates a *silhouette score* (how much impact not having the feature has on the genre) for each feature. The higher the silhouette score, the more spread that feature is in the genre, and thus the less important.

### Spotify Use

Of course, this project wouldn't be complete without a function that automatically uploads the predicted playlist to Spotify. This can be found in the spotify_helper.py file. 

## Next Steps

Great! This project really works. I have found myself using the ML_Daylist (as I have named it) as a replacement for Spotify's Daylist fairly often—much more use than I would have expected my project to get when I started it. Still, there are plenty of ways I could work on improving this project. Here is a quick list:

- [ ] Implement a solution that predicts the genres of songs that aren't labeled by Spotify to improve prediction accuracy
- [ ] Adjust accuracy score calculation to take the direction of the variance into account
- [ ] Tune sklearn models to fit my data more accurately
- [ ] Add a new feature: Relative popularity (how much the specific user listens to this song)
- [ ] Add a way to weight more recent data as more important when training the models

...and the list goes on. This has been a ton of fun, but definitely a huge undertaking that could probably absorb endless effort and tuning.

## Resources

Here are some of the resources I used while creating this project

- ChatGPT and JetBrains AI were great for explaining errors and getting me started with Machine Learning (as this is my first Machine Learning/AI project)
- [Spotipy](https://spotipy.readthedocs.io/en/2.24.0/) was the Library I used to do all my Spotify API calls
- I used [Open Metro](https://open-meteo.com) for all my weather related calls when collecting data
- I got some inspiration for my weighted genres concept from the person who made the actual Spotify recommendation algorithm on [this podcast](https://www.youtube.com/watch?v=Q8W2IGiSdhc)
