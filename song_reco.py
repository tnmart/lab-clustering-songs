#prerequisites
import function
import config
import requests
import pandas as pd
import numpy as np
import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
import pickle

#source dataset
hot_or_not_database = pd.read_csv('hot_or_not.csv') 
hot_or_not_scaled_df = pd.read_csv('hot_or_not_scaled_df.csv')


def music_recommendation_engine():
    while True:
        # User inputs
        U_song_input = input('Enter a song name:')
        U_artist_input = input('Enter an artist name:')

        U_song_id = function.user_search_song(U_song_input, U_artist_input, limit=1)
        U_audio_features = function.get_audio_features(U_song_id['id'])

        U_id_and_audio = function.add_audio_features(U_song_id, U_audio_features)
        U_id_and_audio['hot_or_not'] = 'U'

        columns_to_drop2 = ["type", "uri", "track_href", "analysis_url", "duration_ms"]

        U_clean_query = U_id_and_audio.drop(columns=columns_to_drop2)

        # Dropping the ID column to avoid confusion for the scaling & clustering
        U_numerical_only = U_clean_query.drop(columns=["id", "hot_or_not", "track_name", "artists"])

        # Load the saved scaler from the pickle file
        filename = "hot_or_not_scaler.pickle"  # Path to the scaler pickle file
        with open(filename, "rb") as file:
            scaler = pickle.load(file)

        # Apply the loaded scaler to U_numerical_clean
        U_numerical_scaled = scaler.transform(U_numerical_only)

        # Convert the scaled data back to a DataFrame
        U_num_scaled_df = pd.DataFrame(U_numerical_scaled, columns=U_numerical_only.columns)

        # Check spatial information to assign a cluster
        d = distance_matrix(U_num_scaled_df, hot_or_not_scaled_df)
        closest_song_to_user_song = np.argmin(d)

        # Store cluster in U_cluster
        U_cluster = hot_or_not_database.loc[closest_song_to_user_song, "cluster"]

        # Re-insert cluster found by model into user query with href, id, title, etc...
        U_id_and_audio['cluster'] = U_cluster

        # Recommend songs and print the recommendations
        function.recommend_songs(U_id_and_audio, hot_or_not_database)

        # Ask the user if they want to redo a recommendation or stop
        user_decision = input("\nDo you want to get another recommendation? (yes/no): ").lower()
        if user_decision != 'yes':
            print("\nWe hope you are satisfied with our recommender. Come again soon!")
            break


