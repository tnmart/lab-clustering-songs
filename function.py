import time
from IPython.display import display
import ipywidgets as widgets
import requests
import pickle
import pandas as pd
import numpy as np
import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials




def search_song(df, column_title, column_artist, limit=1):
    '''
    Takes a dataframe with two columns: track_name and artists
    Returns a DataFrame with three columns: track_name, artists, id
    '''
    # Initialize Spotipy
    client_credentials_manager = SpotifyClientCredentials(client_id='YOUR_CLIENT_ID', client_secret='YOUR_CLIENT_SECRET')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame(columns=['track_name', 'artists', 'id'])
    
    # Iterate through each row of the input DataFrame
    for index, row in df.iterrows():
        # Search for the track using the title and artists from the DataFrame
        track_name = row[column_title]
        artists = row[column_artist]
        query = f'track:"{track_name}" artist:"{artists}"'
        results = sp.search(q=query, limit=limit)
        
        # Extract the track ID(s) from the search results
        track_ids = [item['id'] for item in results['tracks']['items']]
        
        # If there are track IDs, append them to the result DataFrame
        if track_ids:
            for track_id in track_ids:
                result_df = pd.concat([result_df, pd.DataFrame({'track_name': [track_name], 'artists': [artists], 'id': [track_id]})], ignore_index=True)
    
    return result_df


def get_audio_features(list_of_song_ids):
    chunk_size = 50
    audio_features_list = []  # List to store audio features

    # Initialize Spotipy
    client_credentials_manager = SpotifyClientCredentials(client_id='e6694c5fb84f4d1f9ce67605909b7019', client_secret='219e8d198a414966babf64454a2348ba')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    progress_bar = widgets.IntProgress(
        min=0, 
        max=len(list_of_song_ids), 
        description='Processing audio features :', 
        bar_style='', 
        style={'bar_color': '#1ED760'})
    
    display(progress_bar)

    for i in range(0, len(list_of_song_ids), chunk_size):
        chunk = list_of_song_ids[i:i + chunk_size]

        try:
            # Retrieve audio features for the chunk of song IDs
            audio_features = sp.audio_features(chunk)
            audio_features_list.extend(audio_features)
        except Exception as e:
            print("Error retrieving audio features:", e)

        time.sleep(1)  # Sleep to avoid rate limiting
        progress_bar.value = i + chunk_size  # Update progress bar value

    # Create a DataFrame from the list of audio features
    df = pd.DataFrame(audio_features_list)

    return df



def add_audio_features(df, audio_features_df):
        # Merge the dataframes on the 'id' column
        merged_df = pd.merge(df, audio_features_df, left_on='id', right_on='id', how='left')
        return merged_df



def user_search_song(title, artist, limit=1):
    '''
    Takes a dataframe with two columns: title and artist in strings
    Returns a DataFrame with three columns: track_name, artists, id
    '''
    # Initialize Spotipy
    client_credentials_manager = SpotifyClientCredentials(client_id='e6694c5fb84f4d1f9ce67605909b7019', client_secret='219e8d198a414966babf64454a2348ba')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame(columns=['track_name', 'artists', 'id'])
    
    query = f'track:"{title}" artist:"{artist}"'
    results = sp.search(q=query, limit=limit)
        
    # Extract the track ID(s) from the search results
    track_ids = [item['id'] for item in results['tracks']['items']]
        
    # If there are track IDs, append them to the result DataFrame
    if track_ids:
        for track_id in track_ids:
            result_df = pd.concat([result_df, pd.DataFrame({'track_name': [title], 'artists': [artist], 'id': [track_id]})], ignore_index=True)
    
    return result_df

def recommend_songs(user_df, hot_or_not_df):
    # Check for matching IDs between user_df and hot_or_not_df
    matching_ids = user_df[user_df['id'].isin(hot_or_not_df['id'])]

    if matching_ids.empty:
        # If matching_ids is empty, fetch 5 songs with the same cluster as the first row in hot_or_not_df
        cluster_value = hot_or_not_df.iloc[0]['cluster'] if not hot_or_not_df.empty else None
        print("Well, this is a rare song. It's not in our database, but maybe you'll like these ones:")
        recommend_similar_songs(user_df, hot_or_not_df, cluster_value, num_recommendations=5)

    for _, row in matching_ids.iterrows():
        song_id = row['id']
        hot_or_not_value = hot_or_not_df[hot_or_not_df['id'] == song_id]['hot_or_not'].values[0]

        # Check hot_or_not value and print phrases accordingly
        if hot_or_not_value == 'H':
            print("Great taste! This song is part of the top 100 hot songs. Let's discover more!")
        elif hot_or_not_value == 'N':
            print("This is not a hot song, but you can discover more similar songs here:")
        
        recommend_similar_songs(user_df, hot_or_not_df, row['cluster'])


def recommend_similar_songs(user_df, hot_or_not_df, cluster_value=None, num_recommendations=5):
    if cluster_value is not None:
        cluster_songs = hot_or_not_df[hot_or_not_df['cluster'] == cluster_value]
        if not cluster_songs.empty:
            random_recommendations = cluster_songs.sample(n=min(num_recommendations, len(cluster_songs)))
            print()
            print("Here are your recommendations:")
            print()
            print("_____________________________________________________________________")
            print()
            for _, row in random_recommendations.iterrows():
                track_name = row['track_name'].title()
                artists = row['artists'].title()
                url_id = row['id']
                track_href = f'https://open.spotify.com/intl-fr/track/{url_id}'
                bold_start = '\033[1m'
                bold_end = '\033[0m'
                print(f"{bold_start}Artist:{bold_end} {artists}")
                print(f"{bold_start}Title:{bold_end} {track_name}")
                print(f"{bold_start}Link:{bold_end} {track_href}")
                print("_____________________________________________________________________")
                print()
            return
    
    print("No similar songs found.")

    import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
import pickle

