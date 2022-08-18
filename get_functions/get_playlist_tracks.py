import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from pandas import json_normalize

scope = "playlist-read-private,user-read-private,playlist-read-collaborative,user-library-read"
username = "josass1"
client_id = "e8d821f241b2408baef01dbf9296a36d"
client_secret = "851229608a314367818738d87e532ee4"
uri = "http://localhost:8040"

genres_dictionary = {
    "Latino": "6iP66p2aJ7DmWcma2OnRCT",
    "Techno": "37i9dQZF1E8UGEwVNmCxpX",
    "House": "37i9dQZF1E8M8Dm0QlxgbL",
    "French Songs": "37i9dQZF1E8KZrW4SHTUK2"
}

oauth_manager = SpotifyOAuth(client_id=client_id,
                             client_secret=client_secret,
                             redirect_uri=uri,
                             state=None,
                             scope=scope,
                             cache_path=None,
                             username=None,
                             proxies=None,
                             show_dialog=True,
                             requests_timeout=None)

sp = spotipy.Spotify(oauth_manager=oauth_manager)


def get_tracks_from_playlist(sp, playlist_id):
    playlist = sp.playlist_items(playlist_id)
    tracks_ids = json_normalize(playlist["items"])["track.id"]
    df = pd.DataFrame(sp.audio_features(tracks_ids))
    return df


def construct_database(genres_dict):
    df_final = pd.DataFrame()
    for genre, playlist_id in genres_dict.items():
        df_tmp = get_tracks_from_playlist(playlist_id)
        df_tmp.loc[:, "genre"] = genre
        df_final = pd.concat([df_final, df_tmp])
    return df_final


if __name__ == "__main__":
    df = construct_database(genres_dict=genres_dictionary)
    df.to_csv("playlist_tracks.csv")
    print(df)