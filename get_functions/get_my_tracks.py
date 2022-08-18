import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from pandas import json_normalize

scope = "playlist-read-private,user-read-private,playlist-read-collaborative,user-library-read"
username = "josass1"
client_id = "e8d821f241b2408baef01dbf9296a36d"
client_secret = "851229608a314367818738d87e532ee4"
uri = "http://localhost:8040"

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

userID = sp.me()['id']
# playlists = sp.user_playlists(userID) #gives a Dictionary of user playlists

i = 0
bulk = 50
offset = 0
df_features = pd.DataFrame()
max_iterations = 100

while i < max_iterations:
    library = sp.current_user_saved_tracks(limit=bulk, offset=offset)
    if len(library["items"]) == 0:
        break
    tracks_ids = json_normalize(library["items"])["track.id"]
    df = pd.DataFrame(sp.audio_features(tracks_ids))
    offset += bulk
    df_features = pd.concat([df_features, df])
    i += 1

# print(sp.audio_analysis("1G391cbiT3v3Cywg8T7DM1"))
df_features = df_features.reset_index().drop(columns=["index"])
# df_features.to_csv("my_spotify_tracks.csv")
print(df_features.iloc[2])
