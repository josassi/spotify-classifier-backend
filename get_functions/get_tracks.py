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

list_tracks_ids = ["3Gpi6ZZyLDbeCUOSUOAq8k",
                   "261lG8mG0KnqzlxGVw2RMX",
                   "7w1SOEhXiaubik6iLjLvPM",
                   "5BE9B2FiFWBbBdoIQ1m1UP",
                   "2o0hVSbnkdvDDKKVNaUxnB",
                   "19Jj5oZJkD9eOTU920PRDr",
                   "6XBaTMiZa77Du2XEl1RNaa"]

i = 0
bulk = 50
offset = 0
# df_features = pd.DataFrame()
max_iterations = 100

# while i < max_iterations:
#     library = sp.tracks(limit=bulk, offset=offset)
#     if len(library["items"]) == 0:
#         break
#     list_tracks_ids = json_normalize(library["items"])["track.id"]
#     df = pd.DataFrame(sp.audio_features(list_tracks_ids))
#     offset += bulk
#     df_features = pd.concat([df_features, df])
#     i += 1
print(sp.audio_features(list_tracks_ids))
df_features = pd.DataFrame(sp.audio_features(list_tracks_ids))
df_features = df_features.reset_index().drop(columns=["index"])
df_features.to_csv("test_tracks.csv")
print(df_features)
