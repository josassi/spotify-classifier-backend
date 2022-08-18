import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd


scope = "playlist-read-private,user-read-private,playlist-read-collaborative,user-library-read,playlist-modify-private"
username = "josass1"
client_id = "e8d821f241b2408baef01dbf9296a36d"
client_secret = "851229608a314367818738d87e532ee4"
uri = "http://localhost:8040"

playlist_name = "test"

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

# Create playlist
sp.user_playlist_create(user=userID, name="playlist_name", public=False)

# Get Playlist ID
df_features = pd.DataFrame(sp.user_playlists(userID)["items"])
playlist_id = df_features[df_features["name"] == playlist_name]["id"].iloc[0]

# Add track to the playlist
sp.playlist_add_items(playlist_id="2qfARh7BRxTdc9Bef0ZDHJ", items=["6XBaTMiZa77Du2XEl1RNaa"])
