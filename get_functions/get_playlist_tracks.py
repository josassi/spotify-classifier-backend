import pandas as pd
from pandas import json_normalize


def get_tracks_from_playlist(sp, playlist_id):
    playlist = sp.playlist_items(playlist_id)
    tracks_ids = json_normalize(playlist["items"])["track.id"]
    df = pd.DataFrame(sp.audio_features(tracks_ids))
    return df
