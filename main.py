import spotipy
from pandas import json_normalize
from config import oauth_manager
import pandas as pd
from get_functions.get_playlist_tracks import get_tracks_from_playlist
from sklearn.ensemble import GradientBoostingClassifier

'''
The application is able to classify your saved tracks and create Playlists by genre from them.
The classification can be done in 3 ways:
1) Using spotify categories
2) Using existing playlists (including personal ones)
3) Using a "radio" playlist from a song
4) Using pretrained model (with my choices)
For now: focus on the 2nd one.
'''


class TrackClassify(spotipy.Spotify):

    def __init__(self, oauth):
        super().__init__(oauth_manager=oauth)
        self.user_id = self.me()['id']
        self.created_playlists = {}
        self.df_saved_tracks = None
        self.categories_dict = None
        self.df_training = None
        self.clf = None

    def get_user_saved_tracks(self, bulk: int = 50, max_iterations: int = 100):
        """
        Function to get the user's tracks with their features for classification.
        :param bulk: Size of the limit for the queries. Max authorized by Spotify=50.
        :param max_iterations: Maximum of iterations.
        :return: pd.DataFrame of tracks with their features.
        """

        if self.df_saved_tracks is not None:
            print("The user tracks have already been uploaded")
            return
        i = 0
        offset = 0
        df_features = pd.DataFrame()

        while i < max_iterations:
            library = self.current_user_saved_tracks(limit=bulk, offset=offset)
            if len(library["items"]) == 0:
                break
            tracks_ids = json_normalize(library["items"])["track.id"]
            df = pd.DataFrame(self.audio_features(tracks_ids))
            offset += bulk
            df_features = pd.concat([df_features, df])
            i += 1
        self.df_saved_tracks = df_features.reset_index().drop(columns=["index"])

    # def get_classes_and_playlists(self, how: str = "existing"):
    #     """
    #     Creates or validate the dictionary used to get
    #     :param how:
    #     :return:
    #     """
    #     if self.categories_dict is not None:
    #         print("The categories are already defined")
    #         return
    #     self.categories_dict = self.get_playlists_and_categories()
    #     return

    def get_training_set(self, categories_dict: dict = None):
        if categories_dict is None:
            categories_dict = self.categories_dict
        else:
            self.categories_dict = categories_dict
        if self.df_training is not None:
            print("The training set already exists.")
            return
        df_final = pd.DataFrame()
        for category, playlist_id in categories_dict.items():
            df_tmp = get_tracks_from_playlist(sp=self, playlist_id=playlist_id)
            df_tmp.loc[:, "category"] = category
            df_final = pd.concat([df_final, df_tmp])
        self.df_training = df_final

    def fit(self, x_columns=None):
        if x_columns is None:
            x_columns = ["danceability",
                         "energy",
                         "speechiness",
                         "acousticness",
                         "instrumentalness",
                         "liveness",
                         "valence",
                         "tempo",
                         "loudness",
                         "duration_ms"]
        if self.df_training is None:
            self.get_training_set()
        clf = GradientBoostingClassifier()
        x = self.df_training[x_columns]
        y = self.df_training["category"]
        clf.fit(X=x, y=y)
        self.clf = clf

    def predict(self, x_columns=None):
        if x_columns is None:
            x_columns = ["danceability",
                         "energy",
                         "speechiness",
                         "acousticness",
                         "instrumentalness",
                         "liveness",
                         "valence",
                         "tempo",
                         "loudness",
                         "duration_ms"]
        if self.df_saved_tracks is None:
            print("Import user tracks first")
            return
        if self.clf is None:
            self.fit(x_columns=x_columns)
        x = self.df_saved_tracks[x_columns]
        self.df_saved_tracks.loc[:, "predicted category"] = self.clf.predict(x)

    def create_playlist(self, playlist_name: str):
        self.user_playlist_create(user=self.user_id, name=playlist_name, public=False)
        df_features = pd.DataFrame(self.user_playlists(self.user_id)["items"])
        playlist_id = df_features[df_features["name"] == playlist_name]["id"].iloc[0]
        self.created_playlists[playlist_name] = playlist_id
        return playlist_id

    def add_tracks_to_playlist(self, playlist_id: str, items: list, bulk: int = 100):
        nb_iteration = int(len(items) / bulk) + (len(items) % bulk > 0)
        for i in range(nb_iteration):
            bulked_items = items[i * bulk:(i + 1) * bulk]
            self.playlist_add_items(playlist_id=playlist_id, items=bulked_items)

    def create_categorized_playlists(self):
        if "predicted category" not in self.df_saved_tracks:
            self.predict()
            print("classified saved tracks")
        for category in self.categories_dict.keys():
            tracks_to_add = list(self.df_saved_tracks[self.df_saved_tracks["predicted category"] == category]["id"])
            playlist_id = self.create_playlist(playlist_name=category)
            self.add_tracks_to_playlist(playlist_id=playlist_id, items=tracks_to_add)

    def saved_to_playlists(self, genres_dict=None):
        self.get_user_saved_tracks()
        self.get_training_set(categories_dict=genres_dict)
        self.fit()
        self.predict()
        self.create_categorized_playlists()


if __name__ == "__main__":
    sp = TrackClassify(oauth=oauth_manager)
    genres_dictionary = {
        "Latino": "6iP66p2aJ7DmWcma2OnRCT",
        "Techno": "37i9dQZF1E8UGEwVNmCxpX",
        "House": "37i9dQZF1E8M8Dm0QlxgbL",
        "French Songs": "37i9dQZF1E8KZrW4SHTUK2"
    }
    # sp.saved_to_playlists(genres_dict=genres_dictionary)
    sp.get_user_saved_tracks()
    # print(sp.df_saved_tracks)
    sp.get_training_set(categories_dict=genres_dictionary)
    # print(sp.df_training)
    sp.fit()
    sp.predict()
    # print(sp.df_saved_tracks)
    print(sp.categories_dict)
    sp.create_categorized_playlists()
