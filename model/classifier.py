from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd


def fit(df_training: pd.DataFrame, x_columns: list = None):
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

    clf = GradientBoostingClassifier()
    x = df_training[x_columns]
    y = df_training["category"]
    clf.fit(X=x, y=y)
    return clf


def predict(clf, df_saved_tracks, x_columns=None):
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
    if df_saved_tracks is None:
        print("Import user tracks first")
        return
    x = df_saved_tracks[x_columns]
    return clf.predict(x)
