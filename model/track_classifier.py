ID_COLUMNS = ["type", "id", "uri", "track_href", "analysis_url"]
COMPUTED_FEATURES_COLUMNS = ["danceability",
                             "energy",
                             "speechiness",
                             "acousticness",
                             "instrumentalness",
                             "liveness",
                             "valence"]
INTRISIC_FEATURES_COLUMNS = ["tempo", "loudness", "duration_ms"]
OTHER_FEATURES_COLUMNS = ["time_signature", "key", "mode"]

SELECTED_FEATURES_COLUMNS = COMPUTED_FEATURES_COLUMNS + INTRISIC_FEATURES_COLUMNS
ALL_FEATURES_COLUMNS = SELECTED_FEATURES_COLUMNS + OTHER_FEATURES_COLUMNS

X = df[COMPUTED_FEATURES_COLUMNS]

clf = GradientBoostingClassifier()
run_model(clf);