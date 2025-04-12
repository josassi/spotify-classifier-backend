import spotipy
from spotipy.oauth2 import SpotifyOAuth
from pandas import json_normalize
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm
from datetime import datetime

from config import oauth_manager
from get_functions.get_playlist_tracks import get_tracks_from_playlist
from model.classifier import fit, predict, predict_proba

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

    def __init__(self, oauth: SpotifyOAuth, cache_file: Optional[str] = None):
        """
        Initialize the TrackClassify class.
        
        Args:
            oauth: SpotifyOAuth manager for authentication
            cache_file: Optional path to a cache file for storing authentication tokens
        """
        super().__init__(auth_manager=oauth, cache_handler=cache_file)
        
        # Get user info
        user_info = self.me()
        self.user_id = user_info['id']
        self.user_name = user_info.get('display_name', self.user_id)
        
        # Initialize data structures
        self.created_playlists = {}
        self.df_saved_tracks = None
        self.categories_dict = None
        self.df_training = None
        self.clf = None
        
        logger.info(f"Initialized TrackClassify for user: {self.user_name}")

    def get_user_saved_tracks(self, limit: int = 50, max_tracks: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve the user's saved tracks with their audio features for classification.
        
        Args:
            limit: Number of tracks to retrieve per API call (max 50)
            max_tracks: Maximum number of tracks to retrieve in total (None for all)
            
        Returns:
            DataFrame containing track information and audio features
        """
        # Check if tracks have already been fetched
        if self.df_saved_tracks is not None:
            logger.info("User tracks already loaded")
            return self.df_saved_tracks
            
        # Initialize variables
        offset = 0
        all_tracks = []
        all_audio_features = []
        total_fetched = 0
        
        # Ensure limit is within Spotify's constraints (max 50 for library endpoints)
        limit = min(limit, 50)
        
        # Get the total number of saved tracks to show progress
        initial_response = self.current_user_saved_tracks(limit=1)
        total_tracks = initial_response.get('total', 0)
        
        if max_tracks is not None:
            total_tracks = min(total_tracks, max_tracks)
            
        logger.info(f"Fetching up to {total_tracks} saved tracks")
        
        # Create progress bar
        pbar = tqdm(total=total_tracks, desc="Fetching saved tracks")
        
        while True:
            # Fetch batch of saved tracks
            library = self.current_user_saved_tracks(limit=limit, offset=offset)
            
            # Check if we got any tracks back
            if not library["items"]:
                break
                
            # Process the tracks
            tracks_data = json_normalize(library["items"])
            tracks_ids = tracks_data["track.id"].dropna().tolist()
            
            if not tracks_ids:
                offset += limit
                continue
                
            # Get audio features in batches (maximum 100 at a time)
            for i in range(0, len(tracks_ids), 100):
                batch_ids = tracks_ids[i:i+100]
                features = self.audio_features(batch_ids)
                valid_features = [f for f in features if f is not None]
                all_audio_features.extend(valid_features)
            
            # Add track metadata to the list
            all_tracks.extend(library["items"])
            
            # Update counters and progress
            offset += limit
            total_fetched += len(library["items"])
            pbar.update(min(len(library["items"]), total_tracks - pbar.n))
            
            # Check if we've reached the desired number of tracks
            if max_tracks is not None and total_fetched >= max_tracks:
                break
                
            # Check if we've fetched all available tracks
            if len(library["items"]) < limit:
                break
                
        pbar.close()
        
        # Check if we found any tracks
        if not all_audio_features:
            logger.warning("No valid tracks with audio features found")
            return pd.DataFrame()
            
        # Create DataFrame with audio features
        df_features = pd.DataFrame(all_audio_features)
        
        # Process track metadata (names, artists, albums, etc.)
        tracks_df = json_normalize(all_tracks)
        
        # Create mappings from track ID to metadata
        id_to_name = dict(zip(tracks_df["track.id"], tracks_df["track.name"]))
        id_to_artist = dict(zip(tracks_df["track.id"], tracks_df["track.artists"].apply(lambda x: x[0]['name'] if x else None)))
        id_to_album = dict(zip(tracks_df["track.id"], tracks_df["track.album.name"]))
        
        # Add metadata to features DataFrame
        df_features["track_name"] = df_features["id"].map(id_to_name)
        df_features["artist"] = df_features["id"].map(id_to_artist)
        df_features["album"] = df_features["id"].map(id_to_album)
        
        # Store the result
        self.df_saved_tracks = df_features
        
        logger.info(f"Successfully fetched {len(self.df_saved_tracks)} tracks with audio features")
        return self.df_saved_tracks

    def get_user_playlists(self) -> Dict[str, str]:
        """
        Get all user playlists and return them as a dictionary of name:id pairs.
        
        Returns:
            Dictionary mapping playlist names to their IDs
        """
        playlists = {}
        offset = 0
        limit = 50  # Spotify API limit
        
        logger.info("Fetching user playlists")
        
        while True:
            response = self.current_user_playlists(limit=limit, offset=offset)
            
            if not response["items"]:
                break
                
            # Add playlists to dictionary
            for playlist in response["items"]:
                playlists[playlist["name"]] = playlist["id"]
                
            # Check if we've fetched all playlists
            if len(response["items"]) < limit:
                break
                
            offset += limit
            
        logger.info(f"Found {len(playlists)} user playlists")
        return playlists

    def get_training_set(self, categories_dict: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Retrieve tracks from playlists for each category to create a training dataset.
        
        Args:
            categories_dict: Dictionary mapping category names to playlist IDs.
                             If None, uses self.categories_dict
                             
        Returns:
            DataFrame containing training data with audio features and categories
        """
        # Set the categories dictionary
        if categories_dict is None:
            if self.categories_dict is None:
                raise ValueError("No categories dictionary provided or previously set")
            categories_dict = self.categories_dict
        else:
            self.categories_dict = categories_dict
            
        # Check if training set already exists
        if self.df_training is not None:
            logger.info("Training set already exists")
            return self.df_training
            
        # Initialize the final dataframe
        df_final = pd.DataFrame()
        
        # For each category, get tracks from the corresponding playlist
        for category, playlist_id in tqdm(categories_dict.items(), desc="Fetching training data"):
            logger.info(f"Fetching tracks for category: {category} (playlist ID: {playlist_id})")
            
            # Get tracks from playlist with audio features
            df_tmp = get_tracks_from_playlist(sp=self, playlist_id=playlist_id)
            
            if df_tmp.empty:
                logger.warning(f"No valid tracks found for category: {category}")
                continue
                
            # Add category label
            df_tmp["category"] = category
            
            # Add to final dataset
            df_final = pd.concat([df_final, df_tmp])
            
            logger.info(f"Added {len(df_tmp)} tracks for category: {category}")
            
        # Check if we have any data
        if df_final.empty:
            logger.error("No training data could be fetched from any playlist")
            return pd.DataFrame()
            
        # Set the training data
        self.df_training = df_final.reset_index(drop=True)
        
        logger.info(f"Created training set with {len(self.df_training)} tracks across {len(categories_dict)} categories")
        return self.df_training

    def fit(self, x_columns: Optional[List[str]] = None) -> None:
        """
        Train the classifier model on the training data.
        
        Args:
            x_columns: List of audio feature columns to use for training.
                       If None, default features will be used
        """
        # Check if training data exists
        if self.df_training is None:
            logger.info("No training data available, fetching it now")
            self.get_training_set()
            
        # Check if training data was successfully fetched
        if self.df_training is None or self.df_training.empty:
            logger.error("Cannot fit model: No training data available")
            return
            
        # Let the user know we're training the model
        logger.info("Training classification model...")
        
        # Train the model
        self.clf = fit(df_training=self.df_training, x_columns=x_columns)
        
        logger.info("Model training complete")

    def predict(self, x_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Predict categories for saved tracks using the trained model.
        
        Args:
            x_columns: List of audio feature columns to use for prediction.
                       If None, default features will be used
                       
        Returns:
            DataFrame with tracks and their predicted categories
        """
        # Check if we have saved tracks
        if self.df_saved_tracks is None or self.df_saved_tracks.empty:
            logger.error("Cannot predict: No saved tracks available")
            return pd.DataFrame()
            
        # Check if model is trained
        if self.clf is None:
            logger.info("No trained model available, training now")
            self.fit(x_columns=x_columns)
            
        # Check if model training was successful
        if self.clf is None:
            logger.error("Cannot predict: Model training failed")
            return pd.DataFrame()
            
        # Make predictions
        logger.info("Classifying saved tracks...")
        predicted_class, predicted_proba = predict_proba(clf=self.clf, df_saved_tracks=self.df_saved_tracks, x_columns=x_columns)
        
        # Add predictions to the DataFrame
        self.df_saved_tracks["predicted_category"] = predicted_class
        self.df_saved_tracks["prediction_confidence"] = predicted_proba
        
        # Create a summary of results
        category_counts = self.df_saved_tracks["predicted_category"].value_counts()
        
        logger.info("Classification complete")
        logger.info("Category distribution:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} tracks ({count/len(self.df_saved_tracks)*100:.1f}%)")
            
        return self.df_saved_tracks[["track_name", "artist", "album", "predicted_category", "prediction_confidence"]]

    def create_playlist(self, playlist_name: str, description: str = "", public: bool = False) -> str:
        """
        Create a new playlist for the user.
        
        Args:
            playlist_name: Name of the playlist to create
            description: Optional description for the playlist
            public: Whether the playlist should be public (default: False)
            
        Returns:
            ID of the created playlist
        """
        logger.info(f"Creating playlist: {playlist_name}")
        
        # Create the playlist
        result = self.user_playlist_create(
            user=self.user_id, 
            name=playlist_name,
            public=public,
            description=description
        )
        
        # Extract the playlist ID
        playlist_id = result["id"]
        
        # Store the playlist ID
        self.created_playlists[playlist_name] = playlist_id
        
        logger.info(f"Created playlist '{playlist_name}' with ID: {playlist_id}")
        return playlist_id

    def add_tracks_to_playlist(self, playlist_id: str, items: List[str], bulk: int = 100) -> None:
        """
        Add tracks to a playlist in batches.
        
        Args:
            playlist_id: ID of the playlist to add tracks to
            items: List of track IDs or URIs to add
            bulk: Maximum number of tracks to add in a single API call (max 100)
        """
        if not items:
            logger.warning("No tracks to add to playlist")
            return
            
        # Ensure the bulk size is within Spotify's limit
        bulk = min(bulk, 100)
        
        # Calculate number of iterations needed
        total_tracks = len(items)
        nb_iterations = (total_tracks + bulk - 1) // bulk  # Ceiling division
        
        logger.info(f"Adding {total_tracks} tracks to playlist in {nb_iterations} batches")
        
        # Add tracks in batches with progress bar
        for i in tqdm(range(nb_iterations), desc="Adding tracks to playlist"):
            start_idx = i * bulk
            end_idx = min((i + 1) * bulk, total_tracks)
            batch_items = items[start_idx:end_idx]
            
            # Add the batch to the playlist
            self.playlist_add_items(playlist_id=playlist_id, items=batch_items)
            
        logger.info(f"Successfully added {total_tracks} tracks to playlist")

    def create_categorized_playlists(self, prefix: str = "Auto: ", min_confidence: float = 0.0) -> Dict[str, str]:
        """
        Create playlists for each category with tracks classified into that category.
        
        Args:
            prefix: Prefix to add to playlist names (default: "Auto: ")
            min_confidence: Minimum confidence threshold for including tracks (0.0 to 1.0)
            
        Returns:
            Dictionary mapping playlist names to their IDs
        """
        # Check if predictions exist
        if self.df_saved_tracks is None or "predicted_category" not in self.df_saved_tracks.columns:
            logger.info("No predictions available, running classification")
            self.predict()
            
        # Check again after attempting to predict
        if self.df_saved_tracks is None or "predicted_category" not in self.df_saved_tracks.columns:
            logger.error("Cannot create playlists: Classification failed")
            return {}
            
        # Get today's date for the playlist description
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Create a playlist for each category
        created_playlists = {}
        for category in self.categories_dict.keys():
            # Get tracks for this category that meet the confidence threshold
            category_tracks = self.df_saved_tracks[
                (self.df_saved_tracks["predicted_category"] == category) & 
                (self.df_saved_tracks["prediction_confidence"] >= min_confidence)
            ]
            
            # Skip if no tracks for this category
            if category_tracks.empty:
                logger.warning(f"No tracks classified as '{category}' with confidence >= {min_confidence}")
                continue
                
            # Track IDs to add to playlist
            tracks_to_add = list(category_tracks["id"])
            
            # Create playlist name and description
            playlist_name = f"{prefix}{category}"
            description = f"Auto-generated {category} playlist based on your saved tracks. Created on {today}."
            
            # Create the playlist
            playlist_id = self.create_playlist(playlist_name=playlist_name, description=description)
            
            # Add tracks to the playlist
            self.add_tracks_to_playlist(playlist_id=playlist_id, items=tracks_to_add)
            
            # Store the playlist ID
            created_playlists[playlist_name] = playlist_id
            
            logger.info(f"Created playlist '{playlist_name}' with {len(tracks_to_add)} tracks")
            
        return created_playlists

    def saved_to_playlists(self, genres_dict: Optional[Dict[str, str]] = None, prefix: str = "Auto: ", 
                          min_confidence: float = 0.0, max_tracks: Optional[int] = None) -> Dict[str, str]:
        """
        Complete pipeline to classify saved tracks and create categorized playlists.
        
        Args:
            genres_dict: Dictionary mapping genre names to playlist IDs for training
            prefix: Prefix to add to created playlist names
            min_confidence: Minimum confidence threshold for including tracks
            max_tracks: Maximum number of saved tracks to process (None for all)
            
        Returns:
            Dictionary of created playlists mapping names to IDs
        """
        logger.info("Starting full classification pipeline")
        
        # Step 1: Get user's saved tracks
        logger.info("Step 1/5: Retrieving saved tracks")
        self.get_user_saved_tracks(max_tracks=max_tracks)
        
        # Step 2: Get training data from genre playlists
        logger.info("Step 2/5: Building training dataset")
        self.get_training_set(categories_dict=genres_dict)
        
        # Step 3: Train the classifier
        logger.info("Step 3/5: Training classifier")
        self.fit()
        
        # Step 4: Classify the saved tracks
        logger.info("Step 4/5: Classifying tracks")
        self.predict()
        
        # Step 5: Create playlists based on classification
        logger.info("Step 5/5: Creating categorized playlists")
        created_playlists = self.create_categorized_playlists(prefix=prefix, min_confidence=min_confidence)
        
        # Summary
        logger.info(f"Pipeline complete! Created {len(created_playlists)} playlists.")
        
        return created_playlists


if __name__ == "__main__":
    # Example genre playlists for training (using Spotify's curated playlists)
    genres_dictionary = {
        "Latino": "6iP66p2aJ7DmWcma2OnRCT",
        "Techno": "37i9dQZF1E8UGEwVNmCxpX",
        "House": "37i9dQZF1E8M8Dm0QlxgbL",
        "French Songs": "37i9dQZF1E8KZrW4SHTUK2"
    }
    
    # Initialize the classifier with OAuth manager
    sp = TrackClassify(oauth=oauth_manager)
    
    # Choose one of the following options:
    
    # Option 1: Full pipeline (fetch, train, classify, create playlists)
    # sp.saved_to_playlists(genres_dict=genres_dictionary, prefix="Genre: ")
    
    # Option 2: Step-by-step process
    # 1. Get saved tracks
    sp.get_user_saved_tracks()
    
    # 2. Get training data from genre playlists
    sp.get_training_set(categories_dict=genres_dictionary)
    
    # Optional: Save training data to CSV
    sp.df_training.to_csv("playlist_tracks.csv")
    
    # 3. Train the classifier
    sp.fit()
    
    # 4. Classify saved tracks
    results = sp.predict()
    
    # 5. Show sample of results
    if not results.empty:
        print("\nSample of classified tracks:")
        print(results.head(10))
    
    # Optional: Save all classified tracks to CSV
    # sp.df_saved_tracks.to_csv("my_spotify_tracks.csv")
    
    # 6. Create playlists based on classification
    # sp.create_categorized_playlists(prefix="Genre: ")
