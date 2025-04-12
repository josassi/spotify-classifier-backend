import logging
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PlaylistGenerator:
    """
    A class to generate playlists based on example tracks by finding similar tracks
    in the user's saved library using audio features.
    """
    
    # Audio features to use for similarity calculations
    AUDIO_FEATURES = [
        "danceability", "energy", "key", "loudness", "mode", 
        "speechiness", "acousticness", "instrumentalness", 
        "liveness", "valence", "tempo"
    ]
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, scope: str = None):
        """
        Initialize the PlaylistGenerator with Spotify API credentials.
        
        Args:
            client_id: Spotify API client ID
            client_secret: Spotify API client secret
            redirect_uri: Redirect URI for OAuth flow
            scope: Spotify API scopes (defaults to necessary scopes for this application)
        """
        if scope is None:
            scope = (
                "user-library-read playlist-read-private playlist-modify-private "
                "playlist-modify-public user-read-private user-top-read"
            )
        
        self.auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope,
            cache_path='.spotify_cache'  # Add cache to preserve token
        )
        
        self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
        self.user_info = self.sp.current_user()
        self.user_id = self.user_info["id"]
        self.df_saved_tracks = None
        
        logger.info(f"PlaylistGenerator initialized for user: {self.user_id}")
    
    def get_user_saved_tracks(self, limit: int = 50, max_tracks: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve user's saved tracks from Spotify including audio features.
        
        Args:
            limit: Number of tracks to retrieve per API call (max 50)
            max_tracks: Maximum number of tracks to retrieve (None for all)
            
        Returns:
            DataFrame containing saved tracks with their audio features
        """
        logger.info("Fetching user's saved tracks...")
        
        tracks = []
        offset = 0
        limit = min(limit, 50)  # Spotify API limit
        total = None
        
        # Set up progress bar if max_tracks is specified
        pbar = None
        if max_tracks:
            pbar = tqdm(total=min(max_tracks, 1000), desc="Fetching saved tracks")
        
        # Fetch tracks in batches
        while True:
            current_tracks = self.sp.current_user_saved_tracks(limit=limit, offset=offset)
            
            if total is None:
                total = current_tracks["total"]
                if pbar is None and total:
                    pbar = tqdm(total=min(total, max_tracks or total), desc="Fetching saved tracks")
            
            # Break if no more tracks
            if not current_tracks["items"]:
                break
                
            # Extract track data
            for item in current_tracks["items"]:
                track = item["track"]
                
                # Skip local tracks (no audio features available)
                if track.get("is_local", False):
                    continue
                
                # Basic track info
                track_data = {
                    "id": track["id"],
                    "name": track["name"],
                    "artist": ", ".join([artist["name"] for artist in track["artists"]]),
                    "artist_id": track["artists"][0]["id"] if track["artists"] else None,
                    "album": track["album"]["name"],
                    "popularity": track["popularity"],
                    "added_at": item["added_at"]
                }
                
                tracks.append(track_data)
            
            if pbar:
                pbar.update(len(current_tracks["items"]))
            
            offset += limit
            
            # Break if reached max_tracks
            if max_tracks and len(tracks) >= max_tracks:
                tracks = tracks[:max_tracks]
                break
                
            # Break if reached the end
            if len(current_tracks["items"]) < limit:
                break
        
        if pbar:
            pbar.close()
        
        if not tracks:
            logger.warning("No saved tracks found.")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(tracks)
        
        # Get audio features in batches (Spotify API has limits for number of IDs)
        logger.info("Fetching audio features for tracks...")
        all_features = []
        track_ids = df["id"].tolist()
        
        # Use smaller batch size (20) to avoid API limits
        for i in tqdm(range(0, len(track_ids), 20), desc="Fetching audio features"):
            batch_ids = track_ids[i:i+20]
            try:
                features = self.sp.audio_features(batch_ids)
                if features:
                    # Log the first feature to see its structure
                    if i == 0 and features[0]:
                        logger.info(f"Sample audio feature: {list(features[0].keys()) if features[0] else 'None'}")
                    valid_features = [f for f in features if f is not None]
                    all_features.extend(valid_features)
            except Exception as e:
                logger.error(f"Error fetching audio features for batch {i}: {str(e)}")
                # Continue with next batch rather than failing completely
        
        # Create audio features DataFrame
        df_features = pd.DataFrame(all_features)
        
        # Check if we have any features
        if df_features.empty:
            logger.warning("No audio features found for tracks.")
            return df
            
        # Make sure 'id' column exists
        if 'id' not in df_features.columns:
            logger.error("The 'id' column is missing from audio features response.")
            # Print column names for debugging
            logger.info(f"Available columns in audio features: {df_features.columns.tolist() if not df_features.empty else 'None'}")
            return df
        
        # Merge track info with audio features
        logger.info(f"Merging {len(df)} tracks with {len(df_features)} audio features")
        df_merged = pd.merge(df, df_features, on="id", how="inner")
        
        logger.info(f"Retrieved {len(df_merged)} saved tracks with audio features.")
        
        # Store the result for later use
        self.df_saved_tracks = df_merged
        return df_merged
    
    def get_tracks_features(self, track_ids: List[str]) -> pd.DataFrame:
        """Attempt to refresh token if needed"""
        # Check if token needs refresh
        try:
            self.sp.current_user()
        except Exception as e:
            logger.warning(f"Token may have expired: {e}. Attempting to refresh...")
            # Try to get a fresh token
            try:
                self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
                logger.info("Successfully refreshed Spotify token")
            except Exception as refresh_error:
                logger.error(f"Failed to refresh token: {refresh_error}")
        
        """
        Get audio features for specific tracks.
        
        Args:
            track_ids: List of Spotify track IDs
            
        Returns:
            DataFrame with audio features for the tracks
        """
        if not track_ids:
            logger.warning("No track IDs provided.")
            return pd.DataFrame()
        
        logger.info(f"Fetching features for {len(track_ids)} tracks...")
        
        # Get track details
        tracks_details = []
        for i in range(0, len(track_ids), 50):
            batch_ids = track_ids[i:i+50]
            tracks_batch = self.sp.tracks(batch_ids)["tracks"]
            for track in tracks_batch:
                if track:
                    tracks_details.append({
                        "id": track["id"],
                        "name": track["name"],
                        "artist": ", ".join([artist["name"] for artist in track["artists"]]),
                        "album": track["album"]["name"],
                        "popularity": track["popularity"]
                    })
        
        # Get audio features with smaller batch size
        audio_features = []
        for i in range(0, len(track_ids), 20):
            batch_ids = track_ids[i:i+20]
            try:
                features = self.sp.audio_features(batch_ids)
                audio_features.extend([f for f in features if f is not None])
            except Exception as e:
                logger.error(f"Error fetching audio features for batch {i}: {str(e)}")
                # Continue with next batch
        
        # Create DataFrames
        df_tracks = pd.DataFrame(tracks_details)
        
        # Handle empty audio features
        if not audio_features:
            logger.warning("No audio features returned for example tracks. API might be rate limiting.")
            # Return just the track info without audio features
            return df_tracks
            
        df_features = pd.DataFrame(audio_features)
        
        # Check if we have features and if id column exists
        if df_features.empty:
            logger.warning("Empty audio features DataFrame for example tracks")
            return df_tracks
            
        if 'id' not in df_features.columns:
            logger.error(f"Missing 'id' column in audio features. Available columns: {df_features.columns.tolist() if not df_features.empty else 'None'}")
            return df_tracks
        
        # Merge with audio features
        df = pd.merge(df_tracks, df_features, on="id", how="inner")
        
        # If merge resulted in empty DataFrame, return just track info
        if df.empty and not df_tracks.empty:
            logger.warning("Merge resulted in empty DataFrame. Returning track info only.")
            return df_tracks
        
        logger.info(f"Retrieved features for {len(df)} tracks.")
        return df
    
    def find_similar_tracks(self, example_track_ids: List[str], top_n: int = 50, 
                            similarity_method: str = "nearest") -> List[Dict]:
        """
        Find tracks in the user's saved library that are similar to the example tracks.
        If audio features are unavailable, will try to continue with basic track info.
        """
        logger.info(f"Finding tracks similar to: {example_track_ids}")
        """
        Find tracks in the user's saved library that are similar to the example tracks.
        
        Args:
            example_track_ids: List of Spotify track IDs to use as examples
            top_n: Number of similar tracks to return
            similarity_method: Method to use for similarity calculation:
                                "nearest" - minimum distance to any example
                                "average" - average distance to all examples
                                "weighted" - weighted average, favoring closer matches
            
        Returns:
            List of dictionaries with similar track information
        """
        if not example_track_ids:
            logger.error("No example tracks provided.")
            return []
        
        # Make sure we have saved tracks loaded
        if self.df_saved_tracks is None or len(self.df_saved_tracks) == 0:
            logger.info("No saved tracks loaded. Fetching saved tracks...")
            self.get_user_saved_tracks()
            
            if self.df_saved_tracks is None or len(self.df_saved_tracks) == 0:
                logger.error("Failed to fetch saved tracks.")
                return []
        
        # Get features for example tracks
        example_tracks_df = self.get_tracks_features(example_track_ids)
        
        if example_tracks_df.empty:
            logger.error("Failed to get any information for example tracks.")
            return []
            
        # Check if we have audio features or just basic track info
        has_audio_features = all(feat in example_tracks_df.columns for feat in ['danceability', 'energy', 'tempo'])
        
        if not has_audio_features:
            logger.warning("Audio features not available for example tracks. Cannot perform audio-based similarity matching.")
            # As a fallback, we could implement a basic popularity or name-based recommendation
            # For now, let's just return an informative message
            return [{"error": "Audio features unavailable from Spotify API. Please try again later."}]
        
        logger.info(f"Finding tracks similar to {len(example_tracks_df)} example tracks using {similarity_method} method...")
        
        # Get feature columns (only numeric features for distance calculation)
        feature_columns = [col for col in self.AUDIO_FEATURES 
                           if col in example_tracks_df.columns and col in self.df_saved_tracks.columns]
        
        logger.info(f"Using features: {', '.join(feature_columns)}")
        
        # Combine example and saved tracks for consistent scaling
        combined_df = pd.concat([
            example_tracks_df[feature_columns], 
            self.df_saved_tracks[feature_columns]
        ])
        
        # Normalize features (important for distance calculation)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(combined_df[feature_columns])
        
        # Split back into example and saved tracks
        example_scaled = scaled_features[:len(example_tracks_df)]
        saved_scaled = scaled_features[len(example_tracks_df):]
        
        # Calculate distances between saved tracks and example tracks
        similarity_scores = []
        
        for i, track in enumerate(tqdm(self.df_saved_tracks.itertuples(), 
                                      total=len(self.df_saved_tracks),
                                      desc="Calculating similarities")):
            # Skip if track is in example tracks
            if track.id in example_track_ids:
                continue
                
            saved_track_vector = saved_scaled[i]
            
            # Calculate distances to each example track
            distances = []
            for j in range(len(example_tracks_df)):
                example_vector = example_scaled[j]
                distance = np.linalg.norm(saved_track_vector - example_vector)
                distances.append(distance)
            
            # Calculate similarity score based on selected method
            if similarity_method == "nearest":
                score = min(distances)  # Smallest distance to any example
            elif similarity_method == "average":
                score = sum(distances) / len(distances)  # Average distance to all examples
            elif similarity_method == "weighted":
                # Weighted average (closer examples have more influence)
                weights = [1/(d + 0.001) for d in distances]  # Avoid division by zero
                score = sum(d * w for d, w in zip(distances, weights)) / sum(weights)
            else:
                score = min(distances)  # Default to nearest neighbor
            
            similarity_scores.append({
                "id": track.id,
                "name": track.name,
                "artist": track.artist,
                "album": track.album,
                "similarity_score": score,
                "nearest_example": example_tracks_df.iloc[np.argmin(distances)]["name"]
            })
        
        # Sort by similarity (lower score = more similar)
        similar_tracks = sorted(similarity_scores, key=lambda x: x["similarity_score"])[:top_n]
        
        logger.info(f"Found {len(similar_tracks)} similar tracks.")
        return similar_tracks
    
    def create_playlist(self, name: str, description: str = "", public: bool = False) -> Optional[str]:
        """
        Create a new empty playlist.
        
        Args:
            name: Name of the playlist
            description: Description of the playlist
            public: Whether the playlist should be public
            
        Returns:
            Playlist ID if successful, None otherwise
        """
        logger.info(f"Creating playlist '{name}'...")
        
        try:
            playlist = self.sp.user_playlist_create(
                user=self.user_id,
                name=name,
                public=public,
                description=description
            )
            logger.info(f"Created playlist: {playlist['name']} (ID: {playlist['id']})")
            return playlist["id"]
        except Exception as e:
            logger.error(f"Failed to create playlist: {str(e)}")
            return None
    
    def add_tracks_to_playlist(self, playlist_id: str, track_ids: List[str]) -> bool:
        """
        Add tracks to an existing playlist.
        
        Args:
            playlist_id: Spotify playlist ID
            track_ids: List of track IDs to add
            
        Returns:
            True if successful, False otherwise
        """
        if not track_ids:
            logger.warning("No tracks to add to the playlist.")
            return False
        
        logger.info(f"Adding {len(track_ids)} tracks to playlist...")
        
        try:
            # Spotify has a limit of 100 tracks per request
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                self.sp.playlist_add_items(playlist_id, batch)
            
            logger.info(f"Added {len(track_ids)} tracks to playlist {playlist_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add tracks to playlist: {str(e)}")
            return False
    
    def create_similar_tracks_playlist(self, example_track_ids: List[str], 
                                      name: Optional[str] = None,
                                      description: Optional[str] = None,
                                      top_n: int = 50) -> Optional[str]:
        """Remove 'spotify:track:' prefix if present in track IDs"""
        # Remove 'spotify:track:' prefix if present
        cleaned_track_ids = []
        for track_id in example_track_ids:
            if track_id.startswith("spotify:track:"):
                cleaned_track_ids.append(track_id.split(":")[-1])
            else:
                cleaned_track_ids.append(track_id)
        
        # Use cleaned track IDs
        example_track_ids = cleaned_track_ids
        """
        Create a playlist with tracks similar to the provided example tracks.
        
        Args:
            example_track_ids: List of track IDs to use as examples
            name: Name for the new playlist (defaults to based on example tracks)
            description: Description for the playlist
            top_n: Number of similar tracks to include
            
        Returns:
            ID of the created playlist if successful, None otherwise
        """
        if not example_track_ids:
            logger.error("No example tracks provided.")
            return None
        
        # Get example track details for naming
        example_tracks = self.get_tracks_features(example_track_ids)
        
        if example_tracks.empty:
            logger.error("Failed to get example track details.")
            return None
        
        # Generate name if not provided
        if name is None:
            if len(example_tracks) == 1:
                name = f"Similar to {example_tracks.iloc[0]['name']}"
            else:
                name = f"Similar to {example_tracks.iloc[0]['name']} and {len(example_tracks)-1} other tracks"
        
        # Generate description if not provided
        if description is None:
            track_names = ", ".join([f"{row['name']} by {row['artist']}" 
                                   for _, row in example_tracks.iterrows()])
            description = f"Tracks similar to: {track_names}"
        
        # Find similar tracks
        similar_tracks = self.find_similar_tracks(example_track_ids, top_n=top_n)
        
        if not similar_tracks:
            logger.error("No similar tracks found.")
            return None
        
        # Create playlist
        playlist_id = self.create_playlist(name, description)
        
        if not playlist_id:
            logger.error("Failed to create playlist.")
            return None
        
        # Add tracks to playlist
        track_ids = [track["id"] for track in similar_tracks]
        success = self.add_tracks_to_playlist(playlist_id, track_ids)
        
        if not success:
            logger.error("Failed to add tracks to playlist.")
            return None
        
        logger.info(f"Successfully created playlist '{name}' with {len(track_ids)} similar tracks.")
        return playlist_id


# Example usage
def authenticate_spotify(client_id, client_secret, redirect_uri, scope=None):
    """Handle Spotify authentication with proper error reporting"""
    if scope is None:
        scope = (
            "user-library-read playlist-read-private playlist-modify-private "
            "playlist-modify-public user-read-private user-top-read"
        )
    
    try:
        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope,
            cache_path='.spotify_cache'
        )
        
        # Force token acquisition to test auth
        token_info = auth_manager.get_cached_token()
        if not token_info or auth_manager.is_token_expired(token_info):
            print("Opening browser for Spotify authorization...")
            auth_manager.get_access_token()
            
        sp = spotipy.Spotify(auth_manager=auth_manager)
        
        # Verify we can make a simple API call
        user = sp.current_user()
        print(f"Successfully authenticated as: {user['display_name']} ({user['id']})")
        return sp, auth_manager
    
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Initial auth test before creating the generator
    print("Testing Spotify authentication...")
    sp, auth_manager = authenticate_spotify(
        client_id="e8d821f241b2408baef01dbf9296a36d",
        client_secret="851229608a314367818738d87e532ee4",
        redirect_uri="http://localhost:8040"
    )
    
    if not sp:
        print("Failed to authenticate with Spotify. Exiting.")
        import sys
        sys.exit(1)
        
    # Initialize the playlist generator with pre-authenticated client
    generator = PlaylistGenerator(
        client_id="e8d821f241b2408baef01dbf9296a36d",
        client_secret="851229608a314367818738d87e532ee4",
        redirect_uri="http://localhost:8040"
    )
    
    try:
        # Get user's saved tracks (limit to 50 for initial testing)
        saved_tracks = generator.get_user_saved_tracks(max_tracks=50)
        print(f"Found {len(saved_tracks)} saved tracks")
        
        # Try getting the user's top tracks instead of using a hardcoded example
        print("Fetching your top tracks to use as examples...")
        try:
            top_tracks = generator.sp.current_user_top_tracks(limit=3, time_range="medium_term")
            example_tracks = [track["id"] for track in top_tracks["items"]][:3]
            print(f"Using your top tracks as examples: {example_tracks}")
        except Exception as e:
            print(f"Failed to get top tracks: {e}. Using default example track.")
            # Fallback to the hardcoded example
            example_tracks = ["27DwBvyy3fmBqU1YCKQEDV"]
        
        # Create a playlist with similar tracks
        playlist_id = generator.create_similar_tracks_playlist(
            example_track_ids=example_tracks,
            name="My Similar Tracks Playlist",
            top_n=30
        )
        
        if playlist_id:
            print(f"Created playlist: {playlist_id}")
            print(f"Check your Spotify account for the new playlist!")
        else:
            print("Failed to create playlist")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
