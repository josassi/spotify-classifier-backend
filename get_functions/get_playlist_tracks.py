import pandas as pd
from pandas import json_normalize
from typing import Optional, List


def get_tracks_from_playlist(sp, playlist_id: str, limit: int = 100, offset: int = 0, max_items: Optional[int] = None) -> pd.DataFrame:
    """
    Retrieves tracks from a Spotify playlist with their audio features.
    
    Args:
        sp: A Spotify API client instance
        playlist_id: The ID of the playlist to retrieve tracks from
        limit: The maximum number of tracks to retrieve per API call (max 100)
        offset: The index of the first track to retrieve
        max_items: Maximum number of tracks to retrieve in total (None for all tracks)
    
    Returns:
        A DataFrame containing the audio features of the tracks in the playlist
    """
    all_tracks = []
    total_fetched = 0
    
    # Set a reasonable limit (Spotify API maximum is 100)
    limit = min(limit, 100)
    
    while True:
        # Get a batch of tracks from the playlist
        playlist_items = sp.playlist_items(
            playlist_id,
            offset=offset,
            limit=limit,
            additional_types=["track"]
        )
        
        if not playlist_items["items"]:
            break
            
        # Extract tracks and filter out None values (e.g., local files or podcasts)
        tracks = [item for item in playlist_items["items"] if item.get("track") is not None]
        all_tracks.extend(tracks)
        
        # Update counters
        offset += limit
        total_fetched += len(tracks)
        
        # Check if we've reached the desired number of tracks or all available tracks
        if (max_items is not None and total_fetched >= max_items) or len(tracks) < limit:
            break
    
    # Limit to max_items if specified
    if max_items is not None and len(all_tracks) > max_items:
        all_tracks = all_tracks[:max_items]
    
    # Extract track IDs and filter out None values
    tracks_df = json_normalize(all_tracks)
    
    # Handle case where no valid tracks were found
    if tracks_df.empty or "track.id" not in tracks_df.columns:
        return pd.DataFrame()
    
    # Filter out None values
    tracks_ids = tracks_df["track.id"].dropna().tolist()
    
    if not tracks_ids:
        return pd.DataFrame()
    
    # Get audio features in batches of 100 (Spotify API limit)
    all_features = []
    for i in range(0, len(tracks_ids), 100):
        batch_ids = tracks_ids[i:i+100]
        features = sp.audio_features(batch_ids)
        all_features.extend([f for f in features if f is not None])
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Add track names if available
    if "track.name" in tracks_df.columns:
        # Create a mapping from ID to name
        id_to_name = dict(zip(tracks_df["track.id"], tracks_df["track.name"]))
        # Add track names to the features DataFrame
        df["track_name"] = df["id"].map(id_to_name)
    
    return df
