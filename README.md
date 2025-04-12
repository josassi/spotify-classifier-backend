# Spotify Classifier Backend

A Python-based backend service for classifying your Spotify tracks into playlists based on audio features and genres.

## Overview

This project uses the Spotify Web API to analyze audio features of your saved tracks and create organized playlists based on genre classification. The classifier uses machine learning techniques to group similar tracks together based on their audio characteristics.

## Features

- **Track Analysis**: Fetch and analyze audio features from your saved Spotify tracks
- **Multiple Classification Methods**:
  - Using Spotify categories
  - Using existing playlists (including personal ones)
  - Using "radio" playlists from a song
  - Using a pre-trained model
- **Playlist Generation**: Automatically create new playlists based on classification results
- **Feature-based Similarity**: Find tracks with similar audio features

## Requirements

- Python 3.7+
- Spotify Developer Account
- Spotify API Credentials (Client ID and Client Secret)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spotify-classifier-backend.git
   cd spotify-classifier-backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install spotipy pandas scikit-learn tqdm
   ```

## Configuration

Create a `config.py` file in the root directory with your Spotify API credentials:

```python
from spotipy.oauth2 import SpotifyOAuth

# Spotify API credentials
client_id = 'your_client_id'
client_secret = 'your_client_secret'
redirect_uri = 'http://localhost:8888/callback'  # Must match your Spotify app settings

# Scope for API access
scope = "user-library-read playlist-read-private playlist-modify-private playlist-modify-public user-read-private user-top-read"

# Create OAuth Manager
oauth_manager = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope=scope
)
```

## Usage

### Basic Usage

```python
from main import TrackClassify
from config import oauth_manager

# Initialize the classifier
classifier = TrackClassify(oauth=oauth_manager)

# Get saved tracks
saved_tracks = classifier.get_user_saved_tracks(max_tracks=500)

# Define genre playlists for training
genres = {
    "Rock": "spotify:playlist:37i9dQZF1DWXRqgorJj26U",
    "Pop": "spotify:playlist:37i9dQZF1DX1ngEVM0lKrb",
    "Electronic": "spotify:playlist:37i9dQZF1DX4dyzvuaRJ0n"
}

# Train classifier
classifier.get_training_set(categories_dict=genres)
classifier.train_classifier()

# Classify tracks and create playlists
classifier.predict_and_create_playlists(prefix="Auto: ")
```

### Using the PlaylistGenerator

```python
from playlist_generator import PlaylistGenerator
from config import client_id, client_secret, redirect_uri

# Initialize the playlist generator
generator = PlaylistGenerator(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri
)

# Get user's saved tracks with audio features
tracks = generator.get_user_saved_tracks(max_tracks=500)

# Generate a playlist based on seed tracks
seed_track_ids = ["spotify:track:4iV5W9uYEdYUVa79Axb7Rh", "spotify:track:1301WleyT98MSxVHPZCA6M"]
generator.create_playlist_from_similar_tracks(
    seed_track_ids=seed_track_ids,
    playlist_name="Similar to My Favorites",
    description="Tracks similar to my favorites based on audio features",
    public=False,
    collaborative=False,
    target_size=30
)
```

## Technical Notes

### Spotify API Limitations

- When retrieving audio features, the Spotify API limits batch requests to 100 track IDs per request, but to avoid 403 Forbidden errors, this implementation uses a maximum of 20 track IDs per batch.
- Comprehensive error handling is implemented for API responses.

### Data Processing

- The project uses pandas for data manipulation and scikit-learn for feature scaling and classification.
- Audio features used for classification include: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, and tempo.

## Project Structure

- `main.py`: Contains the `TrackClassify` class for genre-based classification
- `playlist_generator.py`: Contains the `PlaylistGenerator` class for creating playlists based on track similarity
- `get_functions/`: Helper functions for retrieving data from Spotify
- `model/`: Classification model implementation
- `notebooks/`: Jupyter notebooks for data exploration and analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.
