"""
S3 Data Loader V2: Load pre-processed data from S3 for Streamlit app.

This module handles:
- Loading parquet files from public S3 bucket
- Caching data in memory for fast access
- Fallback to local files for development

=============================================================================
S3 BUCKET CONFIGURATION:
=============================================================================

1. S3 Bucket Name: igdb-streamlitapp-datasets
   Prefix: processed/

2. Apply this bucket policy in AWS Console (S3 → Bucket → Permissions → Bucket Policy):

{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadProcessedData",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::igdb-streamlitapp-datasets/processed/*"
        }
    ]
}

3. Disable "Block all public access" in bucket settings:
   - Go to S3 → igdb-streamlitapp-datasets → Permissions
   - Click "Edit" on "Block public access"
   - Uncheck all 4 boxes
   - Save changes

4. Files should be accessible at:
   https://igdb-streamlitapp-datasets.s3.us-east-2.amazonaws.com/processed/games_enriched.parquet
   https://igdb-streamlitapp-datasets.s3.us-east-2.amazonaws.com/processed/recommendations.parquet
   https://igdb-streamlitapp-datasets.s3.us-east-2.amazonaws.com/processed/embeddings.npy
   https://igdb-streamlitapp-datasets.s3.us-east-2.amazonaws.com/processed/game_id_map.parquet

=============================================================================
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st
import requests
from io import BytesIO


# =============================================================================
# S3 CONFIGURATION - MODIFY THESE IF YOUR BUCKET IS DIFFERENT
# =============================================================================

S3_BUCKET = "igdb-streamlitapp-datasets"
S3_PREFIX = "processed/"  # Changed from "IGDB/processed/" to match actual folder
S3_REGION = "us-east-2"  # Change if your bucket is in a different region

# Construct base URL (works for most regions)
# For us-east-1, the URL format is: https://bucket-name.s3.amazonaws.com/
# For other regions: https://bucket-name.s3.region.amazonaws.com/
S3_BASE_URL = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{S3_PREFIX}"

# =============================================================================
# LOCAL FALLBACK - For development without S3
# =============================================================================

LOCAL_DATA_DIR = Path(__file__).parent / "processed_data"

# =============================================================================
# FILE NAMES
# =============================================================================

FILES = {
    "games": "games_enriched.parquet",
    "recommendations": "recommendations.parquet",
    "embeddings": "embeddings.npy",
    "id_map": "game_id_map.parquet",
}


def _download_file(filename: str) -> bytes:
    """Download a file from S3 public bucket."""
    url = f"{S3_BASE_URL}{filename}"
    try:
        response = requests.get(url, timeout=120)  # 2 min timeout for large files
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download {filename} from S3: {e}\nURL: {url}")


def _load_local_file(filename: str) -> Optional[bytes]:
    """Load a file from local directory (fallback for development)."""
    filepath = LOCAL_DATA_DIR / filename
    if filepath.exists():
        return filepath.read_bytes()
    return None


def _get_file_bytes(filename: str, use_local: bool = False) -> bytes:
    """Get file bytes from S3 or local fallback."""
    if use_local:
        local_bytes = _load_local_file(filename)
        if local_bytes:
            return local_bytes
        # If local not found but use_local is True, try S3 as fallback
    
    return _download_file(filename)


@st.cache_data(ttl=3600, show_spinner="Loading games data from S3...")
def load_games() -> pd.DataFrame:
    """
    Load the enriched games dataframe from S3.
    Cached for 1 hour.
    """
    use_local = os.getenv("USE_LOCAL_DATA", "").lower() == "true"
    
    try:
        data = _get_file_bytes(FILES["games"], use_local)
        df = pd.read_parquet(BytesIO(data))
        return df
    except Exception as e:
        st.error(f"Error loading games data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="Loading recommendations from S3...")
def load_recommendations() -> pd.DataFrame:
    """
    Load pre-computed recommendations from S3.
    Cached for 1 hour.
    """
    use_local = os.getenv("USE_LOCAL_DATA", "").lower() == "true"
    
    try:
        data = _get_file_bytes(FILES["recommendations"], use_local)
        df = pd.read_parquet(BytesIO(data))
        return df
    except Exception as e:
        st.error(f"Error loading recommendations: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="Loading embeddings from S3...")
def load_embeddings() -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load embeddings and ID map for real-time similarity (Option B).
    Cached for 1 hour.
    
    Returns:
        tuple: (embeddings array, id_map dataframe)
    """
    use_local = os.getenv("USE_LOCAL_DATA", "").lower() == "true"
    
    try:
        # Load embeddings
        emb_data = _get_file_bytes(FILES["embeddings"], use_local)
        embeddings = np.load(BytesIO(emb_data))
        
        # Load ID map
        map_data = _get_file_bytes(FILES["id_map"], use_local)
        id_map = pd.read_parquet(BytesIO(map_data))
        
        return embeddings, id_map
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return np.array([]), pd.DataFrame()


@st.cache_data(ttl=3600)
def get_game_by_id(game_id: int) -> Optional[dict]:
    """Get a single game's details by ID."""
    games = load_games()
    if games.empty:
        return None
    
    game = games[games["id"] == game_id]
    if game.empty:
        return None
    
    return game.iloc[0].to_dict()


@st.cache_data(ttl=3600)
def search_games(query: str, limit: int = 50) -> pd.DataFrame:
    """
    Search games by name.
    Uses simple string matching (case-insensitive).
    """
    games = load_games()
    if games.empty or not query:
        return pd.DataFrame()
    
    query_lower = query.lower()
    
    # Search in name
    mask = games["name"].fillna("").str.lower().str.contains(query_lower, regex=False)
    results = games[mask].copy()
    
    # Sort by rating count (most popular first)
    if "total_rating_count" in results.columns:
        results = results.sort_values("total_rating_count", ascending=False, na_position='last')
    
    return results.head(limit)


@st.cache_data(ttl=3600)
def get_recommendations_for_game(game_id: int, top_n: int = 10) -> pd.DataFrame:
    """
    Get pre-computed recommendations for a game.
    
    Args:
        game_id: The game ID to get recommendations for
        top_n: Number of recommendations to return
    
    Returns:
        DataFrame with recommended games and their details
    """
    recommendations = load_recommendations()
    games = load_games()
    
    if recommendations.empty or games.empty:
        return pd.DataFrame()
    
    # Find recommendations for this game
    game_recs = recommendations[recommendations["game_id"] == game_id]
    if game_recs.empty:
        return pd.DataFrame()
    
    # Get recommended IDs and scores
    rec_row = game_recs.iloc[0]
    rec_ids = rec_row["recommended_ids"][:top_n]
    rec_scores = rec_row["scores"][:top_n]
    
    # Handle case where rec_ids might be stored as string
    if isinstance(rec_ids, str):
        import ast
        rec_ids = ast.literal_eval(rec_ids)
        rec_scores = ast.literal_eval(rec_row["scores"])[:top_n]
    
    # Get game details for recommended games
    rec_games = games[games["id"].isin(rec_ids)].copy()
    
    # Add scores and maintain order
    id_to_score = dict(zip(rec_ids, rec_scores))
    id_to_order = {gid: idx for idx, gid in enumerate(rec_ids)}
    
    rec_games["similarity_score"] = rec_games["id"].map(id_to_score)
    rec_games["_order"] = rec_games["id"].map(id_to_order)
    rec_games = rec_games.sort_values("_order").drop(columns=["_order"])
    
    return rec_games


def compute_realtime_recommendations(game_id: int, top_n: int = 10) -> pd.DataFrame:
    """
    Compute recommendations in real-time using embeddings (Option B).
    
    This is for benchmarking against pre-computed recommendations.
    Requires embeddings.npy and game_id_map.parquet to be available.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    embeddings, id_map = load_embeddings()
    games = load_games()
    
    if embeddings.size == 0 or games.empty:
        return pd.DataFrame()
    
    # Find index for this game
    game_idx_row = id_map[id_map["game_id"] == game_id]
    if game_idx_row.empty:
        return pd.DataFrame()
    
    game_idx = game_idx_row.iloc[0]["index"]
    
    # Compute similarity to all games
    game_embedding = embeddings[game_idx].reshape(1, -1)
    similarities = cosine_similarity(game_embedding, embeddings)[0]
    
    # Get top-N (excluding self)
    top_indices = np.argsort(similarities)[::-1][1:top_n + 1]
    top_scores = similarities[top_indices]
    top_ids = id_map.iloc[top_indices]["game_id"].values
    
    # Get game details
    rec_games = games[games["id"].isin(top_ids)].copy()
    
    id_to_score = dict(zip(top_ids, top_scores))
    id_to_order = {gid: idx for idx, gid in enumerate(top_ids)}
    
    rec_games["similarity_score"] = rec_games["id"].map(id_to_score)
    rec_games["_order"] = rec_games["id"].map(id_to_order)
    rec_games = rec_games.sort_values("_order").drop(columns=["_order"])
    
    return rec_games


def check_data_availability() -> dict[str, bool]:
    """
    Check which data files are available.
    Returns dict with file availability status.
    """
    use_local = os.getenv("USE_LOCAL_DATA", "").lower() == "true"
    available = {}
    
    for name, filename in FILES.items():
        try:
            # Try local first if USE_LOCAL_DATA is set
            if use_local:
                local_path = LOCAL_DATA_DIR / filename
                if local_path.exists():
                    available[name] = True
                    continue
            
            # Try S3 (just check if accessible with HEAD request)
            url = f"{S3_BASE_URL}{filename}"
            response = requests.head(url, timeout=10)
            available[name] = response.status_code == 200
        except Exception:
            available[name] = False
    
    return available


def get_data_stats() -> dict:
    """Get statistics about the loaded data."""
    games = load_games()
    recs = load_recommendations()
    
    return {
        "total_games": len(games) if not games.empty else 0,
        "games_with_ratings": len(games[games["total_rating"].notna()]) if not games.empty else 0,
        "games_with_covers": len(games[games["cover_url"].notna()]) if not games.empty else 0,
        "total_recommendations": len(recs) if not recs.empty else 0,
    }
