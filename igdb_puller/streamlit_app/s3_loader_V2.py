"""
S3 Data Loader V2: Load pre-processed data from S3 for Streamlit app.

This module handles:
- Loading parquet files from public S3 bucket (memory-efficient streaming)
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
MEMORY OPTIMIZATION:
=============================================================================

This loader uses fsspec with HTTP filesystem to stream parquet files directly
from S3 without downloading the entire file into memory first. This avoids
memory spikes that can crash Streamlit Community Cloud (1GB limit).

=============================================================================
"""

import os
import gc
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
    
    Uses fsspec HTTP streaming for memory efficiency on Streamlit Cloud.
    """
    use_local = os.getenv("USE_LOCAL_DATA", "").lower() == "true"
    
    # Try local first if requested
    if use_local:
        local_path = LOCAL_DATA_DIR / FILES["games"]
        if local_path.exists():
            try:
                df = pd.read_parquet(local_path)
                return df
            except Exception as e:
                st.warning(f"Local file read failed, trying S3: {e}")
    
    # Stream from S3 using fsspec (memory-efficient)
    url = f"{S3_BASE_URL}{FILES['games']}"
    try:
        import fsspec
        # Use HTTP filesystem to stream parquet without full download
        with fsspec.open(url, mode='rb') as f:
            df = pd.read_parquet(f)
        gc.collect()  # Help free any temporary memory
        return df
    except ImportError:
        # Fallback to BytesIO method if fsspec not available
        st.warning("fsspec not available, using fallback method")
        try:
            data = _download_file(FILES["games"])
            df = pd.read_parquet(BytesIO(data))
            del data  # Free the bytes immediately
            gc.collect()
            return df
        except Exception as e:
            st.error(f"Error loading games data: {e}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading games data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="Loading recommendations from S3...")
def load_recommendations() -> pd.DataFrame:
    """
    Load pre-computed recommendations from S3.
    Cached for 1 hour.
    
    WARNING: This loads the entire recommendations file (~167 MB compressed).
    For memory-constrained environments (like Streamlit Cloud), 
    use load_recommendations_for_game() instead.
    
    Uses fsspec streaming for better memory efficiency.
    """
    use_local = os.getenv("USE_LOCAL_DATA", "").lower() == "true"
    
    # Try local first if requested
    if use_local:
        local_path = LOCAL_DATA_DIR / FILES["recommendations"]
        if local_path.exists():
            try:
                df = pd.read_parquet(local_path)
                return df
            except Exception as e:
                st.warning(f"Local file read failed, trying S3: {e}")
    
    # Stream from S3
    url = f"{S3_BASE_URL}{FILES['recommendations']}"
    try:
        import fsspec
        with fsspec.open(url, mode='rb') as f:
            df = pd.read_parquet(f)
        gc.collect()
        return df
    except ImportError:
        try:
            data = _download_file(FILES["recommendations"])
            df = pd.read_parquet(BytesIO(data))
            del data
            gc.collect()
            return df
        except Exception as e:
            st.error(f"Error loading recommendations: {e}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading recommendations: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="Fetching recommendations...")
def load_recommendations_for_game(game_id: int) -> Optional[pd.Series]:
    """
    Load recommendations for a SINGLE game using pyarrow filtering.
    
    This is much more memory-efficient than loading the entire file.
    Only loads the specific row needed by iterating through batches.
    Uses fsspec streaming to avoid downloading entire file.
    
    Args:
        game_id: The game ID to get recommendations for
        
    Returns:
        Series with recommendation data, or None if not found
    """
    import pyarrow.parquet as pq
    
    use_local = os.getenv("USE_LOCAL_DATA", "").lower() == "true"
    
    # Try local first if requested
    if use_local:
        local_path = LOCAL_DATA_DIR / FILES["recommendations"]
        if local_path.exists():
            try:
                parquet_file = pq.ParquetFile(local_path)
                for batch in parquet_file.iter_batches(batch_size=10000):
                    df_batch = batch.to_pandas()
                    game_rec = df_batch[df_batch["game_id"] == game_id]
                    if not game_rec.empty:
                        return game_rec.iloc[0]
                return None
            except Exception as e:
                st.warning(f"Local recommendations read failed, trying S3: {e}")
    
    # Stream from S3 using fsspec
    url = f"{S3_BASE_URL}{FILES['recommendations']}"
    try:
        import fsspec
        with fsspec.open(url, mode='rb') as f:
            parquet_file = pq.ParquetFile(f)
            # Read in batches and filter to find the target game
            for batch in parquet_file.iter_batches(batch_size=10000):
                df_batch = batch.to_pandas()
                game_rec = df_batch[df_batch["game_id"] == game_id]
                if not game_rec.empty:
                    del df_batch
                    gc.collect()
                    return game_rec.iloc[0]
                del df_batch
            gc.collect()
        return None
    except ImportError:
        # Fallback to BytesIO method if fsspec not available
        try:
            data = _download_file(FILES["recommendations"])
            parquet_file = pq.ParquetFile(BytesIO(data))
            for batch in parquet_file.iter_batches(batch_size=10000):
                df_batch = batch.to_pandas()
                game_rec = df_batch[df_batch["game_id"] == game_id]
                if not game_rec.empty:
                    del data
                    gc.collect()
                    return game_rec.iloc[0]
            del data
            gc.collect()
            return None
        except Exception as e:
            st.error(f"Error loading recommendations for game {game_id}: {e}")
            return None
    except Exception as e:
        st.error(f"Error loading recommendations for game {game_id}: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner="Loading embeddings from S3...")
def load_embeddings() -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load embeddings and ID map for real-time similarity (Option B).
    Cached for 1 hour.
    
    Returns:
        tuple: (embeddings array, id_map dataframe)
    """
    use_local = os.getenv("USE_LOCAL_DATA", "").lower() == "true"
    
    # Try local first if requested
    if use_local:
        emb_path = LOCAL_DATA_DIR / FILES["embeddings"]
        map_path = LOCAL_DATA_DIR / FILES["id_map"]
        if emb_path.exists() and map_path.exists():
            try:
                embeddings = np.load(emb_path)
                id_map = pd.read_parquet(map_path)
                return embeddings, id_map
            except Exception as e:
                st.warning(f"Local embeddings read failed, trying S3: {e}")
    
    # Load from S3
    try:
        import fsspec
        # Load embeddings (numpy files need download, can't stream)
        emb_url = f"{S3_BASE_URL}{FILES['embeddings']}"
        emb_data = _download_file(FILES["embeddings"])
        embeddings = np.load(BytesIO(emb_data))
        del emb_data
        gc.collect()
        
        # Load ID map (parquet can stream)
        map_url = f"{S3_BASE_URL}{FILES['id_map']}"
        with fsspec.open(map_url, mode='rb') as f:
            id_map = pd.read_parquet(f)
        
        return embeddings, id_map
    except ImportError:
        try:
            emb_data = _download_file(FILES["embeddings"])
            embeddings = np.load(BytesIO(emb_data))
            del emb_data
            
            map_data = _download_file(FILES["id_map"])
            id_map = pd.read_parquet(BytesIO(map_data))
            del map_data
            gc.collect()
            
            return embeddings, id_map
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
            return np.array([]), pd.DataFrame()
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
    Deduplicates by canonical_game_id to show only one entry per game.
    """
    games = load_games()
    if games.empty or not query:
        return pd.DataFrame()
    
    query_lower = query.lower()
    
    # Search in name
    mask = games["name"].fillna("").str.lower().str.contains(query_lower, regex=False)
    results = games[mask].copy()
    
    # Deduplicate by canonical_game_id if available
    if "canonical_game_id" in results.columns and not results.empty:
        # Sort so earliest release (or highest rating count) comes first per canonical group
        sort_cols = []
        ascending = []
        
        if "canonical_release_year" in results.columns:
            sort_cols.append("canonical_release_year")
            ascending.append(True)
        if "release_year" in results.columns:
            sort_cols.append("release_year")
            ascending.append(True)
        
        # Always prefer higher rating count as tiebreaker
        if "total_rating_count" in results.columns:
            sort_cols.append("total_rating_count")
            ascending.append(False)
        
        if sort_cols:
            results = results.sort_values(sort_cols, ascending=ascending, na_position='last')
        
        # Keep only the first (earliest/most popular) entry per canonical game
        results = results.drop_duplicates(subset=["canonical_game_id"], keep="first")
    
    # Final sort by rating count (most popular first)
    if "total_rating_count" in results.columns:
        results = results.sort_values("total_rating_count", ascending=False, na_position='last')
    
    return results.head(limit)


@st.cache_data(ttl=3600)
def get_recommendations_for_game(game_id: int, top_n: int = 10) -> pd.DataFrame:
    """
    Get pre-computed recommendations for a game.
    
    Uses memory-efficient loading that only fetches the specific game's 
    recommendations instead of loading the entire file.
    
    Filters out:
    - Games in the same canonical group (same game, different ports/versions)
    - Duplicate canonical games (shows only one variant per game)
    
    Args:
        game_id: The game ID to get recommendations for
        top_n: Number of recommendations to return
    
    Returns:
        DataFrame with recommended games and their details
    """
    # Use memory-efficient single-game loader instead of load_recommendations()
    rec_row = load_recommendations_for_game(game_id)
    
    if rec_row is None:
        return pd.DataFrame()
    
    games = load_games()
    if games.empty:
        return pd.DataFrame()
    
    # Get recommended IDs and scores - request more than needed to account for filtering
    buffer_size = top_n * 3  # Get 3x to account for filtering
    rec_ids = rec_row["recommended_ids"][:buffer_size]
    rec_scores = rec_row["scores"][:buffer_size]
    
    # Handle case where rec_ids might be stored as string
    if isinstance(rec_ids, str):
        import ast
        rec_ids = ast.literal_eval(rec_ids)[:buffer_size]
        rec_scores = ast.literal_eval(rec_row["scores"])[:buffer_size]
    
    # Get the selected game's canonical_game_id
    selected_row = games[games["id"] == game_id]
    selected_canonical = None
    if not selected_row.empty and "canonical_game_id" in games.columns:
        selected_canonical = selected_row.iloc[0].get("canonical_game_id")
    
    # Get game details for recommended games
    rec_games = games[games["id"].isin(rec_ids)].copy()
    
    # Add scores and maintain order
    id_to_score = dict(zip(rec_ids, rec_scores))
    id_to_order = {gid: idx for idx, gid in enumerate(rec_ids)}
    
    rec_games["similarity_score"] = rec_games["id"].map(id_to_score)
    rec_games["_order"] = rec_games["id"].map(id_to_order)
    
    # Filter out games in the same canonical group as the selected game
    if selected_canonical is not None and "canonical_game_id" in rec_games.columns:
        rec_games = rec_games[rec_games["canonical_game_id"] != selected_canonical]
    
    # Deduplicate by canonical_game_id (show only one variant per game)
    if "canonical_game_id" in rec_games.columns:
        rec_games = rec_games.sort_values("_order")  # Keep original order priority
        rec_games = rec_games.drop_duplicates(subset=["canonical_game_id"], keep="first")
    
    # Sort by original order and limit to top_n
    rec_games = rec_games.sort_values("_order").drop(columns=["_order"]).head(top_n)
    
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


def get_data_stats(include_recommendations: bool = False) -> dict:
    """Get statistics about the loaded data.
    
    Args:
        include_recommendations: If False (default), skip loading recommendations
                                  to save memory on Streamlit Cloud.
    """
    games = load_games()
    
    stats = {
        "total_games": len(games) if not games.empty else 0,
        "games_with_ratings": len(games[games["total_rating"].notna()]) if not games.empty else 0,
        "games_with_covers": len(games[games["cover_url"].notna()]) if not games.empty else 0,
        "total_recommendations": "N/A",
    }
    
    # Only load recommendations if explicitly requested (saves memory)
    if include_recommendations:
        recs = load_recommendations()
        stats["total_recommendations"] = len(recs) if not recs.empty else 0
    
    return stats
