"""
S3 Data Loader V2 Backup: Load SMALLER pre-processed data from S3 for demo app.

This version reads files with "_S3Implementation" suffix:
- games_enriched_S3Implementation.parquet (~8-10 MB)
- recommendations_S3Implementation.parquet (~30-40 MB)
- embeddings_S3Implementation.npy
- game_id_map_S3Implementation.parquet

These smaller files contain only games with ratings (~33K games)
for reliable Streamlit Cloud deployment.

=============================================================================
TWO IMPLEMENTATIONS:
=============================================================================

1. s3_loader_V2_backup.py (this file):
   - Reads *_S3Implementation.parquet files
   - ~33K games with ratings only
   - Used by: streamlit_app_V2_backup.py

2. s3_loader_V2.py (main loader):
   - Reads games_enriched.parquet, recommendations.parquet
   - ~220K games (year >= 2000)
   - Used by: streamlit_app_V2.py

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
# S3 CONFIGURATION
# =============================================================================

S3_BUCKET = "igdb-streamlitapp-datasets"
S3_PREFIX = "processed/"
S3_REGION = "us-east-2"

S3_BASE_URL = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{S3_PREFIX}"

# =============================================================================
# LOCAL FALLBACK - For development without S3
# =============================================================================

LOCAL_DATA_DIR = Path(__file__).parent / "processed_data_backup"

# =============================================================================
# FILE NAMES - With _S3Implementation suffix
# =============================================================================

FILE_SUFFIX = "_S3Implementation"

FILES = {
    "games": f"games_enriched{FILE_SUFFIX}.parquet",
    "recommendations": f"recommendations{FILE_SUFFIX}.parquet",
    "embeddings": f"embeddings{FILE_SUFFIX}.npy",
    "id_map": f"game_id_map{FILE_SUFFIX}.parquet",
}


def _download_file(filename: str) -> bytes:
    """Download a file from S3 public bucket."""
    url = f"{S3_BASE_URL}{filename}"
    try:
        response = requests.get(url, timeout=120)
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
        with fsspec.open(url, mode='rb') as f:
            df = pd.read_parquet(f)
        gc.collect()
        return df
    except ImportError:
        # Fallback to BytesIO method if fsspec not available
        st.warning("fsspec not available, using fallback method")
        try:
            data = _download_file(FILES["games"])
            df = pd.read_parquet(BytesIO(data))
            del data
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
    
    Uses fsspec streaming for better memory efficiency.
    """
    use_local = os.getenv("USE_LOCAL_DATA", "").lower() == "true"
    
    if use_local:
        local_path = LOCAL_DATA_DIR / FILES["recommendations"]
        if local_path.exists():
            try:
                df = pd.read_parquet(local_path)
                return df
            except Exception as e:
                st.warning(f"Local file read failed, trying S3: {e}")
    
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
    Uses fsspec streaming to avoid downloading entire file.
    """
    import pyarrow.parquet as pq
    
    use_local = os.getenv("USE_LOCAL_DATA", "").lower() == "true"
    
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
                st.warning(f"Local file read failed, trying S3: {e}")
    
    # Stream from S3 using fsspec
    url = f"{S3_BASE_URL}{FILES['recommendations']}"
    try:
        import fsspec
        with fsspec.open(url, mode='rb') as f:
            parquet_file = pq.ParquetFile(f)
            for batch in parquet_file.iter_batches(batch_size=10000):
                df_batch = batch.to_pandas()
                game_rec = df_batch[df_batch["game_id"] == game_id]
                if not game_rec.empty:
                    gc.collect()
                    return game_rec.iloc[0]
        gc.collect()
        return None
    except ImportError:
        # Fallback: load full file
        recs = load_recommendations()
        if recs.empty:
            return None
        game_rec = recs[recs["game_id"] == game_id]
        if game_rec.empty:
            return None
        return game_rec.iloc[0]
    except Exception as e:
        st.warning(f"Error loading recommendations for game {game_id}: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner="Loading embeddings...")
def load_embeddings() -> np.ndarray:
    """
    Load text embeddings from S3.
    
    Uses fsspec streaming to avoid downloading entire file to memory at once.
    """
    use_local = os.getenv("USE_LOCAL_DATA", "").lower() == "true"
    
    if use_local:
        local_path = LOCAL_DATA_DIR / FILES["embeddings"]
        if local_path.exists():
            try:
                return np.load(local_path)
            except Exception as e:
                st.warning(f"Local file read failed, trying S3: {e}")
    
    url = f"{S3_BASE_URL}{FILES['embeddings']}"
    try:
        import fsspec
        with fsspec.open(url, mode='rb') as f:
            embeddings = np.load(f)
        gc.collect()
        return embeddings
    except ImportError:
        try:
            data = _download_file(FILES["embeddings"])
            embeddings = np.load(BytesIO(data))
            del data
            gc.collect()
            return embeddings
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
            return np.array([])
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return np.array([])


@st.cache_data(ttl=3600, show_spinner="Loading ID mapping...")
def load_id_map() -> pd.DataFrame:
    """Load game ID to embedding index mapping."""
    use_local = os.getenv("USE_LOCAL_DATA", "").lower() == "true"
    
    if use_local:
        local_path = LOCAL_DATA_DIR / FILES["id_map"]
        if local_path.exists():
            try:
                return pd.read_parquet(local_path)
            except Exception:
                pass
    
    try:
        data = _download_file(FILES["id_map"])
        df = pd.read_parquet(BytesIO(data))
        del data
        gc.collect()
        return df
    except Exception as e:
        st.error(f"Error loading ID map: {e}")
        return pd.DataFrame()


def search_games(query: str, limit: int = 20) -> pd.DataFrame:
    """
    Search games by name using substring matching.
    Returns top matches sorted by rating.
    """
    games = load_games()
    if games.empty:
        return pd.DataFrame()
    
    query_lower = query.lower()
    
    mask = games["name"].astype(str).str.lower().str.contains(query_lower, na=False)
    matches = games[mask].copy()
    
    if matches.empty:
        return pd.DataFrame()
    
    # Sort by total_rating (descending), handling NaN
    matches["sort_rating"] = matches["total_rating"].fillna(0)
    matches = matches.sort_values("sort_rating", ascending=False)
    matches = matches.drop(columns=["sort_rating"])
    
    return matches.head(limit)


def get_recommendations_for_game(game_id: int, top_n: int = 5) -> pd.DataFrame:
    """
    Get pre-computed recommendations for a game.
    Returns a DataFrame with full game info for recommendations.
    """
    rec_row = load_recommendations_for_game(game_id)
    
    if rec_row is None:
        return pd.DataFrame()
    
    recommended_ids = rec_row.get("recommended_ids", [])
    scores = rec_row.get("scores", [])
    
    # Handle numpy arrays - convert to list and check length
    if isinstance(recommended_ids, np.ndarray):
        recommended_ids = recommended_ids.tolist()
    if isinstance(scores, np.ndarray):
        scores = scores.tolist()
    
    if not recommended_ids or len(recommended_ids) == 0:
        return pd.DataFrame()
    
    # Limit to top_n
    recommended_ids = recommended_ids[:top_n]
    scores = scores[:top_n]
    
    games = load_games()
    if games.empty:
        return pd.DataFrame()
    
    # Get full game info for recommended games
    recs = games[games["id"].isin(recommended_ids)].copy()
    
    if recs.empty:
        return pd.DataFrame()
    
    # Add similarity scores
    id_to_score = dict(zip(recommended_ids, scores))
    recs["similarity_score"] = recs["id"].map(id_to_score)
    
    # Sort by similarity score
    recs = recs.sort_values("similarity_score", ascending=False)
    
    return recs


@st.cache_data(ttl=300, show_spinner=False)
def check_data_availability() -> dict[str, bool]:
    """
    Check which data files are available.
    Returns dict with file availability status.
    
    Cached for 5 minutes to avoid repeated S3 HEAD requests.
    """
    use_local = os.getenv("USE_LOCAL_DATA", "").lower() == "true"
    available = {}
    
    for name, filename in FILES.items():
        try:
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
    """Get statistics about the loaded data."""
    games = load_games()
    
    stats = {
        "total_games": len(games) if not games.empty else 0,
        "games_with_ratings": len(games[games["total_rating"].notna()]) if not games.empty else 0,
        "games_with_covers": len(games[games["cover_url"].notna()]) if not games.empty else 0,
        "total_recommendations": "N/A",
    }
    
    if include_recommendations:
        recs = load_recommendations()
        stats["total_recommendations"] = len(recs) if not recs.empty else 0
    
    return stats
