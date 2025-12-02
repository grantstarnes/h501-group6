"""
Data Pipeline V2: Process IGDB data and generate recommendation artifacts.

Run this locally on your machine with GPU to:
1. Load all CSV files from igdb_data/
2. Merge and clean data
3. Generate text embeddings using Sentence Transformers
4. Compute content-based similarity features
5. Pre-compute top-N recommendations for all games
6. Save artifacts locally (manually upload to S3)

Requirements (local):
    pip install pandas numpy sentence-transformers scikit-learn pyarrow tqdm

Usage:
    python data_pipeline_V2.py                    # Process only (saves to processed_data/)
    python data_pipeline_V2.py --skip-embeddings  # Skip text embeddings (faster, less accurate)

After running, manually upload the files from processed_data/ to your S3 bucket:
    - games_enriched.parquet
    - recommendations.parquet  
    - embeddings.npy
    - game_id_map.parquet

=============================================================================
YOUR INPUT REQUIRED:
=============================================================================
After running this script, upload the generated files to:
    s3://igdb-streamlitapp-datasets/IGDB/processed/

You can use AWS CLI:
    aws s3 cp processed_data/ s3://igdb-streamlitapp-datasets/IGDB/processed/ --recursive

Or manually via AWS Console.
=============================================================================
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# These imports are for local processing only (not needed on Streamlit Cloud)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Text embeddings will be skipped.")
    print("Install with: pip install sentence-transformers")


# ============================================================================
# CONFIGURATION - Modify paths if needed
# ============================================================================

# Path to the igdb_data folder containing CSVs
DATA_DIR = Path(__file__).parent.parent / "igdb_data"

# Output directory for processed files
OUTPUT_DIR = Path(__file__).parent / "processed_data"

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast, good quality
TOP_N_RECOMMENDATIONS = 50  # Pre-compute top 50 for each game


def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from the data directory."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"  Warning: {filename} not found at {filepath}")
        return pd.DataFrame()
    print(f"  Loading {filename}...")
    return pd.read_csv(filepath, low_memory=False)


def load_all_data() -> dict[str, pd.DataFrame]:
    """Load all required CSV files."""
    print("\n" + "="*60)
    print("LOADING CSV FILES")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    
    data = {
        "games": load_csv("df_games_full.csv"),
        "genres": load_csv("df_genres.csv"),
        "platforms": load_csv("df_platforms.csv"),
        "covers": load_csv("df_covers_full.csv"),
        "companies": load_csv("df_involved_companies_full.csv"),
        "game_modes": load_csv("df_game_modes.csv"),
        "player_perspectives": load_csv("df_player_perspectives.csv"),
        "keywords": load_csv("df_keywords.csv"),
    }
    
    # Optional loads - wrapped in try/except to avoid crashing if files are missing
    optional_files = {
        "age_ratings": "df_age_rating_full.csv",
        "age_rating_orgs": "df_age_rating_organizations.csv",
        "language_supports": "df_language_supports_full.csv",
        "videos": "df_game_videos_full.csv",
        "game_time_to_beats": "df_game_time_to_beats_full.csv",
    }
    
    print("\nLoading optional files...")
    for name, filename in optional_files.items():
        try:
            df = load_csv(filename)
            data[name] = df
        except Exception as e:
            print(f"  Warning: Could not load {filename}: {e}")
            data[name] = pd.DataFrame()
    
    print("\nLoaded data summary:")
    for name, df in data.items():
        print(f"  {name}: {len(df):,} rows")
    
    return data


def parse_list_column(value) -> list:
    """
    Parse a column that contains pipe-separated values (e.g., "1|2|3").
    IGDB data uses | as delimiter for list columns.
    """
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (int, float)):
        return [int(value)] if not pd.isna(value) else []
    if isinstance(value, str):
        # Handle pipe-separated format: "1|2|3"
        value = value.strip()
        if not value:
            return []
        if "|" in value:
            try:
                return [int(x.strip()) for x in value.split("|") if x.strip()]
            except ValueError:
                return [x.strip() for x in value.split("|") if x.strip()]
        # Handle formats like "[1, 2, 3]" or "1,2,3"
        value = value.strip("[]").replace("'", "").replace('"', '')
        if not value:
            return []
        try:
            return [int(x.strip()) for x in value.split(",") if x.strip()]
        except ValueError:
            return [x.strip() for x in value.split(",") if x.strip()]
    return []


def build_genre_mapping(genres_df: pd.DataFrame) -> dict[int, str]:
    """Build genre ID to name mapping."""
    if genres_df.empty:
        return {}
    return dict(zip(genres_df["id"], genres_df["name"]))


def build_platform_mapping(platforms_df: pd.DataFrame) -> dict[int, str]:
    """Build platform ID to name mapping."""
    if platforms_df.empty:
        return {}
    return dict(zip(platforms_df["id"], platforms_df["name"]))


def build_game_mode_mapping(game_modes_df: pd.DataFrame) -> dict[int, str]:
    """Build game mode ID to name mapping."""
    if game_modes_df.empty:
        return {}
    return dict(zip(game_modes_df["id"], game_modes_df["name"]))


def build_player_perspective_mapping(df: pd.DataFrame) -> dict[int, str]:
    """Build player perspective ID to name mapping."""
    if df.empty or "id" not in df.columns or "name" not in df.columns:
        return {}
    return dict(zip(df["id"], df["name"]))


# IGDB language ID to name mapping (common languages)
# Full list at: https://api-docs.igdb.com/#language
LANGUAGE_ID_MAP = {
    1: "Arabic", 2: "Chinese (Simplified)", 3: "Chinese (Traditional)",
    4: "Czech", 5: "Danish", 6: "Dutch", 7: "English", 8: "Finnish",
    9: "French", 10: "German", 11: "Greek", 12: "Hebrew", 13: "Hungarian",
    14: "Italian", 15: "Japanese", 16: "Korean", 17: "Norwegian",
    18: "Polish", 19: "Portuguese (Brazil)", 20: "Portuguese (Portugal)",
    21: "Romanian", 22: "Russian", 23: "Spanish (Spain)", 24: "Spanish (Mexico)",
    25: "Swedish", 26: "Thai", 27: "Turkish", 28: "Ukrainian", 29: "Vietnamese",
    30: "Catalan", 31: "Bulgarian", 32: "Croatian", 33: "Slovak",
    34: "Indonesian", 35: "Malay",
}


def enrich_games(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge and enrich games data with related information.
    Creates a single denormalized dataframe with all game info.
    """
    print("\n" + "="*60)
    print("ENRICHING GAMES DATA")
    print("="*60)
    
    games = data["games"].copy()
    
    # Ensure 'id' column exists and is the primary key
    if "id" not in games.columns:
        print("Error: 'id' column not found in games data")
        return pd.DataFrame()
    
    print(f"Processing {len(games):,} games...")
    
    # Build mappings
    genre_map = build_genre_mapping(data["genres"])
    platform_map = build_platform_mapping(data["platforms"])
    game_mode_map = build_game_mode_mapping(data["game_modes"])
    player_perspective_map = build_player_perspective_mapping(data["player_perspectives"])
    
    print(f"  Genre mapping: {len(genre_map)} genres")
    print(f"  Platform mapping: {len(platform_map)} platforms")
    print(f"  Game mode mapping: {len(game_mode_map)} game modes")
    print(f"  Player perspective mapping: {len(player_perspective_map)} perspectives")
    
    # Parse list columns and map to names
    print("  Parsing genres...")
    games["genre_ids"] = games.get("genres", pd.Series(dtype=object)).apply(parse_list_column)
    games["genre_names"] = games["genre_ids"].apply(
        lambda ids: [genre_map.get(i, f"Unknown({i})") for i in ids if i in genre_map]
    )
    
    print("  Parsing platforms...")
    games["platform_ids"] = games.get("platforms", pd.Series(dtype=object)).apply(parse_list_column)
    games["platform_names"] = games["platform_ids"].apply(
        lambda ids: [platform_map.get(i, f"Unknown({i})") for i in ids if i in platform_map]
    )
    
    print("  Parsing game modes...")
    games["game_mode_ids"] = games.get("game_modes", pd.Series(dtype=object)).apply(parse_list_column)
    games["game_mode_names"] = games["game_mode_ids"].apply(
        lambda ids: [game_mode_map.get(i, f"Unknown({i})") for i in ids if i in game_mode_map]
    )
    
    # Parse player perspectives
    print("  Parsing player perspectives...")
    games["player_perspective_ids"] = games.get("player_perspectives", pd.Series(dtype=object)).apply(parse_list_column)
    games["player_perspective_names"] = games["player_perspective_ids"].apply(
        lambda ids: [player_perspective_map.get(i, f"Unknown({i})") for i in ids if i in player_perspective_map]
    )
    
    # Add cover URLs
    if not data["covers"].empty and "game" in data["covers"].columns:
        print("  Adding cover images...")
        covers = data["covers"][["game", "url", "image_id"]].drop_duplicates(subset=["game"])
        covers = covers.rename(columns={"game": "id", "url": "cover_url", "image_id": "cover_image_id"})
        # Fix cover URLs (IGDB returns protocol-relative URLs)
        covers["cover_url"] = covers["cover_url"].apply(
            lambda x: f"https:{x}" if pd.notna(x) and str(x).startswith("//") else x
        )
        # Make cover URLs larger (replace t_thumb with t_cover_big)
        covers["cover_url"] = covers["cover_url"].apply(
            lambda x: str(x).replace("t_thumb", "t_cover_big") if pd.notna(x) else x
        )
        games = games.merge(covers, on="id", how="left")
    
    # Add company info (developers and publishers) from involved_companies
    if not data["companies"].empty and "game" in data["companies"].columns:
        print("  Adding company info...")
        companies = data["companies"].copy()
        
        # Get developers
        if "developer" in companies.columns:
            developers = companies[companies["developer"] == True][["game", "company"]]
            developers = developers.groupby("game")["company"].apply(list).reset_index()
            developers = developers.rename(columns={"game": "id", "company": "developer_ids"})
            games = games.merge(developers, on="id", how="left")
        
        # Get publishers
        if "publisher" in companies.columns:
            publishers = companies[companies["publisher"] == True][["game", "company"]]
            publishers = publishers.groupby("game")["company"].apply(list).reset_index()
            publishers = publishers.rename(columns={"game": "id", "company": "publisher_ids"})
            games = games.merge(publishers, on="id", how="left")
    
    # Extract year from first_release_date (Unix timestamp)
    if "first_release_date" in games.columns:
        print("  Extracting release years...")
        games["release_year"] = pd.to_datetime(
            games["first_release_date"], unit="s", errors="coerce"
        ).dt.year
    
    # =========================================================================
    # AGE RATING
    # =========================================================================
    if not data["age_ratings"].empty and "id" in data["age_ratings"].columns:
        print("  Adding age ratings...")
        ar = data["age_ratings"].copy()
        # The rating column already has the label (e.g., "T", "M", "E10+")
        if "rating" in ar.columns and "organization" in ar.columns:
            # Build org ID to name mapping
            org_map = {}
            if not data.get("age_rating_orgs", pd.DataFrame()).empty:
                age_orgs = data["age_rating_orgs"]
                if "id" in age_orgs.columns and "name" in age_orgs.columns:
                    org_map = dict(zip(age_orgs["id"], age_orgs["name"]))
            
            # Create combined label like "ESRB T" or "PEGI 16"
            ar["org_name"] = ar["organization"].map(org_map).fillna("")
            ar["age_rating_label"] = ar.apply(
                lambda row: f"{row['org_name']} {row['rating']}" if row['org_name'] else str(row['rating']),
                axis=1
            )
            ar = ar[["id", "age_rating_label"]].dropna()
            ar = ar.drop_duplicates(subset=["id"], keep="first")
            ar = ar.rename(columns={"id": "age_rating_id", "age_rating_label": "age_rating"})
            
            # Parse the age_ratings column from games (it contains rating IDs)
            games["age_rating_ids"] = games.get("age_ratings", pd.Series(dtype=object)).apply(parse_list_column)
            
            # Get the first rating for each game (prefer ESRB - org 1)
            def get_first_age_rating(rating_ids):
                if not rating_ids:
                    return None
                for rid in rating_ids:
                    if rid in ar["age_rating_id"].values:
                        return ar[ar["age_rating_id"] == rid]["age_rating"].iloc[0]
                return None
            
            games["age_rating"] = games["age_rating_ids"].apply(get_first_age_rating)
    
    # =========================================================================
    # SUPPORTED LANGUAGES
    # =========================================================================
    if not data["language_supports"].empty and "game" in data["language_supports"].columns:
        print("  Adding language support...")
        ls = data["language_supports"][["game", "language"]].dropna()
        ls["language_name"] = ls["language"].map(LANGUAGE_ID_MAP)
        ls = ls[ls["language_name"].notna()]
        # Get unique languages per game
        grouped = ls.groupby("game")["language_name"].apply(lambda x: list(set(x))).reset_index()
        grouped = grouped.rename(columns={"game": "id", "language_name": "language_names"})
        games = games.merge(grouped, on="id", how="left")
    
    # =========================================================================
    # TOP 2 VIDEO URLS
    # =========================================================================
    if not data["videos"].empty and "game" in data["videos"].columns:
        print("  Adding video links...")
        vids = data["videos"].copy()
        # Build YouTube URLs from video_id
        if "video_id" in vids.columns:
            vids["video_url"] = "https://www.youtube.com/watch?v=" + vids["video_id"].astype(str)
        elif "url" in vids.columns:
            vids["video_url"] = vids["url"]
        else:
            vids["video_url"] = None
        
        vids = vids[["game", "video_url"]].dropna()
        grouped = vids.groupby("game")["video_url"].apply(list).reset_index()
        grouped = grouped.rename(columns={"game": "id", "video_url": "video_urls"})
        games = games.merge(grouped, on="id", how="left")
        
        # Trim to top 2 videos
        if "video_urls" in games.columns:
            games["video_urls"] = games["video_urls"].apply(
                lambda lst: lst[:2] if isinstance(lst, list) else []
            )
    
    # =========================================================================
    # CANONICAL/PARENT GAME ID (for deduplication)
    # =========================================================================
    print("  Computing canonical game IDs...")
    if "version_parent" in games.columns:
        games["canonical_game_id"] = games["version_parent"].fillna(games["id"]).astype(int)
    elif "parent_game" in games.columns:
        games["canonical_game_id"] = games["parent_game"].fillna(games["id"]).astype(int)
    else:
        games["canonical_game_id"] = games["id"].astype(int)
    
    # Compute canonical release year (minimum release year per canonical id)
    print("  Computing canonical release year...")
    if "release_year" in games.columns:
        tmp = games[["canonical_game_id", "release_year"]].dropna()
        if not tmp.empty:
            grp = tmp.groupby("canonical_game_id")["release_year"].min().reset_index()
            grp = grp.rename(columns={"release_year": "canonical_release_year"})
            games = games.merge(grp, on="canonical_game_id", how="left")
        else:
            games["canonical_release_year"] = games.get("release_year")
    else:
        games["canonical_release_year"] = None
    
    # =========================================================================
    # TIME-TO-BEAT METRICS
    # =========================================================================
    if not data["game_time_to_beats"].empty:
        print("  Adding time-to-beat metrics...")
        ttb = data["game_time_to_beats"].copy()
        
        # TTB values are in seconds - convert to hours
        ttb_cols = ["hastily", "normally", "completely"]
        for col in ttb_cols:
            if col in ttb.columns:
                ttb[col] = pd.to_numeric(ttb[col], errors="coerce") / 3600  # seconds to hours
        
        # Handle multiple entries per game by taking median
        if "game_id" in ttb.columns:
            agg_cols = [c for c in ttb_cols if c in ttb.columns]
            if agg_cols:
                ttb = ttb.groupby("game_id")[agg_cols].median().reset_index()
                ttb = ttb.rename(columns={
                    "game_id": "id",
                    "hastily": "ttb_hastily",
                    "normally": "ttb_normally",
                    "completely": "ttb_completely",
                })
                games = games.merge(ttb, on="id", how="left")
    
    # Fill NaN values for list columns
    list_columns = ["genre_ids", "genre_names", "platform_ids", "platform_names", 
                    "game_mode_ids", "game_mode_names", "developer_ids", "publisher_ids",
                    "player_perspective_ids", "player_perspective_names",
                    "language_names", "video_urls"]
    for col in list_columns:
        if col in games.columns:
            games[col] = games[col].apply(lambda x: x if isinstance(x, list) else [])
    
    # Create text for embedding (combine name, summary, storyline)
    print("  Creating text for embeddings...")
    games["text_for_embedding"] = (
        games.get("name", pd.Series(dtype=str)).fillna("").astype(str) + " " +
        games.get("summary", pd.Series(dtype=str)).fillna("").astype(str) + " " +
        games.get("storyline", pd.Series(dtype=str)).fillna("").astype(str)
    ).str.strip()
    
    # Replace empty strings with a placeholder
    games["text_for_embedding"] = games["text_for_embedding"].replace("", "Unknown game")
    
    print(f"  Enriched {len(games):,} games")
    return games


def compute_embeddings(games: pd.DataFrame, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """
    Compute text embeddings for all games using Sentence Transformers.
    Uses GPU if available.
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        print("\nSkipping embeddings (sentence-transformers not installed)")
        print("Install with: pip install sentence-transformers")
        return np.zeros((len(games), 384), dtype=np.float32)
    
    print("\n" + "="*60)
    print("COMPUTING TEXT EMBEDDINGS")
    print("="*60)
    print(f"Model: {model_name}")
    
    model = SentenceTransformer(model_name)
    
    # Check if GPU is available
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Get texts
    texts = games["text_for_embedding"].fillna("Unknown game").tolist()
    
    # Compute embeddings in batches
    print(f"Encoding {len(texts):,} texts...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # For cosine similarity
        device=device
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def build_feature_matrix(games: pd.DataFrame, genres_df: pd.DataFrame, 
                         platforms_df: pd.DataFrame) -> np.ndarray:
    """
    Build a feature matrix from categorical features (genres, platforms, game modes).
    Uses multi-hot encoding.
    """
    print("\n" + "="*60)
    print("BUILDING FEATURE MATRIX")
    print("="*60)
    
    # Get all unique genre and platform IDs
    all_genre_ids = sorted(genres_df["id"].unique()) if not genres_df.empty else []
    all_platform_ids = sorted(platforms_df["id"].unique()) if not platforms_df.empty else []
    
    n_games = len(games)
    n_genres = len(all_genre_ids)
    n_platforms = len(all_platform_ids)
    
    print(f"Games: {n_games:,}")
    print(f"Genres: {n_genres}")
    print(f"Platforms: {n_platforms}")
    
    # Create ID to index mappings
    genre_to_idx = {gid: idx for idx, gid in enumerate(all_genre_ids)}
    platform_to_idx = {pid: idx for idx, pid in enumerate(all_platform_ids)}
    
    # Build multi-hot encoded matrix
    genre_matrix = np.zeros((n_games, n_genres), dtype=np.float32)
    platform_matrix = np.zeros((n_games, n_platforms), dtype=np.float32)
    
    print("Encoding features...")
    games_reset = games.reset_index(drop=True)
    for i, row in tqdm(games_reset.iterrows(), total=n_games, desc="Encoding"):
        # Genres
        for gid in row.get("genre_ids", []):
            if gid in genre_to_idx:
                genre_matrix[i, genre_to_idx[gid]] = 1.0
        
        # Platforms
        for pid in row.get("platform_ids", []):
            if pid in platform_to_idx:
                platform_matrix[i, platform_to_idx[pid]] = 1.0
    
    # Add numeric features (normalized)
    numeric_features = []
    for col in ["rating", "total_rating", "aggregated_rating"]:
        if col in games.columns:
            values = pd.to_numeric(games[col], errors='coerce').fillna(0).values.astype(np.float32)
            # Normalize to 0-1 range
            max_val = values.max()
            if max_val > 0:
                values = values / max_val
            numeric_features.append(values.reshape(-1, 1))
    
    if numeric_features:
        numeric_matrix = np.hstack(numeric_features)
    else:
        numeric_matrix = np.zeros((n_games, 1), dtype=np.float32)
    
    # Combine all features with weights
    # Weight: genres (0.4) + platforms (0.2) + numeric (0.1)
    feature_matrix = np.hstack([
        genre_matrix * 0.4,
        platform_matrix * 0.2,
        numeric_matrix * 0.1
    ])
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    return feature_matrix


def compute_recommendations(
    games: pd.DataFrame,
    embeddings: np.ndarray,
    feature_matrix: np.ndarray,
    top_n: int = TOP_N_RECOMMENDATIONS
) -> pd.DataFrame:
    """
    Compute top-N recommendations for every game.
    
    Combines:
    - Text embeddings (semantic similarity) - 60% weight
    - Feature matrix (genre/platform/numeric similarity) - 40% weight
    
    Returns DataFrame with columns: [game_id, recommended_ids, scores]
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    
    print("\n" + "="*60)
    print("COMPUTING RECOMMENDATIONS")
    print("="*60)
    print(f"Computing top-{top_n} recommendations for {len(games):,} games...")
    
    # Combine embeddings and features with weights
    print("Combining embeddings with features...")
    # Embeddings already normalized, features need normalization
    feature_normalized = normalize(feature_matrix, axis=1)
    
    # Weighted combination: 60% text, 40% features
    combined = np.hstack([
        embeddings * 0.6,
        feature_normalized * 0.4
    ])
    combined = normalize(combined, axis=1)
    
    print(f"Combined feature shape: {combined.shape}")
    
    # Compute similarity in batches (full 34K x 34K is ~4.6GB)
    print("Computing similarity matrix in batches...")
    game_ids = games["id"].values
    n_games = len(games)
    batch_size = 500  # Process 500 games at a time to manage memory
    
    recommendations = []
    
    for start_idx in tqdm(range(0, n_games, batch_size), desc="Computing"):
        end_idx = min(start_idx + batch_size, n_games)
        batch = combined[start_idx:end_idx]
        
        # Compute similarity between this batch and all games
        sim_batch = cosine_similarity(batch, combined)
        
        # For each game in batch, get top-N (excluding self)
        for i, game_idx in enumerate(range(start_idx, end_idx)):
            sim_scores = sim_batch[i]
            
            # Get indices of top-N+1 (including self), then exclude self
            top_indices = np.argsort(sim_scores)[::-1][:top_n + 5]  # Get extra in case of ties
            top_indices = [idx for idx in top_indices if idx != game_idx][:top_n]
            
            rec_ids = game_ids[top_indices].tolist()
            rec_scores = [float(sim_scores[idx]) for idx in top_indices]
            
            recommendations.append({
                "game_id": int(game_ids[game_idx]),
                "recommended_ids": rec_ids,
                "scores": rec_scores
            })
    
    rec_df = pd.DataFrame(recommendations)
    print(f"Generated recommendations for {len(rec_df):,} games")
    return rec_df


def save_artifacts(
    games: pd.DataFrame,
    recommendations: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: Path
) -> dict[str, Path]:
    """Save all artifacts to local files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("SAVING ARTIFACTS")
    print("="*60)
    print(f"Output directory: {output_dir}")
    
    paths = {}
    
    # Select columns to save (drop text_for_embedding to reduce size)
    columns_to_keep = [
        "id", "name", "summary", "storyline", 
        "rating", "total_rating", "aggregated_rating",
        "rating_count", "total_rating_count", "aggregated_rating_count",
        "first_release_date", "release_year",
        "genre_ids", "genre_names", 
        "platform_ids", "platform_names",
        "game_mode_ids", "game_mode_names",
        "developer_ids", "publisher_ids",
        "cover_url", "cover_image_id",
        "follows", "hypes", "url", "slug",
        # New columns added for V2 enrichment
        "player_perspective_ids", "player_perspective_names",
        "age_rating",
        "language_names",
        "video_urls",
        "canonical_game_id", "canonical_release_year",
        "ttb_hastily", "ttb_normally", "ttb_completely",
    ]
    columns_to_save = [c for c in columns_to_keep if c in games.columns]
    
    # Save enriched games
    games_path = output_dir / "games_enriched.parquet"
    games_to_save = games[columns_to_save].copy()
    games_to_save.to_parquet(games_path, index=False)
    paths["games"] = games_path
    print(f"  games_enriched.parquet: {games_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save recommendations
    rec_path = output_dir / "recommendations.parquet"
    recommendations.to_parquet(rec_path, index=False)
    paths["recommendations"] = rec_path
    print(f"  recommendations.parquet: {rec_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save embeddings (for Option B real-time similarity)
    emb_path = output_dir / "embeddings.npy"
    np.save(emb_path, embeddings.astype(np.float32))
    paths["embeddings"] = emb_path
    print(f"  embeddings.npy: {emb_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save game ID to index mapping (for embeddings lookup)
    id_map_path = output_dir / "game_id_map.parquet"
    id_map = pd.DataFrame({
        "game_id": games["id"].values,
        "index": range(len(games))
    })
    id_map.to_parquet(id_map_path, index=False)
    paths["id_map"] = id_map_path
    print(f"  game_id_map.parquet: {id_map_path.stat().st_size / 1024:.1f} KB")
    
    return paths


def main():
    parser = argparse.ArgumentParser(description="Process IGDB data and generate recommendations")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip text embedding computation")
    parser.add_argument("--top-n", type=int, default=TOP_N_RECOMMENDATIONS, help="Number of recommendations per game")
    args = parser.parse_args()
    
    print("="*60)
    print("IGDB DATA PIPELINE V2")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Top-N recommendations: {args.top_n}")
    print(f"Skip embeddings: {args.skip_embeddings}")
    
    # Load data
    data = load_all_data()
    
    if data["games"].empty:
        print("\nError: No games data found!")
        print(f"Expected location: {DATA_DIR / 'df_games_full.csv'}")
        return
    
    # Enrich games
    games = enrich_games(data)
    
    if games.empty:
        print("\nError: Failed to enrich games data")
        return
    
    # Reset index for consistent indexing
    games = games.reset_index(drop=True)
    
    # Compute embeddings
    if args.skip_embeddings:
        print("\nSkipping embeddings as requested")
        embeddings = np.zeros((len(games), 384), dtype=np.float32)
    else:
        embeddings = compute_embeddings(games)
    
    # Build feature matrix
    feature_matrix = build_feature_matrix(games, data["genres"], data["platforms"])
    
    # Compute recommendations
    recommendations = compute_recommendations(games, embeddings, feature_matrix, args.top_n)
    
    # Save artifacts
    paths = save_artifacts(games, recommendations, embeddings, args.output_dir)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nGenerated files:")
    for name, path in paths.items():
        print(f"  {path}")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("""
1. Upload the files from processed_data/ to your S3 bucket:

   Using AWS CLI:
   aws s3 cp processed_data/ s3://igdb-streamlitapp-datasets/IGDB/processed/ --recursive

   Or manually via AWS Console:
   - Go to S3 → igdb-streamlitapp-datasets → IGDB/
   - Create folder 'processed' if it doesn't exist
   - Upload all files from processed_data/

2. Configure S3 bucket for public read access (see s3_loader_V2.py for bucket policy)

3. Deploy streamlit_app_V2.py to Streamlit Community Cloud
""")


if __name__ == "__main__":
    main()
