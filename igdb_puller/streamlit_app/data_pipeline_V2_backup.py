"""
Data Pipeline V2 Backup: Generate SMALLER dataset for S3 Demo.

This is a simplified version that ONLY includes games with ratings (~33K games).
This reduces file sizes dramatically for Streamlit Cloud deployment:
- games_enriched_S3Implementation.parquet (~8-10 MB vs ~90 MB)
- recommendations_S3Implementation.parquet (~30-40 MB vs ~167 MB)

Output files have "_S3Implementation" suffix to distinguish from full dataset.

Usage:
    python data_pipeline_V2_backup.py                    # Full processing
    python data_pipeline_V2_backup.py --skip-embeddings  # Skip text embeddings

After running, upload to S3:
    aws s3 cp processed_data_backup/ s3://igdb-streamlitapp-datasets/processed/ --recursive

=============================================================================
TWO IMPLEMENTATIONS:
=============================================================================

1. S3 Demo (this pipeline):
   - ~33K games with ratings
   - Small file sizes for reliable cloud deployment
   - Files: *_S3Implementation.parquet
   - Used by: streamlit_app_V2_backup.py

2. Full Offline (main pipeline):
   - ~220K games (year >= 2000)
   - Larger files, requires local or powerful cloud
   - Files: games_enriched.parquet, recommendations.parquet
   - Used by: streamlit_app_V2.py

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
# CONFIGURATION
# ============================================================================

# Path to the igdb_data folder containing CSVs
DATA_DIR = Path(__file__).parent.parent / "igdb_data"

# Output directory for backup processed files
OUTPUT_DIR = Path(__file__).parent / "processed_data_backup"

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast, good quality
TOP_N_RECOMMENDATIONS = 50  # Pre-compute top 50 for each game

# File suffix for S3 implementation
FILE_SUFFIX = "_S3Implementation"

# CRITICAL FILTER: Only games with ratings
# This reduces ~343K games to ~33K games with actual rating data
MIN_RATING_COUNT = 1  # Must have at least 1 rating


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
    
    # Optional loads
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
    """Parse a column that contains pipe-separated values."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (int, float)):
        return [int(value)] if not pd.isna(value) else []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if "|" in value:
            try:
                return [int(x.strip()) for x in value.split("|") if x.strip()]
            except ValueError:
                return [x.strip() for x in value.split("|") if x.strip()]
        value = value.strip("[]").replace("'", "").replace('"', '')
        if not value:
            return []
        try:
            return [int(x.strip()) for x in value.split(",") if x.strip()]
        except ValueError:
            return [x.strip() for x in value.split(",") if x.strip()]
    return []


def build_genre_mapping(genres_df: pd.DataFrame) -> dict[int, str]:
    if genres_df.empty:
        return {}
    return dict(zip(genres_df["id"], genres_df["name"]))


def build_platform_mapping(platforms_df: pd.DataFrame) -> dict[int, str]:
    if platforms_df.empty:
        return {}
    return dict(zip(platforms_df["id"], platforms_df["name"]))


def build_game_mode_mapping(game_modes_df: pd.DataFrame) -> dict[int, str]:
    if game_modes_df.empty:
        return {}
    return dict(zip(game_modes_df["id"], game_modes_df["name"]))


def build_player_perspective_mapping(df: pd.DataFrame) -> dict[int, str]:
    if df.empty or "id" not in df.columns or "name" not in df.columns:
        return {}
    return dict(zip(df["id"], df["name"]))


# IGDB language ID to name mapping
LANGUAGE_ID_MAP = {
    1: "Arabic", 2: "Chinese (Simplified)", 3: "Chinese (Traditional)",
    4: "Czech", 5: "Danish", 6: "Dutch", 7: "English", 8: "Finnish",
    9: "French", 10: "German", 11: "Greek", 12: "Hebrew", 13: "Hungarian",
    14: "Italian", 15: "Japanese", 16: "Korean", 17: "Norwegian",
    18: "Polish", 19: "Portuguese (Brazil)", 20: "Portuguese (Portugal)",
    21: "Romanian", 22: "Russian", 23: "Spanish (Spain)", 24: "Spanish (Mexico)",
    25: "Swedish", 26: "Thai", 27: "Turkish", 28: "Ukrainian", 29: "Vietnamese",
}


def enrich_games(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge and enrich games data with related information."""
    print("\n" + "="*60)
    print("ENRICHING GAMES DATA")
    print("="*60)
    
    games = data["games"].copy()
    
    if "id" not in games.columns:
        print("Error: 'id' column not found in games data")
        return pd.DataFrame()
    
    print(f"Processing {len(games):,} games...")
    
    # Build mappings
    genre_map = build_genre_mapping(data["genres"])
    platform_map = build_platform_mapping(data["platforms"])
    game_mode_map = build_game_mode_mapping(data["game_modes"])
    player_perspective_map = build_player_perspective_mapping(data["player_perspectives"])
    
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
        covers["cover_url"] = covers["cover_url"].apply(
            lambda x: f"https:{x}" if pd.notna(x) and str(x).startswith("//") else x
        )
        covers["cover_url"] = covers["cover_url"].apply(
            lambda x: str(x).replace("t_thumb", "t_cover_big") if pd.notna(x) else x
        )
        games = games.merge(covers, on="id", how="left")
    
    # Add company info
    if not data["companies"].empty and "game" in data["companies"].columns:
        print("  Adding company info...")
        companies = data["companies"].copy()
        
        if "developer" in companies.columns:
            developers = companies[companies["developer"] == True][["game", "company"]]
            developers = developers.groupby("game")["company"].apply(list).reset_index()
            developers = developers.rename(columns={"game": "id", "company": "developer_ids"})
            games = games.merge(developers, on="id", how="left")
        
        if "publisher" in companies.columns:
            publishers = companies[companies["publisher"] == True][["game", "company"]]
            publishers = publishers.groupby("game")["company"].apply(list).reset_index()
            publishers = publishers.rename(columns={"game": "id", "company": "publisher_ids"})
            games = games.merge(publishers, on="id", how="left")
    
    # Extract year from first_release_date
    if "first_release_date" in games.columns:
        print("  Extracting release years...")
        games["release_year"] = pd.to_datetime(
            games["first_release_date"], unit="s", errors="coerce"
        ).dt.year
    
    # Age ratings (simplified)
    if not data["age_ratings"].empty and "id" in data["age_ratings"].columns:
        print("  Adding age ratings...")
        ar = data["age_ratings"].copy()
        if "rating" in ar.columns:
            ar = ar[["id", "rating"]].drop_duplicates(subset=["id"], keep="first")
            ar = ar.rename(columns={"id": "age_rating_id", "rating": "age_rating"})
            games["age_rating_ids"] = games.get("age_ratings", pd.Series(dtype=object)).apply(parse_list_column)
            
            def get_first_age_rating(rating_ids):
                if not rating_ids:
                    return None
                for rid in rating_ids:
                    if rid in ar["age_rating_id"].values:
                        return ar[ar["age_rating_id"] == rid]["age_rating"].iloc[0]
                return None
            
            games["age_rating"] = games["age_rating_ids"].apply(get_first_age_rating)
    
    # Supported languages
    if not data["language_supports"].empty and "game" in data["language_supports"].columns:
        print("  Adding language support...")
        ls = data["language_supports"][["game", "language"]].dropna()
        ls["language_name"] = ls["language"].map(LANGUAGE_ID_MAP)
        ls = ls[ls["language_name"].notna()]
        grouped = ls.groupby("game")["language_name"].apply(lambda x: list(set(x))).reset_index()
        grouped = grouped.rename(columns={"game": "id", "language_name": "language_names"})
        games = games.merge(grouped, on="id", how="left")
    
    # Video URLs
    if not data["videos"].empty and "game" in data["videos"].columns:
        print("  Adding video links...")
        vids = data["videos"].copy()
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
        
        if "video_urls" in games.columns:
            games["video_urls"] = games["video_urls"].apply(
                lambda lst: lst[:2] if isinstance(lst, list) else []
            )
    
    # Canonical game ID
    print("  Computing canonical game IDs...")
    if "version_parent" in games.columns:
        games["canonical_game_id"] = games["version_parent"].fillna(games["id"]).astype(int)
    elif "parent_game" in games.columns:
        games["canonical_game_id"] = games["parent_game"].fillna(games["id"]).astype(int)
    else:
        games["canonical_game_id"] = games["id"].astype(int)
    
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
    
    # Time-to-beat metrics
    if not data["game_time_to_beats"].empty:
        print("  Adding time-to-beat metrics...")
        ttb = data["game_time_to_beats"].copy()
        ttb_cols = ["hastily", "normally", "completely"]
        for col in ttb_cols:
            if col in ttb.columns:
                ttb[col] = pd.to_numeric(ttb[col], errors="coerce") / 3600
        
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
    
    # Create text for embedding
    print("  Creating text for embeddings...")
    games["text_for_embedding"] = (
        games.get("name", pd.Series(dtype=str)).fillna("").astype(str) + " " +
        games.get("summary", pd.Series(dtype=str)).fillna("").astype(str) + " " +
        games.get("storyline", pd.Series(dtype=str)).fillna("").astype(str)
    ).str.strip()
    games["text_for_embedding"] = games["text_for_embedding"].replace("", "Unknown game")
    
    print(f"  Enriched {len(games):,} games")
    return games


def compute_embeddings(games: pd.DataFrame, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Compute text embeddings for all games using Sentence Transformers."""
    if not HAS_SENTENCE_TRANSFORMERS:
        print("\nSkipping embeddings (sentence-transformers not installed)")
        return np.zeros((len(games), 384), dtype=np.float32)
    
    print("\n" + "="*60)
    print("COMPUTING TEXT EMBEDDINGS")
    print("="*60)
    print(f"Model: {model_name}")
    
    model = SentenceTransformer(model_name)
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    texts = games["text_for_embedding"].fillna("Unknown game").tolist()
    
    print(f"Encoding {len(texts):,} texts...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def build_feature_matrix(games: pd.DataFrame, genres_df: pd.DataFrame, 
                         platforms_df: pd.DataFrame) -> np.ndarray:
    """Build a feature matrix from categorical features."""
    print("\n" + "="*60)
    print("BUILDING FEATURE MATRIX")
    print("="*60)
    
    all_genre_ids = sorted(genres_df["id"].unique()) if not genres_df.empty else []
    all_platform_ids = sorted(platforms_df["id"].unique()) if not platforms_df.empty else []
    
    n_games = len(games)
    n_genres = len(all_genre_ids)
    n_platforms = len(all_platform_ids)
    
    print(f"Games: {n_games:,}")
    print(f"Genres: {n_genres}")
    print(f"Platforms: {n_platforms}")
    
    genre_to_idx = {gid: idx for idx, gid in enumerate(all_genre_ids)}
    platform_to_idx = {pid: idx for idx, pid in enumerate(all_platform_ids)}
    
    genre_matrix = np.zeros((n_games, n_genres), dtype=np.float32)
    platform_matrix = np.zeros((n_games, n_platforms), dtype=np.float32)
    
    print("Encoding features...")
    games_reset = games.reset_index(drop=True)
    for i, row in tqdm(games_reset.iterrows(), total=n_games, desc="Encoding"):
        for gid in row.get("genre_ids", []):
            if gid in genre_to_idx:
                genre_matrix[i, genre_to_idx[gid]] = 1.0
        for pid in row.get("platform_ids", []):
            if pid in platform_to_idx:
                platform_matrix[i, platform_to_idx[pid]] = 1.0
    
    # Numeric features
    numeric_features = []
    for col in ["rating", "total_rating", "aggregated_rating"]:
        if col in games.columns:
            values = pd.to_numeric(games[col], errors='coerce').fillna(0).values.astype(np.float32)
            max_val = values.max()
            if max_val > 0:
                values = values / max_val
            numeric_features.append(values.reshape(-1, 1))
    
    if numeric_features:
        numeric_matrix = np.hstack(numeric_features)
    else:
        numeric_matrix = np.zeros((n_games, 1), dtype=np.float32)
    
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
    """Compute top-N recommendations for every game."""
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    
    print("\n" + "="*60)
    print("COMPUTING RECOMMENDATIONS")
    print("="*60)
    print(f"Computing top-{top_n} recommendations for {len(games):,} games...")
    
    print("Combining embeddings with features...")
    feature_normalized = normalize(feature_matrix, axis=1)
    
    combined = np.hstack([
        embeddings * 0.6,
        feature_normalized * 0.4
    ])
    combined = normalize(combined, axis=1)
    
    print(f"Combined feature shape: {combined.shape}")
    print("Computing similarity matrix in batches...")
    
    game_ids = games["id"].values
    n_games = len(games)
    batch_size = 500
    
    recommendations = []
    
    for start_idx in tqdm(range(0, n_games, batch_size), desc="Computing"):
        end_idx = min(start_idx + batch_size, n_games)
        batch = combined[start_idx:end_idx]
        sim_batch = cosine_similarity(batch, combined)
        
        for i, game_idx in enumerate(range(start_idx, end_idx)):
            sim_scores = sim_batch[i]
            top_indices = np.argsort(sim_scores)[::-1][:top_n + 5]
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
    """Save all artifacts with _S3Implementation suffix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("SAVING ARTIFACTS (S3 IMPLEMENTATION)")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"File suffix: {FILE_SUFFIX}")
    
    paths = {}
    
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
        "player_perspective_ids", "player_perspective_names",
        "age_rating",
        "language_names",
        "video_urls",
        "canonical_game_id", "canonical_release_year",
        "ttb_hastily", "ttb_normally", "ttb_completely",
    ]
    columns_to_save = [c for c in columns_to_keep if c in games.columns]
    
    # Save enriched games with _S3Implementation suffix
    games_path = output_dir / f"games_enriched{FILE_SUFFIX}.parquet"
    games_to_save = games[columns_to_save].copy()
    games_to_save.to_parquet(games_path, index=False)
    paths["games"] = games_path
    print(f"  games_enriched{FILE_SUFFIX}.parquet: {games_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save recommendations with _S3Implementation suffix
    rec_path = output_dir / f"recommendations{FILE_SUFFIX}.parquet"
    recommendations.to_parquet(rec_path, index=False)
    paths["recommendations"] = rec_path
    print(f"  recommendations{FILE_SUFFIX}.parquet: {rec_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save embeddings with _S3Implementation suffix
    emb_path = output_dir / f"embeddings{FILE_SUFFIX}.npy"
    np.save(emb_path, embeddings.astype(np.float32))
    paths["embeddings"] = emb_path
    print(f"  embeddings{FILE_SUFFIX}.npy: {emb_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save game ID to index mapping with _S3Implementation suffix
    id_map_path = output_dir / f"game_id_map{FILE_SUFFIX}.parquet"
    id_map = pd.DataFrame({
        "game_id": games["id"].values,
        "index": range(len(games))
    })
    id_map.to_parquet(id_map_path, index=False)
    paths["id_map"] = id_map_path
    print(f"  game_id_map{FILE_SUFFIX}.parquet: {id_map_path.stat().st_size / 1024:.1f} KB")
    
    return paths


def main():
    parser = argparse.ArgumentParser(description="Process IGDB data - S3 Demo (rated games only)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip text embedding computation")
    parser.add_argument("--top-n", type=int, default=TOP_N_RECOMMENDATIONS, help="Number of recommendations per game")
    args = parser.parse_args()
    
    print("="*60)
    print("IGDB DATA PIPELINE V2 - BACKUP (S3 DEMO)")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Top-N recommendations: {args.top_n}")
    print(f"Skip embeddings: {args.skip_embeddings}")
    print(f"File suffix: {FILE_SUFFIX}")
    print("")
    print("This creates SMALLER files with only RATED games (~33K)")
    print("for reliable Streamlit Cloud deployment.")
    
    # Load data
    data = load_all_data()
    
    if data["games"].empty:
        print("\nError: No games data found!")
        return
    
    # Enrich games
    games = enrich_games(data)
    
    if games.empty:
        print("\nError: Failed to enrich games data")
        return
    
    # =========================================================================
    # CRITICAL FILTER: Only games with ratings
    # =========================================================================
    print("\n" + "="*60)
    print("FILTERING TO GAMES WITH RATINGS ONLY")
    print("="*60)
    
    pre_filter_count = len(games)
    
    # Filter to games that have at least one rating metric
    rating_cols = ["total_rating_count", "rating_count", "aggregated_rating_count"]
    has_rating = pd.Series([False] * len(games), index=games.index)
    
    for col in rating_cols:
        if col in games.columns:
            has_rating = has_rating | (pd.to_numeric(games[col], errors='coerce').fillna(0) >= MIN_RATING_COUNT)
    
    games = games[has_rating].copy()
    post_filter_count = len(games)
    
    print(f"  Pre-filter: {pre_filter_count:,} games")
    print(f"  Post-filter: {post_filter_count:,} games WITH RATINGS")
    print(f"  Removed: {pre_filter_count - post_filter_count:,} games without ratings")
    print(f"  Size reduction: {(1 - post_filter_count/pre_filter_count)*100:.1f}%")
    
    # Reset index
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
    print("\nGenerated files (S3 DEMO - small files):")
    for name, path in paths.items():
        print(f"  {path}")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"""
1. Upload the files from {args.output_dir}/ to your S3 bucket:

   Using AWS CLI:
   aws s3 cp {args.output_dir}/ s3://igdb-streamlitapp-datasets/processed/ --recursive

2. The backup Streamlit app (streamlit_app_V2_backup.py) will read these files
   with the {FILE_SUFFIX} suffix.

3. Deploy streamlit_app_V2_backup.py to Streamlit Community Cloud
""")


if __name__ == "__main__":
    main()
