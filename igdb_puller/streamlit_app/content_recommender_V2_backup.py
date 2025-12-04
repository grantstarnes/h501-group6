"""Content Recommender V2 Backup: Recommendation utilities for S3 demo app.

This version uses s3_loader_V2_backup which reads the smaller
_S3Implementation.parquet files (~33K games with ratings).
"""

import numpy as np
import pandas as pd
from typing import Optional
import time

# Import BACKUP S3 loader functions
from s3_loader_V2_backup import (
    get_recommendations_for_game,
    load_games
)


def get_recommendations(
    game_id: int, 
    top_n: int = 5, 
    method: str = "precomputed"
) -> tuple[pd.DataFrame, float]:
    """
    Get pre-computed recommendations for a game.
    
    Args:
        game_id: The game ID to get recommendations for
        top_n: Number of recommendations (default 5)
        method: Ignored, always uses precomputed recommendations
    
    Returns:
        tuple: (recommendations DataFrame, time_taken in seconds)
    """
    start_time = time.time()
    
    # Always use precomputed recommendations
    recs = get_recommendations_for_game(game_id, top_n)
    
    elapsed = time.time() - start_time
    return recs, elapsed


def format_recommendation_card(game: pd.Series) -> dict:
    """
    Format a game row for display as a recommendation card.
    
    Returns dict with display-ready values.
    """
    def parse_list(val):
        import numpy as np
        if val is None:
            return []
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                import ast
                return ast.literal_eval(val)
            except:
                return []
        return []
    
    genres = parse_list(game.get("genre_names", []))
    platforms = parse_list(game.get("platform_names", []))
    
    rating = game.get("rating") or game.get("total_rating")
    rating_str = f"{float(rating):.1f}/100" if pd.notna(rating) else "N/A"
    
    score = game.get("similarity_score", 0)
    score_str = f"{float(score):.1%}" if pd.notna(score) and score > 0 else ""
    
    cover_url = game.get("cover_url", "")
    if pd.isna(cover_url) or not cover_url:
        cover_url = "https://via.placeholder.com/264x374?text=No+Cover"
    
    year = game.get("release_year")
    year_str = str(int(year)) if pd.notna(year) else "Unknown"
    
    summary = str(game.get("summary", "")) if pd.notna(game.get("summary")) else ""
    if len(summary) > 200:
        summary = summary[:200] + "..."
    
    return {
        "id": int(game.get("id", 0)),
        "name": str(game.get("name", "Unknown Game")),
        "cover_url": cover_url,
        "rating": rating_str,
        "year": year_str,
        "genres": genres[:3],
        "platforms": platforms[:5],
        "similarity": score_str,
        "summary": summary,
    }


def get_game_display_info(game_id: int) -> Optional[dict]:
    """
    Get full display info for a game.
    
    Returns dict with all fields needed for the details page.
    """
    games = load_games()
    if games.empty:
        return None
    
    game = games[games["id"] == game_id]
    if game.empty:
        return None
    
    game = game.iloc[0]
    
    def parse_list(val):
        import numpy as np
        if val is None:
            return []
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                import ast
                return ast.literal_eval(val)
            except:
                return []
        return []
    
    genres = parse_list(game.get("genre_names", []))
    platforms = parse_list(game.get("platform_names", []))
    game_modes = parse_list(game.get("game_mode_names", []))
    player_perspectives = parse_list(game.get("player_perspective_names", []))
    languages = parse_list(game.get("language_names", []))
    videos = parse_list(game.get("video_urls", []))
    
    cover_url = game.get("cover_url", "")
    if pd.isna(cover_url) or not cover_url:
        cover_url = "https://via.placeholder.com/264x374?text=No+Cover"
    
    rating = game.get("rating")
    total_rating = game.get("total_rating")
    rating_count = game.get("total_rating_count", 0)
    
    year = game.get("release_year")
    
    summary = game.get("summary", "")
    if pd.isna(summary):
        summary = "No summary available."
    
    storyline = game.get("storyline", "")
    if pd.isna(storyline):
        storyline = ""
    
    age_rating = game.get("age_rating", "")
    age_rating_str = str(age_rating) if pd.notna(age_rating) and age_rating else "N/A"
    
    ttb_hastily = game.get("ttb_hastily")
    ttb_normally = game.get("ttb_normally")
    ttb_completely = game.get("ttb_completely")
    
    return {
        "id": int(game.get("id", 0)),
        "name": str(game.get("name", "Unknown Game")),
        "cover_url": cover_url,
        "summary": summary,
        "storyline": storyline,
        "rating": f"{float(rating):.1f}" if pd.notna(rating) else "N/A",
        "total_rating": f"{float(total_rating):.1f}" if pd.notna(total_rating) else "N/A",
        "rating_count": int(rating_count) if pd.notna(rating_count) else 0,
        "year": str(int(year)) if pd.notna(year) else "Unknown",
        "genres": genres,
        "platforms": platforms,
        "game_modes": game_modes,
        "player_perspectives": player_perspectives,
        "age_rating": age_rating_str,
        "languages": languages,
        "videos": videos[:2],
        "ttb_hastily": float(ttb_hastily) if pd.notna(ttb_hastily) else None,
        "ttb_normally": float(ttb_normally) if pd.notna(ttb_normally) else None,
        "ttb_completely": float(ttb_completely) if pd.notna(ttb_completely) else None,
        "follows": int(game.get("follows", 0)) if pd.notna(game.get("follows", 0)) else 0,
        "hypes": int(game.get("hypes", 0)) if pd.notna(game.get("hypes", 0)) else 0,
        "url": game.get("url", "") if pd.notna(game.get("url", "")) else "",
    }


def get_random_games(n: int = 10) -> pd.DataFrame:
    """Get random games for display (e.g., featured games)."""
    games = load_games()
    if games.empty:
        return pd.DataFrame()
    
    # Filter to games with covers and ratings for better display
    filtered = games[
        games["cover_url"].notna() & 
        games["total_rating"].notna() &
        (games["total_rating"] > 70)
    ]
    
    if len(filtered) < n:
        filtered = games[games["cover_url"].notna()]
    
    if len(filtered) < n:
        filtered = games
    
    return filtered.sample(min(n, len(filtered)))
