"""
Content Recommender V2: Recommendation utilities for Streamlit app.

This is a simplified version that works with pre-computed recommendations.
For Option B (real-time), it also provides similarity computation.
"""

import numpy as np
import pandas as pd
from typing import Optional
import time

# Import S3 loader functions
from s3_loader_V2 import (
    get_recommendations_for_game,
    compute_realtime_recommendations,
    load_games
)


def get_recommendations(
    game_id: int, 
    top_n: int = 10, 
    method: str = "precomputed"
) -> tuple[pd.DataFrame, float]:
    """
    Get recommendations for a game.
    
    Args:
        game_id: The game ID to get recommendations for
        top_n: Number of recommendations
        method: "precomputed" (Option A) or "realtime" (Option B)
    
    Returns:
        tuple: (recommendations DataFrame, time_taken in seconds)
    """
    start_time = time.time()
    
    if method == "precomputed":
        recs = get_recommendations_for_game(game_id, top_n)
    elif method == "realtime":
        recs = compute_realtime_recommendations(game_id, top_n)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'precomputed' or 'realtime'.")
    
    elapsed = time.time() - start_time
    return recs, elapsed


def format_recommendation_card(game: pd.Series) -> dict:
    """
    Format a game row for display as a recommendation card.
    
    Returns dict with display-ready values.
    """
    # Handle list columns (may be stored as lists or strings)
    def parse_list(val):
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
    
    # Format rating
    rating = game.get("rating") or game.get("total_rating")
    rating_str = f"{float(rating):.1f}/100" if pd.notna(rating) else "N/A"
    
    # Format similarity score
    score = game.get("similarity_score", 0)
    score_str = f"{float(score):.1%}" if pd.notna(score) and score > 0 else ""
    
    # Cover URL
    cover_url = game.get("cover_url", "")
    if pd.isna(cover_url) or not cover_url:
        cover_url = "https://via.placeholder.com/264x374?text=No+Cover"
    
    # Release year
    year = game.get("release_year")
    year_str = str(int(year)) if pd.notna(year) else "Unknown"
    
    # Summary (truncate if too long)
    summary = str(game.get("summary", "")) if pd.notna(game.get("summary")) else ""
    if len(summary) > 200:
        summary = summary[:200] + "..."
    
    return {
        "id": int(game.get("id", 0)),
        "name": str(game.get("name", "Unknown Game")),
        "cover_url": cover_url,
        "rating": rating_str,
        "year": year_str,
        "genres": genres[:3],  # Limit to 3 genres
        "platforms": platforms[:5],  # Limit to 5 platforms
        "similarity": score_str,
        "summary": summary,
    }


def get_game_display_info(game_id: int) -> Optional[dict]:
    """
    Get full display info for a game.
    """
    games = load_games()
    if games.empty:
        return None
    
    game = games[games["id"] == game_id]
    if game.empty:
        return None
    
    game = game.iloc[0]
    
    # Handle list columns
    def parse_list(val):
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
    
    # Cover URL
    cover_url = game.get("cover_url", "")
    if pd.isna(cover_url) or not cover_url:
        cover_url = "https://via.placeholder.com/264x374?text=No+Cover"
    
    # Ratings
    rating = game.get("rating")
    total_rating = game.get("total_rating")
    rating_count = game.get("total_rating_count", 0)
    
    # Year
    year = game.get("release_year")
    
    # Summary and storyline
    summary = game.get("summary", "")
    if pd.isna(summary):
        summary = "No summary available."
    
    storyline = game.get("storyline", "")
    if pd.isna(storyline):
        storyline = ""
    
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
        "follows": int(game.get("follows", 0)) if pd.notna(game.get("follows")) else 0,
        "hypes": int(game.get("hypes", 0)) if pd.notna(game.get("hypes")) else 0,
        "url": game.get("url", ""),
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
        (games["total_rating"] > 70)  # Only highly rated games
    ]
    
    if len(filtered) < n:
        filtered = games[games["cover_url"].notna()]
    
    if len(filtered) < n:
        filtered = games
    
    return filtered.sample(min(n, len(filtered)))
