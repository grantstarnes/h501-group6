"""
Content Recommender V3: IGDB API–based recommendations.

This version:
- Uses IGDB's `similar_games` field instead of precomputed S3 recommendations.
- Exposes the same public functions as V2 so streamlit_app_V2.py works:
    - get_recommendations(game_id, top_n=5, method="igdb")
    - format_recommendation_card(game: pd.Series) -> dict
    - get_game_display_info(game_id: int) -> dict | None
    - get_random_games(n: int) -> pd.DataFrame
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

from s3_loader_V2 import (
    load_games,
)

# ---------------------------------------------------------------------------
# IGDB API CONFIG
# ---------------------------------------------------------------------------

GAMES_URL = "https://api.igdb.com/v4/games"

# Expect these keys in .streamlit/secrets.toml:
# CLIENT_ID="..."
# ACCESS_TOKEN="..."
_CLIENT_ID = st.secrets.get("CLIENT_ID") or st.secrets.get("TWITCH_CLIENT_ID")
_ACCESS_TOKEN = st.secrets.get("ACCESS_TOKEN") or st.secrets.get("TWITCH_CLIENT_SECRET")

FIELDS = (
    "name,summary,genres.name,platforms.name,release_dates.date,"
    "cover.url,rating,similar_games"
)

TOP_N_DEFAULT = 5


# ---------------------------------------------------------------------------
# IGDB HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def _igdb_headers() -> Dict[str, str]:
    if not _CLIENT_ID or not _ACCESS_TOKEN:
        raise RuntimeError(
            "IGDB credentials missing. Set CLIENT_ID and ACCESS_TOKEN in secrets.toml."
        )
    return {
        "Client-ID": _CLIENT_ID,
        "Authorization": f"Bearer {_ACCESS_TOKEN}",
    }


def _format_cover(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    # Upgrade t_thumb -> t_cover_big
    return "https:" + url.replace("t_thumb", "t_cover_big")


def _parse_year(timestamp: Optional[int]) -> Optional[int]:
    if not timestamp:
        return None
    try:
        return datetime.utcfromtimestamp(timestamp).year
    except Exception:
        return None


def _clean_game(game: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize IGDB game object to a simpler dict."""
    genres = [g["name"] for g in game.get("genres", [])] if game.get("genres") else []
    platforms = (
        [p["name"] for p in game.get("platforms", [])]
        if game.get("platforms")
        else []
    )
    release_dates = game.get("release_dates") or []
    year = None
    if release_dates:
        year = _parse_year(release_dates[0].get("date"))

    rating = game.get("rating")
    rating_clean = round(float(rating), 1) if isinstance(rating, (int, float)) else None

    cover_url = _format_cover(game.get("cover", {}).get("url"))

    return {
        "id": game.get("id"),
        "name": game.get("name"),
        "summary": game.get("summary"),
        "genres": genres,
        "platforms": platforms,
        "year": year,
        "cover_url": cover_url,
        "rating": rating_clean,
    }

@st.cache_data(show_spinner=False)
def _fetch_igdb_cover_for_name(name: str) -> Optional[str]:
    """
    Look up a game by name on IGDB and return a high-res cover URL if available.
    Cached so we don't repeatedly hit the API for the same title.
    """
    if not name:
        return None

    try:
        headers = _igdb_headers()
    except RuntimeError:
        # Missing credentials – skip lookup
        return None

    query = f'fields cover.url; search "{name}"; limit 1;'
    resp = requests.post(GAMES_URL, headers=headers, data=query)
    if resp.status_code != 200:
        return None

    data = resp.json()
    if not data:
        return None

    cover_obj = data[0].get("cover")
    if not cover_obj:
        return None

    return _format_cover(cover_obj.get("url"))


def _compute_similarity(main: Dict[str, Any], rec: Dict[str, Any]) -> float:
    """
    Compute a simple 0–100 similarity score based on genres, platforms,
    rating proximity, and year proximity.
    """
    score = 0.0

    # Genres overlap (up to 50 points)
    main_genres = set(main.get("genres") or [])
    rec_genres = set(rec.get("genres") or [])
    if main_genres:
        shared = main_genres & rec_genres
        score += (len(shared) / max(1, len(main_genres))) * 50.0

    # Platforms overlap (up to 20 points)
    main_plat = set(main.get("platforms") or [])
    rec_plat = set(rec.get("platforms") or [])
    if main_plat:
        shared_p = main_plat & rec_plat
        score += (len(shared_p) / max(1, len(main_plat))) * 20.0

    # Rating proximity (up to 20 points)
    main_rating = main.get("rating")
    rec_rating = rec.get("rating")
    if isinstance(main_rating, (int, float)) and isinstance(rec_rating, (int, float)):
        diff = abs(main_rating - rec_rating)
        score += max(0.0, 20.0 - diff)

    # Year proximity (up to 10 points)
    main_year = main.get("year")
    rec_year = rec.get("year")
    if isinstance(main_year, int) and isinstance(rec_year, int):
        diff_y = abs(main_year - rec_year)
        score += max(0.0, 10.0 - diff_y)

    # Clamp 0–100
    return float(round(min(score, 100.0), 1))

@st.cache_data(show_spinner=False)
def _fetch_igdb_recommendations(game_title: str, top_n: int) -> Dict[str, Any]:
    """
    Core IGDB flow:
    - Search by title (limit 1)
    - Get its similar_games ids
    - Fetch those games and compute similarity scores
    """
    headers = _igdb_headers()

    # 1️⃣ Search main game
    query = f'fields {FIELDS}; search "{game_title}"; limit 1;'
    resp = requests.post(GAMES_URL, headers=headers, data=query)
    if resp.status_code != 200:
        return {"error": f"IGDB search failed: {resp.text}"}

    data = resp.json()
    if not data:
        return {"error": f"No results found on IGDB for '{game_title}'"}

    main_raw = data[0]
    main_game = _clean_game(main_raw)
    similar_ids = main_raw.get("similar_games", [])[:top_n]

    recommendations: List[Dict[str, Any]] = []
    if similar_ids:
        id_list = ",".join(str(i) for i in similar_ids)
        query_recs = f"fields {FIELDS}; where id = ({id_list});"
        rec_resp = requests.post(GAMES_URL, headers=headers, data=query_recs)
        if rec_resp.status_code != 200:
            return {"error": f"IGDB recommendations failed: {rec_resp.text}"}

        recs_raw = rec_resp.json()
        for r in recs_raw:
            rec_game = _clean_game(r)
            sim_score_100 = _compute_similarity(main_game, rec_game)
            # Store 0–1 score so the UI can format as percent
            rec_game["similarity_score"] = sim_score_100 / 100.0
            recommendations.append(rec_game)

        # Sort high to low similarity
        recommendations = sorted(
            recommendations, key=lambda x: x["similarity_score"], reverse=True
        )

    return {"searched_game": main_game, "recommendations": recommendations}


# ---------------------------------------------------------------------------
# PUBLIC API: get_recommendations (used by streamlit_app_V2.py)
# ---------------------------------------------------------------------------

def get_recommendations(
    game_id: int,
    top_n: int = TOP_N_DEFAULT,
    method: str = "igdb",
) -> tuple[pd.DataFrame, float]:
    """
    Given a local game_id (from S3 dataset), look up its name and get
    IGDB-based recommendations.

    Returns:
        (DataFrame, elapsed_time_seconds)

        DataFrame columns (at least):
            - id (int)
            - name (str)
            - cover_url (str)
            - rating (float or None)
            - release_year (int or None)
            - summary (str)
            - genre_names (list[str])
            - platform_names (list[str])
            - similarity_score (float, 0–1)
    """
    start = time.time()

    games = load_games()
    if games.empty:
        return pd.DataFrame(), time.time() - start

    row = games[games["id"] == game_id]
    if row.empty:
        return pd.DataFrame(), time.time() - start

    game_name = str(row.iloc[0]["name"])

    try:
        result = _fetch_igdb_recommendations(game_name, top_n=top_n)
    except RuntimeError as e:
        # Missing IGDB credentials
        st.warning(str(e))
        return pd.DataFrame(), time.time() - start
    except Exception as e:
        st.warning(f"Error calling IGDB: {e}")
        return pd.DataFrame(), time.time() - start

    if "error" in result:
        st.info(result["error"])
        return pd.DataFrame(), time.time() - start

    recs = result["recommendations"]
    if not recs:
        return pd.DataFrame(), time.time() - start

    # Build DataFrame compatible with format_recommendation_card
    df = pd.DataFrame.from_records(recs)

    # Normalize column names to match V2 expectations
    df = df.rename(
        columns={
            "year": "release_year",
            "genres": "genre_names",
            "platforms": "platform_names",
        }
    )

    elapsed = time.time() - start
    return df, elapsed


# ---------------------------------------------------------------------------
# PUBLIC API: format_recommendation_card (unchanged from V2 style)
# ---------------------------------------------------------------------------

def format_recommendation_card(game: pd.Series) -> dict:
    """
    Format a game row (from get_recommendations DataFrame) for display
    as a recommendation card.

    Returns dict with:
        id, name, cover_url, rating, year, genres, platforms,
        similarity (string like '82.3%'), summary
    """

    def parse_list(val):
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
            except Exception:
                return []
        return []

    genres = parse_list(game.get("genre_names", []))
    platforms = parse_list(game.get("platform_names", []))

    rating = game.get("rating") or game.get("total_rating")
    rating_str = f"{float(rating):.1f}/100" if pd.notna(rating) else "N/A"

    score = game.get("similarity_score", 0)
    # similarity_score is 0–1 float → convert to "82.3%"
    similarity_str = ""
    try:
        if pd.notna(score) and float(score) > 0:
            similarity_str = f"{float(score) * 100:.1f}%"
    except Exception:
        similarity_str = ""

    cover_url = game.get("cover_url", "")
    if pd.isna(cover_url) or not cover_url:
        cover_url = "https://via.placeholder.com/264x374?text=No+Cover"

    year = game.get("release_year")
    year_str = str(int(year)) if pd.notna(year) else "Unknown"

    summary = (
        str(game.get("summary", "")) if pd.notna(game.get("summary")) else ""
    )
    if len(summary) > 200:
        summary = summary[:200] + "..."

    return {
        "id": int(game.get("id", 0)) if pd.notna(game.get("id", 0)) else 0,
        "name": str(game.get("name", "Unknown Game")),
        "cover_url": cover_url,
        "rating": rating_str,
        "year": year_str,
        "genres": genres[:3],
        "platforms": platforms[:5],
        "similarity": similarity_str,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# PUBLIC API: get_game_display_info / get_random_games
# (copied from V2, unchanged – these depend on S3 data only)
# ---------------------------------------------------------------------------

def get_game_display_info(game_id: int) -> Optional[dict]:
    """
    Get full display info for a game from the local S3 dataset.

    Returns dict with all fields needed for the details page including:
    - Basic info (name, cover, year, ratings)
    - Genres, platforms, game modes
    - Player perspectives
    - Age rating
    - Supported languages
    - Video URLs (top 2)
    - Time-to-beat metrics
    """
    games = load_games()
    if games.empty:
        return None

    game = games[games["id"] == game_id]
    if game.empty:
        return None

    game = game.iloc[0]

    def parse_list(val):
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
            except Exception:
                return []
        return []

    genres = parse_list(game.get("genre_names", []))
    platforms = parse_list(game.get("platform_names", []))
    game_modes = parse_list(game.get("game_mode_names", []))
    player_perspectives = parse_list(game.get("player_perspective_names", []))
    languages = parse_list(game.get("language_names", []))
    videos = parse_list(game.get("video_urls", []))

    cover_url = game.get("cover_url", "")

    # If local data is missing a cover, try IGDB by name
    if pd.isna(cover_url) or not cover_url or "placeholder.com" in str(cover_url):
        try:
            igdb_cover = _fetch_igdb_cover_for_name(str(game.get("name", "")))
            if igdb_cover:
                cover_url = igdb_cover
        except Exception:
            # If anything goes wrong, we silently fall back to placeholder
            pass

    # Final fallback
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
    """Get random high-rated games for display (e.g., featured games)."""
    games = load_games()
    if games.empty:
        return pd.DataFrame()

    filtered = games[
        games["cover_url"].notna()
        & games["total_rating"].notna()
        & (games["total_rating"] > 70)
    ]

    if len(filtered) < n:
        filtered = games[games["cover_url"].notna()]

    if len(filtered) < n:
        filtered = games

    return filtered.sample(min(n, len(filtered)))
