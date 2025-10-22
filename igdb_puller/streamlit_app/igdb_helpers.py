# streamlit_app/igdb_helpers.py
from __future__ import annotations
import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

# IMPORTANT: set env FIRST, then import igdb_puller (it checks env on import)
def _ensure_env(client_id: str, client_secret: str) -> None:
    os.environ.setdefault("TWITCH_CLIENT_ID", client_id)
    os.environ.setdefault("TWITCH_CLIENT_SECRET", client_secret)

# Lazy import so we can inject secrets in Streamlit before importing the package.
_IGDB_IMPORTED = False
def _import_igdb():
    global _IGDB_IMPORTED, IGDBClient
    if not _IGDB_IMPORTED:
        try:
            # 1) Normal case: package is importable (editable install or running from repo root)
            from igdb_puller import IGDBClient
        except Exception:
            # 2) Add repo root to sys.path dynamically (works if running from nested folder)
            import sys, pathlib
            here = pathlib.Path(__file__).resolve()
            # try ../../ as repo root if file is .../igdb_puller/streamlit_app/igdb_helpers.py
            candidates = [
                here.parents[1],  # repo_root/streamlit_app/...
                here.parents[2],  # repo_root/igdb_puller/streamlit_app/...
                here.parents[3],  # safety net if structure differs
            ]
            for p in candidates:
                if (p / "igdb_puller").exists() and str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            try:
                from igdb_puller import IGDBClient  # retry after path fix
            except Exception:
                # 3) Final fallback: relative import if this module is inside igdb_puller/*
                try:
                    from ..client import IGDBClient  # type: ignore
                except Exception as e:
                    raise ModuleNotFoundError(
                        "Cannot import IGDBClient. Run from repo root and/or `pip install -e .`"
                    ) from e
        _IGDB_IMPORTED = True
    return IGDBClient

# ---------- URL builders ----------
def igdb_image_url(image_id: str, size: str = "t_cover_big") -> str:
    # sizes: t_thumb, t_cover_small, t_cover_big, t_720p, t_screenshot_big, etc.
    return f"https://images.igdb.com/igdb/image/upload/{size}/{image_id}.jpg"

# ---------- Core fetchers ----------
def get_client(client_id: str, client_secret: str, rate_sleep: float = 0.40):
    _ensure_env(client_id, client_secret)
    IGDBClient = _import_igdb()
    return IGDBClient(rate_sleep=rate_sleep)

def search_games(client, query: str, limit: int = 25) -> List[Dict]:
    # IGDB supports `search "..."` on /games
    fields = ",".join([
        "id","name","slug","first_release_date",
        "total_rating","total_rating_count",
        "aggregated_rating","aggregated_rating_count",
        "cover.image_id"
    ])
    body_where = None  # we rely on search instead of where
    results = []
    # Use client.paged to respect rate limits (your client streams rows)
    for row in client.paged(
        endpoint="games",
        fields=fields,
        where=body_where,
        sort="total_rating_count desc",
        limit=50,
        max_rows=300
    ):
        # client.paged doesn't yet allow passing "search"; we emulate via where on name ~ *query*
        # Alternative: IGDB supports separate /games endpoint with "search" directive,
        # but to keep using your client.paged, we'll filter client-side for now.
        # If you want perfect IGDB search, add a `search` parameter path in IGDBClient later.
        if query.lower() in str(row.get("name","")).lower():
            results.append(row)
        if len(results) >= limit:
            break
    return results

def fetch_game_details(client, game_id: int) -> Dict:
    # Pull a rich game record
    fields = ",".join([
        "id","name","slug","summary","storyline","first_release_date",
        "total_rating","total_rating_count","aggregated_rating","aggregated_rating_count",
        "genres","platforms","involved_companies","websites",
        "cover.image_id","artworks.image_id","screenshots.image_id"
    ])
    rows = list(client.paged(
        endpoint="games",
        fields=fields,
        where=f"id = {game_id}",
        sort="id asc",
        limit=1,
        max_rows=1
    ))
    if not rows:
        return {}
    game = rows[0]

    # Resolve lookups (genres, platforms, involved companies -> companies)
    genre_ids = _to_int_list(game.get("genres"))
    plat_ids  = _to_int_list(game.get("platforms"))
    ic_ids    = _to_int_list(game.get("involved_companies"))

    genres = fetch_names(client, "genres", genre_ids)
    platforms = fetch_platforms(client, plat_ids)
    publishers, developers = fetch_companies_from_involved(client, ic_ids)

    websites = fetch_websites(client, _to_int_list(game.get("websites")))

    # Build images
    cover = _first(game.get("cover.image_id"))
    artwork_ids = _to_str_list(game.get("artworks.image_id"))
    screenshot_ids = _to_str_list(game.get("screenshots.image_id"))
    images = []
    if cover:
        images.append(igdb_image_url(cover, "t_cover_big"))
    images += [igdb_image_url(i, "t_screenshot_big") for i in artwork_ids[:8]]
    images += [igdb_image_url(i, "t_screenshot_big") for i in screenshot_ids[:8]]

    return {
        "raw": game,
        "name": game.get("name"),
        "summary": game.get("summary"),
        "storyline": game.get("storyline"),
        "first_release_date": game.get("first_release_date"),
        "ratings": {
            "total_rating": game.get("total_rating"),
            "total_rating_count": game.get("total_rating_count"),
            "aggregated_rating": game.get("aggregated_rating"),
            "aggregated_rating_count": game.get("aggregated_rating_count"),
        },
        "genres": genres,
        "platforms": platforms,
        "publishers": publishers,
        "developers": developers,
        "websites": websites,
        "images": images
    }

def fetch_names(client, endpoint: str, ids: List[int]) -> List[str]:
    if not ids:
        return []
    rows = list(client.paged(endpoint=endpoint, fields="id,name", where=f"id = ({','.join(map(str, ids))})", max_rows=1000))
    id_to_name = {int(r["id"]): r.get("name","") for r in rows}
    return [id_to_name.get(i, str(i)) for i in ids]

def fetch_platforms(client, ids: List[int]) -> List[str]:
    if not ids:
        return []
    rows = list(client.paged(endpoint="platforms", fields="id,name,abbreviation", where=f"id = ({','.join(map(str, ids))})", max_rows=1000))
    id_to_disp = {int(r["id"]): (r.get("abbreviation") or r.get("name") or str(r["id"])) for r in rows}
    return [id_to_disp.get(i, str(i)) for i in ids]

def fetch_websites(client, ids: List[int]) -> List[Tuple[str, Optional[int]]]:
    if not ids:
        return []
    rows = list(client.paged(endpoint="websites", fields="id,url,category", where=f"id = ({','.join(map(str, ids))})", max_rows=1000))
    return [(r.get("url",""), r.get("category")) for r in rows]

def fetch_companies_from_involved(client, involved_ids: List[int]) -> Tuple[List[str], List[str]]:
    if not involved_ids:
        return ([], [])
    ic_rows = list(client.paged(endpoint="involved_companies",
                                fields="id,company,developer,publisher",
                                where=f"id = ({','.join(map(str, involved_ids))})",
                                max_rows=1000))
    comp_ids = sorted({int(r["company"]) for r in ic_rows if r.get("company") is not None})
    if not comp_ids:
        return ([], [])
    c_rows = list(client.paged(endpoint="companies", fields="id,name", where=f"id = ({','.join(map(str, comp_ids))})", max_rows=1000))
    id_to_name = {int(r["id"]): r.get("name","") for r in c_rows}
    publishers = [id_to_name.get(int(r["company"]), "") for r in ic_rows if r.get("publisher")]
    developers = [id_to_name.get(int(r["company"]), "") for r in ic_rows if r.get("developer")]
    # de-duplicate while preserving order
    publishers = list(dict.fromkeys([x for x in publishers if x]))
    developers = list(dict.fromkeys([x for x in developers if x]))
    return (publishers, developers)

# ---------- Aggregations for “quick questions” ----------
def load_games_for_analytics(client, max_rows: int = 20000) -> pd.DataFrame:
    # Pull ratings + genres + platforms + first_release_date
    fields = "id,name,total_rating,total_rating_count,genres,platforms,first_release_date"
    rows = list(client.paged("games", fields=fields, where="total_rating != null", sort="total_rating_count desc", max_rows=max_rows))
    df = pd.DataFrame(rows)
    return df

def resolve_genre_names(client, ids: List[int]) -> Dict[int, str]:
    if not ids:
        return {}
    rows = list(client.paged("genres", fields="id,name", where=f"id = ({','.join(map(str, ids))})", max_rows=2000))
    return {int(r["id"]): r.get("name","") for r in rows}

def resolve_platform_names(client, ids: List[int]) -> Dict[int, str]:
    if not ids:
        return {}
    rows = list(client.paged("platforms", fields="id,name,abbreviation", where=f"id = ({','.join(map(str, ids))})", max_rows=4000))
    return {int(r["id"]): (r.get("abbreviation") or r.get("name","")) for r in rows}

def df_most_rated_genre(client, df: pd.DataFrame) -> pd.DataFrame:
    # explode genres -> mean(total_rating)
    genre_ids = sorted({int(i) for lst in df.get("genres",[]).dropna().tolist() for i in _to_int_list(lst)})
    gmap = resolve_genre_names(client, genre_ids)
    ex = (df[["id","total_rating","genres"]]
          .dropna(subset=["genres"])
          .assign(genres=df["genres"].apply(_to_int_list))
          .explode("genres")
          .rename(columns={"genres":"genre_id"}))
    ex["genre"] = ex["genre_id"].map(gmap).fillna(ex["genre_id"].astype(str))
    out = (ex.groupby("genre", as_index=False)["total_rating"]
             .mean()
             .rename(columns={"total_rating":"avg_rating"})
             .sort_values("avg_rating", ascending=False))
    return out

def df_best_year(client, df: pd.DataFrame) -> pd.DataFrame:
    # year with highest avg rating (filter by count threshold to reduce noise)
    d2 = df.dropna(subset=["first_release_date", "total_rating"]).copy()
    d2["year"] = (pd.to_datetime(d2["first_release_date"], unit="s", errors="coerce").dt.year)
    out = (d2.groupby("year", as_index=False)
             .agg(avg_rating=("total_rating","mean"), n=("id","count"))
             .query("n >= 20")
             .sort_values(["avg_rating","n"], ascending=[False, False]))
    return out

def df_best_platform(client, df: pd.DataFrame) -> pd.DataFrame:
    plat_ids = sorted({int(i) for lst in df.get("platforms",[]).dropna().tolist() for i in _to_int_list(lst)})
    pmap = resolve_platform_names(client, plat_ids)
    ex = (df[["id","total_rating","platforms"]]
          .dropna(subset=["platforms"])
          .assign(platforms=df["platforms"].apply(_to_int_list))
          .explode("platforms")
          .rename(columns={"platforms":"platform_id"}))
    ex["platform"] = ex["platform_id"].map(pmap).fillna(ex["platform_id"].astype(str))
    out = (ex.groupby("platform", as_index=False)["total_rating"]
             .mean()
             .rename(columns={"total_rating":"avg_rating"})
             .sort_values("avg_rating", ascending=False))
    return out

def df_best_publisher(client, df: pd.DataFrame, sample_games: int = 8000) -> pd.DataFrame:
    # Approx: sample top-N by rating_count, fetch involved_companies->companies (publisher only)
    d2 = df.sort_values("total_rating_count", ascending=False).head(sample_games).copy()
    game_ids = d2["id"].astype(int).tolist()
    # fetch involved_companies for all these games via /involved_companies? where game = (..)
    ic_rows = list(client.paged("involved_companies",
                                fields="id,company,publisher,game",
                                where=f"game = ({','.join(map(str, game_ids))})",
                                sort="id asc",
                                max_rows=200000))
    comp_ids = sorted({int(r["company"]) for r in ic_rows if r.get("publisher")})
    if not comp_ids:
        return pd.DataFrame(columns=["publisher","avg_rating","n"])

    c_rows = list(client.paged("companies", fields="id,name", where=f"id = ({','.join(map(str, comp_ids))})", max_rows=50000))
    id_to_name = {int(r["id"]): r.get("name","") for r in c_rows}
    ic_df = pd.DataFrame(ic_rows)
    ic_df = ic_df[ic_df["publisher"] == True].copy()
    ic_df["publisher_name"] = ic_df["company"].astype(int).map(id_to_name).fillna(ic_df["company"].astype(str))

    merged = ic_df.merge(d2[["id","total_rating"]], left_on="game", right_on="id", how="left")
    out = (merged.groupby("publisher_name", as_index=False)
           .agg(avg_rating=("total_rating","mean"), n=("game","count"))
           .query("n >= 10")
           .sort_values(["avg_rating","n"], ascending=[False, False]))
    out = out.rename(columns={"publisher_name":"publisher"})
    return out

# ---------- helpers ----------
def _first(v):
    if isinstance(v, list) and v:
        return v[0]
    return v

def _to_int_list(v) -> List[int]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, list):
        return [int(x) for x in v]
    # pipe-joined strings may appear from your flattener; split if needed
    if isinstance(v, str) and "|" in v:
        return [int(x) for x in v.split("|") if x]
    try:
        return [int(v)]
    except Exception:
        return []

def _to_str_list(v) -> List[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str) and "|" in v:
        return [x for x in v.split("|") if x]
    return [str(v)]
