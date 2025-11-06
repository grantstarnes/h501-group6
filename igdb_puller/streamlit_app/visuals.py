# visuals.py
# Simple Plotly visuals for the Streamlit app
# Author: Raj + ChatGPT
# Purpose: Provide easy-to-read functions to build visuals without clutter.

from typing import Optional, Literal
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Public API (what you'll import)
# -----------------------------
ByOption = Literal["genre", "decade", "release_window", "time_to_beat"]
TTBMode = Literal["hastily", "normally", "completely"]

def scatter_avg_rating_by(
    df: pd.DataFrame,
    by: ByOption,
    searched_game_name: Optional[str] = None,
    ttb_mode: TTBMode = "normally",
    rating_col: str = "total_rating",
    total_playtime_col: str = "total_playtime_hours",
    name_col: str = "name",
    release_date_col: str = "first_release_date",
    genres_col: str = "genres",
    debug: bool = False,
    ttb_cols: dict = None,
):
    """
    Build a scatter plot: Average rating vs (average) playtime, grouped by `by`.

    - If by in {"genre", "decade", "release_window"}:
        X-axis = mean of `total_playtime_col` within group
        Y-axis = mean of `rating_col` within group
        Point/group = selected grouping (e.g., each Genre or each Decade)

    - If by == "time_to_beat":
        X-axis = mean time-to-beat for selected `ttb_mode`
        Y-axis = mean of `rating_col`
        Point/group = the TTB mode ("hastily", "normally", "completely")

    - Highlights the searched game (exact name match) with a distinct marker & annotation.
    - Returns a Plotly Figure (go.Figure). If essential columns are missing, returns a
      simple empty figure with a helpful message in the title.

    Parameters
    ----------
    df : DataFrame
        Source table with at least rating, playtime or time-to-beat, names, etc.
    by : {"genre","decade","release_window","time_to_beat"}
        Grouping toggle.
    searched_game_name : str or None
        If provided, we highlight this game (exact match on `name_col`).
    ttb_mode : {"hastily","normally","completely"}
        Which time-to-beat column to use when by == "time_to_beat".
    rating_col : str
        Column with numeric rating (0-100 typical IGDB).
    total_playtime_col : str
        Column with total playtime (in hours). Used for all groupings except time_to_beat.
    name_col : str
        Column with game names.
    release_date_col : str
        Column with release date (datetime-like or str). Used to derive decade & release_window.
    genres_col : str
        Column with genre names (list-like or delimited string). Used for by="genre".
    ttb_cols : dict or None
        Mapping for time-to-beat columns; defaults to:
        {"hastily": "ttb_hastily_hours", "normally": "ttb_normally_hours", "completely": "ttb_completely_hours"}

    Returns
    -------
    go.Figure
    """
    # ---- Defensive defaults ----
    if ttb_cols is None:
        ttb_cols = {
            "hastily": "ttb_hastily_hours",
            "normally": "ttb_normally_hours",
            "completely": "ttb_completely_hours",
        }

    # Work on a copy to avoid mutating caller's df
    data = df.copy()

    # Ensure numeric columns are numeric
    for col in [rating_col, total_playtime_col, *ttb_cols.values()]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Normalize/derive helper columns
    data = _ensure_release_helpers(data, release_date_col)  # adds "decade", "release_window"
    data = _ensure_genres_as_rows(data, genres_col)         # ensures a string/list is usable
    data = _ensure_total_playtime(data, total_playtime_col, ttb_cols)

    debug_info = {}
    if debug:
        present_cols = list(data.columns)
        ttb_present = [c for c in ttb_cols.values() if c in data.columns] if ttb_cols else []
        debug_info["rows_total"] = len(data)
        debug_info["present_columns"] = present_cols
        debug_info["ttb_present"] = ttb_present
        debug_info["nonnull_counts"] = {
            "rating_nonnull": int(data[rating_col].notna().sum()) if rating_col in data.columns else 0,
            "playtime_nonnull": int(data[total_playtime_col].notna().sum()) if total_playtime_col in data.columns else 0,
            **{f"{c}_nonnull": int(data[c].notna().sum()) for c in ttb_present}
        }
    # ---- Build grouped dataset for the requested view ----
    if by in {"genre", "decade", "release_window"}:
        # Validate required columns
        missing = [c for c in [rating_col, total_playtime_col] if c not in data.columns]
        if missing:
            return _message_figure(
                f"Missing required columns: {', '.join(missing)}. "
                f"Cannot build '{by}' scatter."
            )

        # Choose group label column
        if by == "genre":
            group_col = "genre_group"
        elif by == "decade":
            group_col = "decade"
        else:
            group_col = "release_window"

        g = (
            data.dropna(subset=[rating_col, total_playtime_col])
                .groupby(group_col, dropna=False, as_index=False)
                .agg(
                    avg_rating=(rating_col, "mean"),
                    avg_playtime=(total_playtime_col, "mean"),
                    count=("id" if "id" in data.columns else name_col, "count"),
                )
        )
        if debug:
            debug_info["by"] = by
            debug_info["rows_used_for_grouping"] = int(len(g))
            debug_info["group_preview"] = g.head(15)  # DataFrame

        # A little ordering for readability
        g = g.sort_values("avg_rating", ascending=False)

        fig = px.scatter(
            g,
            x="avg_playtime",
            y="avg_rating",
            hover_name=group_col,
            size="count",
            labels={
                "avg_playtime": "Average Playtime (hrs)",
                "avg_rating": "Average Rating",
                group_col: by.replace("_", " ").title(),
                "count": "Games in group",
            },
            title=_title_for(by, ttb_mode=None),
        )

    elif by == "time_to_beat":
        # Validate TTB column
        ttb_col = ttb_cols.get(ttb_mode)
        if ttb_col is None or ttb_col not in data.columns:
            return _message_figure(
                f"Missing time-to-beat column for mode '{ttb_mode}'. "
                f"Expected column '{ttb_cols.get(ttb_mode, '<not set>')}'."
            )
        if rating_col not in data.columns:
            return _message_figure(f"Missing required column: {rating_col}.")

        g = (
            data.dropna(subset=[rating_col, ttb_col])
                .assign(ttb_mode=ttb_mode)  # single mode selected at a time
                .groupby("ttb_mode", as_index=False)
                .agg(
                    avg_rating=(rating_col, "mean"),
                    avg_playtime=(ttb_col, "mean"),
                    count=("id" if "id" in data.columns else name_col, "count"),
                )
        )

        if debug:
            debug_info["by"] = by
            debug_info["rows_used_for_grouping"] = int(len(g))
            debug_info["group_preview"] = g.head(15)  # DataFrame

        fig = px.scatter(
            g,
            x="avg_playtime",
            y="avg_rating",
            hover_name="ttb_mode",
            size="count",
            labels={
                "avg_playtime": f"Avg Time-to-Beat ({ttb_mode}) hrs",
                "avg_rating": "Average Rating",
                "ttb_mode": "TTB Mode",
                "count": "Games in group",
            },
            title=_title_for(by, ttb_mode=ttb_mode),
        )

    else:
        return _message_figure(f"Unknown 'by' value: {by}")

    # ---- Aesthetics ----
    fig.update_traces(marker=dict(opacity=0.9), selector=dict(mode="markers"))
    fig.update_layout(
        legend_title_text=by.replace("_", " ").title(),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

    # ---- Optional: highlight searched game ----
    if searched_game_name:
        fig = _add_highlight_annotation(
            fig=fig,
            full_df=data,
            by=by,
            searched_game_name=searched_game_name,
            rating_col=rating_col,
            total_playtime_col=total_playtime_col,
            ttb_mode=ttb_mode,
            ttb_cols=ttb_cols,
            name_col=name_col,
        )

    if debug:
        return fig, debug_info
    return fig


# -----------------------------
# Helpers (internal)
# -----------------------------
def _ensure_release_helpers(df: pd.DataFrame, release_date_col: str) -> pd.DataFrame:
    out = df.copy()
    if release_date_col in out.columns:
        # Try to coerce into datetime
        out[release_date_col] = pd.to_datetime(out[release_date_col], errors="coerce", utc=True)
        # Decade (e.g., 1990s, 2000s)
        decade = out[release_date_col].dt.year // 10 * 10
        out["decade"] = decade.where(decade.notna(), other=pd.NA).astype("Int64").astype("string") + "s"
        # Release window by calendar quarter
        q = out[release_date_col].dt.quarter
        out["release_window"] = q.map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})
    else:
        out["decade"] = pd.NA
        out["release_window"] = pd.NA
    return out


def _ensure_genres_as_rows(df: pd.DataFrame, genres_col: str) -> pd.DataFrame:
    """
    Normalize `genres` so we can group by genre.
    Accepts: list-like, pipe/comma delimited, or single string.
    Produces a 'genre_group' column (single string per row).
    """
    out = df.copy()
    if genres_col not in out.columns:
        out["genre_group"] = pd.NA
        return out

    # Create a normalized list column
    def _to_list(val):
        if pd.isna(val):
            return []
        if isinstance(val, (list, tuple, set)):
            return list(val)
        if isinstance(val, str):
            # common delimiters from APIs/ETL
            for sep in ["|", ",", ";", "/"]:
                if sep in val:
                    return [s.strip() for s in val.split(sep) if s.strip()]
            return [val.strip()]
        return []

    out["_genres_list"] = out[genres_col].apply(_to_list)

    if out["_genres_list"].apply(len).sum() == 0:
        # No usable genres
        out["genre_group"] = pd.NA
        return out

    # Explode to one row per genre, but keep original rows too (for highlight lookup)
    exploded = out.explode("_genres_list", ignore_index=True)
    exploded["genre_group"] = exploded["_genres_list"].where(
        exploded["_genres_list"].notna(), other=pd.NA
    )
    exploded.drop(columns=["_genres_list"], inplace=True)
    return exploded


def _title_for(by: str, ttb_mode: Optional[str]) -> str:
    if by == "time_to_beat":
        return f"Average Rating vs Time-to-Beat ({ttb_mode})"
    label = by.replace("_", " ").title()
    return f"Average Rating vs Average Playtime by {label}"


def _message_figure(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=msg, margin=dict(l=10, r=10, t=60, b=10))
    return fig

def _ensure_total_playtime(df: pd.DataFrame, total_playtime_col: str, ttb_cols: dict) -> pd.DataFrame:
    """
    If `total_playtime_col` is missing, create it as the row-wise mean of any
    available time-to-beat columns (hastily/normally/completely), ignoring NaNs.
    """
    out = df.copy()
    if total_playtime_col in out.columns:
        return out

    candidate_cols = [c for c in ttb_cols.values() if c in out.columns]
    if candidate_cols:
        # row-wise mean of available TTB columns
        out[total_playtime_col] = out[candidate_cols].astype(float).mean(axis=1, skipna=True)
    else:
        # no TTB columns to derive from
        out[total_playtime_col] = pd.NA
    return out

def _add_highlight_annotation(
    fig: go.Figure,
    full_df: pd.DataFrame,
    by: str,
    searched_game_name: str,
    rating_col: str,
    total_playtime_col: str,
    ttb_mode: str,
    ttb_cols: dict,
    name_col: str,
) -> go.Figure:
    # Locate the searched game row
    row = (
        full_df.loc[full_df[name_col].astype(str) == str(searched_game_name)]
        .copy()
        .head(1)
    )
    if row.empty:
        return fig  # nothing to highlight

    # Determine x position based on view
    if by == "time_to_beat":
        x_col = ttb_cols.get(ttb_mode)
    else:
        x_col = total_playtime_col

    if x_col not in full_df.columns or rating_col not in full_df.columns:
        return fig

    x = pd.to_numeric(row.iloc[0][x_col])
    y = pd.to_numeric(row.iloc[0][rating_col])
    if pd.isna(x) or pd.isna(y):
        return fig

    # Add a single-point scatter trace with larger marker
    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers+text",
            text=[searched_game_name],
            textposition="top center",
            hoverinfo="skip",
            marker=dict(size=16, symbol="star", line=dict(width=1)),
            name="Searched Game",
            showlegend=True,
        )
    )

    return fig
