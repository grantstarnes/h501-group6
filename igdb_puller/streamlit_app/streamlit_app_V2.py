"""
Streamlit App V2: Games Popularity Explorer with S3-based data.

This version uses pre-processed data from S3 instead of live IGDB API calls.
Benefits:
- No API credentials needed for end users
- Instant search and recommendations
- Works on Streamlit Community Cloud (no GPU required)

=============================================================================
DEPLOYMENT TO STREAMLIT COMMUNITY CLOUD:
=============================================================================

1. Push this repo to GitHub

2. Go to https://share.streamlit.io

3. Click "New app" and select:
   - Repository: grantstarnes/h501-group6
   - Branch: main
   - Main file path: igdb_puller/streamlit_app/streamlit_app_V2.py

4. No secrets needed (data is public on S3)

5. Click "Deploy"

=============================================================================
LOCAL DEVELOPMENT:
=============================================================================

To run locally with local data (after running data_pipeline_V2.py):
    
    $env:USE_LOCAL_DATA = "true"
    streamlit run streamlit_app_V2.py

To run locally with S3 data:

    streamlit run streamlit_app_V2.py

=============================================================================
"""

import streamlit as st
import pandas as pd
import time
import plotly.express as px

# Import V2 modules
from s3_loader_V2 import (
    load_games,
    search_games,
    check_data_availability,
    get_data_stats,
)
from content_recommender_V3 import (
    get_recommendations,
    format_recommendation_card,
    get_game_display_info,
    get_random_games,
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Everything Games!",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Main container */
    .main > div {
        padding-top: 2rem;
    }

    /* Game card styling */
    .game-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #333333;
    }

    .game-title {
        font-size: 1.5em;
        font-weight: bold;
        color: #ffffff;  /* white title text for dark background */
    }

    .game-meta {
        color: #888888;
        font-size: 0.9em;
    }

    .similarity-badge {
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        margin-left: 10px;
        color: white;
    }

    .genre-tag {
        background-color: #2196F3;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75em;
        margin-right: 5px;
        display: inline-block;
        margin-bottom: 3px;
    }

    .platform-tag {
        background-color: #9C27B0;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75em;
        margin-right: 5px;
        display: inline-block;
        margin-bottom: 3px;
    }

    .rating-display {
        font-size: 2em;
        font-weight: bold;
        color: #4CAF50;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)



# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def display_game_details(game: dict):
    """Display detailed game information with all new fields."""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(game["cover_url"], width=264)
        
        # IGDB link
        if game.get("url"):
            st.markdown(f"[View on IGDB]({game['url']})")
    
    with col2:
        st.markdown(f"# {game['name']}")
        st.markdown(f"**Released:** {game['year']}")
        
        # Age rating (new)
        if game.get("age_rating") and game["age_rating"] != "N/A":
            st.markdown(f"**Age Rating:** {game['age_rating']}")
        
        # Ratings in columns
        rating_col1, rating_col2, rating_col3 = st.columns(3)
        with rating_col1:
            st.metric("Critics Rating", game["rating"])
        with rating_col2:
            st.metric("User Rating", game["total_rating"])
        with rating_col3:
            st.metric("Rating Count", f"{game['rating_count']:,}")
        
        # Genres
        if game["genres"]:
            genre_html = " ".join([f'<span class="genre-tag">{g}</span>' for g in game["genres"]])
            st.markdown(f"**Genres:** {genre_html}", unsafe_allow_html=True)
        
        # Platforms (limit display to avoid clutter)
        if game["platforms"]:
            platforms_to_show = game["platforms"][:8]
            platform_html = " ".join([f'<span class="platform-tag">{p}</span>' for p in platforms_to_show])
            extra = f" +{len(game['platforms']) - 8} more" if len(game["platforms"]) > 8 else ""
            st.markdown(f"**Platforms:** {platform_html}{extra}", unsafe_allow_html=True)
        
        # Game modes
        if game["game_modes"]:
            st.markdown(f"**Game Modes:** {', '.join(game['game_modes'])}")
        
        # Player perspective (new)
        if game.get("player_perspectives"):
            st.markdown(f"**Player Perspective:** {', '.join(game['player_perspectives'])}")
        
        # Stats (follows may not exist in data, hypes does)
        follows = game.get("follows", 0) or 0
        hypes = game.get("hypes", 0) or 0
        if follows > 0 or hypes > 0:
            stats_parts = []
            if follows > 0:
                stats_parts.append(f"**Follows:** {follows:,}")
            if hypes > 0:
                stats_parts.append(f"**Hypes:** {hypes:,}")
            st.markdown(" | ".join(stats_parts))
        
        # Video links (new)
        if game.get("videos"):
            st.markdown("**Videos:**")
            for url in game["videos"][:2]:
                st.markdown(f"- [{url}]({url})")
    
    # Supported languages expander (new)
    if game.get("languages"):
        with st.expander("Supported Languages"):
            st.markdown(", ".join(sorted(set(game["languages"]))))
    
    # Summary and storyline
    st.markdown("---")
    
    if game["summary"]:
        st.markdown("### Summary")
        st.markdown(game["summary"])
    
    if game["storyline"]:
        with st.expander("Storyline", expanded=False):
            st.markdown(game["storyline"])
    
    # Completion time bar chart (new - TTB metrics)
    ttb_data = []
    for label, key in [("Hastily", "ttb_hastily"),
                       ("Normally", "ttb_normally"),
                       ("Completely", "ttb_completely")]:
        val = game.get(key)
        if val is not None:
            try:
                v = float(val)
            except (TypeError, ValueError):
                v = None
            if v is not None and v > 0:
                ttb_data.append({"Completion Type": label, "Hours": v})
    
    if ttb_data:
        st.markdown("---")
        st.markdown("### Completion Time")
        ttb_df = pd.DataFrame(ttb_data)
        fig = px.bar(ttb_df, x="Completion Type", y="Hours",
                     labels={"Hours": "Hours to Complete"},
                     title=None)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def display_recommendation_card(rec: dict, index: int):
    """Display a single recommendation as a card."""
    with st.container():
        col1, col2 = st.columns([1, 3])

        with col1:
            st.image(rec["cover_url"], width=120)

        with col2:
            # Color-coded similarity badge
            sim = rec.get("similarity", "")
            color = "#888888"
            try:
                value = float(sim.replace("%", ""))
                if value >= 70:
                    color = "#4CAF50"   # green
                elif value >= 40:
                    color = "#FF9800"   # orange
                else:
                    color = "#F44336"   # red
            except Exception:
                pass

            # Title with badge
            title_html = f'<span class="game-title">{rec["name"]}</span>'
            if sim:
                title_html += (
                    f' <span class="similarity-badge" '
                    f'style="background-color:{color};">{sim} match</span>'
                )
            st.markdown(title_html, unsafe_allow_html=True)

            # Meta info
            st.markdown(
                f'<span class="game-meta">{rec["year"]} | Rating: {rec["rating"]}</span>',
                unsafe_allow_html=True,
            )

            # Genres
            if rec["genres"]:
                genre_html = " ".join(
                    f'<span class="genre-tag">{g}</span>' for g in rec["genres"]
                )
                st.markdown(genre_html, unsafe_allow_html=True)

            # Summary preview
            if rec["summary"]:
                st.caption(rec["summary"])

        st.markdown("---")



def get_genre_games_df(game_info: dict) -> tuple:
    """
    Return (df, primary_genre) where df contains all games in the selected game's primary genre.
    Filter to rows with valid release_year and ratings.
    
    Returns:
        tuple: (DataFrame, str) or (None, None) if no data available
    """
    import numpy as np
    
    games = load_games()
    if games.empty:
        return None, None

    genres = game_info.get("genres") or []
    if not genres:
        return None, None

    primary_genre = genres[0]  # first genre as primary

    def has_genre(val):
        if isinstance(val, np.ndarray):
            val = val.tolist()
        if isinstance(val, list):
            return primary_genre in val
        if isinstance(val, str):
            # handle stringified lists if present
            try:
                import ast
                lst = ast.literal_eval(val)
                return primary_genre in lst
            except Exception:
                return False
        return False

    if "genre_names" not in games.columns:
        return None, None

    mask = games["genre_names"].apply(has_genre)
    df = games[mask].copy()

    # Keep only rows with valid year and aggregated_rating / rating_count
    if "release_year" in df.columns:
        df = df[df["release_year"].notna()]
    if "aggregated_rating" in df.columns:
        df = df[df["aggregated_rating"].notna()]
    if "total_rating_count" in df.columns:
        df["total_rating_count"] = df["total_rating_count"].fillna(0)

    if df.empty:
        return None, None

    return df, primary_genre


def plot_genre_scatter(game_info: dict):
    """
    Plot 1: Updated scatter - Year vs Aggregated Rating, size = rating count.
    Highlights the selected game.
    """
    df, primary_genre = get_genre_games_df(game_info)
    if df is None:
        st.info("No genre data available for this game.")
        return

    # Base scatter: one point per game, same genre
    fig = px.scatter(
        df,
        x="release_year",
        y="aggregated_rating",
        size="total_rating_count",
        hover_data=["name", "release_year", "aggregated_rating", "total_rating_count"],
        labels={
            "release_year": "Year",
            "aggregated_rating": "Aggregated Rating",
            "total_rating_count": "Rating Count",
        },
        title=f"Games in '{primary_genre}' Genre by Year, Rating, and Rating Count",
        opacity=0.4,
    )

    # Highlight selected game
    selected_id = game_info.get("id")
    if selected_id is not None and "id" in df.columns:
        selected = df[df["id"] == selected_id]
        if not selected.empty:
            fig.add_scatter(
                x=selected["release_year"],
                y=selected["aggregated_rating"],
                mode="markers+text",
                text=[game_info["name"]],
                textposition="top center",
                marker=dict(size=14, color="red"),
                name="Selected Game",
            )

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)


def plot_genre_rating_distribution(game_info: dict):
    """
    Plot 2: Rating distribution (histogram) for the genre.
    Shows a vertical line for the selected game's rating.
    """
    df, primary_genre = get_genre_games_df(game_info)
    if df is None:
        return

    if "aggregated_rating" not in df.columns:
        return

    rating_col = "aggregated_rating"
    fig = px.histogram(
        df,
        x=rating_col,
        nbins=20,
        labels={rating_col: "Aggregated Rating"},
        title=f"Rating Distribution in '{primary_genre}' Genre",
    )

    # Add vertical line for selected game's rating
    # Try to get the rating from game_info (it may be stored as string "XX.X" or float)
    selected_rating = None
    rating_val = game_info.get("rating")
    if rating_val and rating_val != "N/A":
        try:
            selected_rating = float(rating_val)
        except (TypeError, ValueError):
            pass
    
    # Fallback to total_rating if rating not available
    if selected_rating is None:
        total_rating_val = game_info.get("total_rating")
        if total_rating_val and total_rating_val != "N/A":
            try:
                selected_rating = float(total_rating_val)
            except (TypeError, ValueError):
                pass

    if selected_rating is not None:
        fig.add_vline(
            x=selected_rating,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=f"{game_info['name']} ({selected_rating:.1f})",
            annotation_position="top right",
        )

    st.plotly_chart(fig, use_container_width=True)


def plot_genre_output_over_time(game_info: dict):
    """
    Plot 3: Genre output over time (bar chart).
    Shows number of games in the genre released per year.
    """
    df, primary_genre = get_genre_games_df(game_info)
    if df is None:
        return

    if "release_year" not in df.columns:
        return

    counts = (
        df.groupby("release_year")["id"]
        .count()
        .reset_index()
        .rename(columns={"id": "num_games"})
    )

    fig = px.bar(
        counts,
        x="release_year",
        y="num_games",
        labels={"release_year": "Year", "num_games": "Number of Games"},
        title=f"Number of '{primary_genre}' Games Released per Year",
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_game_vs_genre_radar(game_info: dict):
    """
    Dark-mode radar chart comparing the selected game to its primary genre average.
    Metrics: Critic/User rating, rating count, hypes, follows (when available).
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    df, primary_genre = get_genre_games_df(game_info)
    if df is None:
        st.info("No genre data available for this game.")
        return

    if "id" not in df.columns:
        st.info("No game ID information available for radar chart.")
        return

    selected_id = game_info.get("id")
    selected_rows = df[df["id"] == selected_id]
    if selected_rows.empty:
        st.info("Could not locate this game in the genre dataset.")
        return

    selected = selected_rows.iloc[0]

    # Candidate metrics: (label, possible column names)
    metric_candidates = [
        ("Critic Rating", ["aggregated_rating", "rating", "total_rating"]),
        ("User Rating", ["total_rating"]),
        ("Rating Count", ["total_rating_count", "rating_count"]),
        ("Hypes", ["hypes"]),
        ("Follows", ["follows"]),
    ]

    labels = []
    selected_values = []
    genre_values = []

    for label, col_options in metric_candidates:
        col = next((c for c in col_options if c in df.columns), None)
        if col is None:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue

        val_sel = pd.to_numeric(pd.Series([selected.get(col)]), errors="coerce").iloc[0]
        if pd.isna(val_sel):
            continue

        min_v = series.min()
        max_v = series.max()
        mean_v = series.mean()

        if max_v == min_v:
            sel_norm = 1.0
            mean_norm = 1.0
        else:
            sel_norm = (val_sel - min_v) / (max_v - min_v)
            mean_norm = (mean_v - min_v) / (max_v - min_v)

        labels.append(label)
        selected_values.append(float(sel_norm))
        genre_values.append(float(mean_norm))

    if not labels:
        st.info("No numeric metrics available for radar chart.")
        return

    # Close polygon
    if len(labels) == 1:
        labels_closed = labels * 2
        selected_closed = selected_values * 2
        genre_closed = genre_values * 2
    else:
        labels_closed = labels + [labels[0]]
        selected_closed = selected_values + [selected_values[0]]
        genre_closed = genre_values + [genre_values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=selected_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(0,229,255,0.25)",   # cyan fill
        line=dict(color="#00E5FF", width=3),
        name=game_info["name"],
    ))

    fig.add_trace(go.Scatterpolar(
        r=genre_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(255,183,77,0.22)",  # amber fill
        line=dict(color="#FFB74D", width=3),
        name=f"{primary_genre} Avg",
    ))

    fig.update_layout(
        template="plotly_dark",
        title={
            "text": f"{game_info['name']} vs '{primary_genre}' Genre Averages",
            "font": {"size": 24, "color": "#FFFFFF"},
            "x": 0.5,
        },
        font=dict(color="#FFFFFF"),
        polar=dict(
            domain=dict(x=[0, 1], y=[0, 1]),
            radialaxis=dict(
                range=[0, 1],
                tickfont=dict(size=14, color="#FFFFFF"),
            ),
            angularaxis=dict(
                tickfont=dict(size=16, color="#FFFFFF"),
            ),
        ),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=80, b=40),
        height=550,
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # =========================================================================
    # SESSION STATE INITIALIZATION
    # =========================================================================
    # Phase control: "search", "search_results", or "details"
    if "phase" not in st.session_state:
        st.session_state.phase = "search"
    
    # Cached search results (list of dicts to avoid DataFrame copying issues)
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    
    # The committed query string for which search_results were generated
    if "search_query_committed" not in st.session_state:
        st.session_state.search_query_committed = ""
    
    # Selected and confirmed game IDs
    if "selected_game_id" not in st.session_state:
        st.session_state.selected_game_id = None
    if "confirmed_game_id" not in st.session_state:
        st.session_state.confirmed_game_id = None
    
    # =========================================================================
    # DATA AVAILABILITY CHECK (runs every time, but is lightweight - HEAD requests only)
    # =========================================================================
    with st.spinner("Checking data availability..."):
        availability = check_data_availability()
    
    # =========================================================================
    # SIDEBAR - Debug toggle only
    # =========================================================================
    with st.sidebar:
        debug_mode = st.checkbox("Debug mode", value=False)
        
        if debug_mode:
            st.markdown("### Data Status")
            for name, available in availability.items():
                status_text = "available" if available else "missing"
                st.markdown(f"- {name}: {status_text}")
            
            current_phase = st.session_state.get("phase", "unknown")
            st.markdown(f"**Phase:** {current_phase}")
    
    # =========================================================================
    # HEADER - Centered title with stats top-right
    # =========================================================================
    st.markdown(
        "<h1 style='text-align: center;'>Everything Games!</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center;'>Search for any game and discover similar titles using precomputed recommendations.</p>",
        unsafe_allow_html=True,
    )
    
    # Dataset stats in top-right area
    header_left, header_center, header_right = st.columns([1, 2, 1])
    with header_right:
        if availability.get("games", False):
            stats = get_data_stats(include_recommendations=False)
            st.markdown(
                f"""
                <div style="text-align: right; font-size: 0.85em; color: #888;">
                    <div>Total games: {stats['total_games']:,}</div>
                    <div>With ratings: {stats['games_with_ratings']:,}</div>
                    <div>With covers: {stats['games_with_covers']:,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    # Check if essential data is available
    if not availability.get("games", False):
        st.error("Game data is not available. Please check S3 bucket configuration.")
        st.info("""
        **For developers:** 
        1. Run `data_pipeline_V2.py` to generate processed data files
        2. Upload the files from `processed_data/` to S3 bucket
        3. Ensure the S3 bucket has public read access
        4. Or set `USE_LOCAL_DATA=true` environment variable for local development
        
        **S3 Bucket Required Files:**
        - `games_enriched.parquet`
        - `recommendations.parquet`
        """)
        st.stop()
    
    if not availability.get("recommendations", False):
        st.warning("Pre-computed recommendations not available. Only search will work.")
    
    # =========================================================================
    # SEARCH SECTION (Phase: "search" or "search_results")
    # =========================================================================
    st.markdown("---")
    
    # Show search UI in "search" or "search_results" phase
    if st.session_state.phase in ["search", "search_results"]:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            search_query = st.text_input(
                "Search for a game",
                value=st.session_state.search_query_committed,
                placeholder="Enter game name (e.g., 'Zelda', 'Final Fantasy', 'Portal')...",
                key="game_search"
            )
            
            # Search button - ONLY place where search_games() is called
            if st.button("Search", type="primary", use_container_width=True):
                if search_query.strip():
                    with st.spinner("Searching games..."):
                        start_time = time.time()
                        df_results = search_games(search_query.strip(), limit=20)
                        search_time = time.time() - start_time
                    
                    # Store results as list of dicts (memory efficient for session state)
                    st.session_state.search_results = df_results.to_dict("records")
                    st.session_state.search_query_committed = search_query.strip()
                    st.session_state.phase = "search_results"
                    st.session_state.selected_game_id = None
                    st.session_state.confirmed_game_id = None
                    
                    # Show search timing
                    if st.session_state.search_results:
                        st.success(f"Found {len(st.session_state.search_results)} games in {search_time*1000:.0f}ms")
                    else:
                        st.warning(f"No games found matching '{search_query.strip()}'")
                    
                    st.rerun()
                else:
                    st.warning("Please enter a game name to search.")
    
    # In "details" phase, show a "New Search" button instead
    elif st.session_state.phase == "details":
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown(f"**Current search:** *{st.session_state.search_query_committed}*")
            if st.button("New Search", use_container_width=True):
                st.session_state.phase = "search"
                st.session_state.confirmed_game_id = None
                st.session_state.selected_game_id = None
                st.session_state.search_results = []
                st.session_state.search_query_committed = ""
                st.rerun()
    
    # =========================================================================
    # DROPDOWN + VIEW GAME BUTTON (Phase: "search_results" or "details")
    # This section uses CACHED search results only - NO search_games() calls
    # =========================================================================
    if st.session_state.phase in ["search_results", "details"] and st.session_state.search_results:
        results = st.session_state.search_results
        
        # Build dropdown options from cached results
        options = []
        for row in results:
            year = row.get("release_year", "")
            year_str = f" ({int(year)})" if year and pd.notna(year) else ""
            rating = row.get("total_rating", 0)
            rating_str = f" - {rating:.1f}/100" if rating and pd.notna(rating) and rating > 0 else ""
            options.append({
                "id": row["id"],
                "label": f"{row['name']}{year_str}{rating_str}"
            })
        
        st.markdown(f"**{len(options)} games found** for *\"{st.session_state.search_query_committed}\"*")
        
        # Two-step selection: dropdown + confirm button
        col_select, col_button = st.columns([3, 1])
        
        with col_select:
            selected_label = st.selectbox(
                "Select a game:",
                options=[o["label"] for o in options],
                index=0,
                key="game_select"
            )
        
        with col_button:
            st.write("")  # Spacing to align with selectbox
            if st.button("View Game", type="primary", use_container_width=True):
                if selected_label:
                    selected_id = next(o["id"] for o in options if o["label"] == selected_label)
                    st.session_state.selected_game_id = selected_id
                    st.session_state.confirmed_game_id = selected_id
                    st.session_state.phase = "details"
                    st.rerun()
        
        # Show hint if in search_results phase (not yet confirmed)
        if st.session_state.phase == "search_results":
            st.info("Select a game from the dropdown and click View Game to see details and recommendations.")
    
    # =========================================================================
    # DETAILS + RECOMMENDATIONS / VISUALIZATIONS SECTION
    # Phase "details" = details + recommendations (no heavy plots)
    # Phase "details_visuals" = details + genre plots (no recommendations)
    # This separation prevents memory crashes on Streamlit Cloud
    # =========================================================================
    if st.session_state.phase in ["details", "details_visuals"] and st.session_state.confirmed_game_id is not None:
        st.markdown("---")
        
        # Load game details (uses cached games df from s3_loader_V2)
        with st.spinner("Loading game details..."):
            game_info = get_game_display_info(st.session_state.confirmed_game_id)
        
        if game_info:
            display_game_details(game_info)
            
            # Toggle buttons to switch between recommendations and visualizations
            st.markdown("---")
            toggle_col1, toggle_col2 = st.columns([1, 1])
            
            with toggle_col1:
                if st.session_state.phase == "details":
                    if st.button("📊 Show Visualizations", use_container_width=True):
                        st.session_state.phase = "details_visuals"
                        st.rerun()
                elif st.session_state.phase == "details_visuals":
                    if st.button("🎮 Back to Recommendations", use_container_width=True):
                        st.session_state.phase = "details"
                        st.rerun()
            
            # Recommendations section - runs ONLY in "details" phase
            if st.session_state.phase == "details":
                if availability.get("recommendations", False):
                    st.markdown("---")
                    st.markdown("## Similar Games You Might Like")
                    
                    # Fixed number of recommendations (5)
                    top_n = 5
                    
                    # Get recommendations - THIS IS THE ONLY PLACE THIS RUNS
                    with st.spinner("Finding similar games..."):
                        recommendations, rec_time = get_recommendations(
                            st.session_state.confirmed_game_id,
                            top_n=top_n,
                            method="precomputed"
                        )
                    
                    if recommendations.empty:
                        st.info("No recommendations available for this game.")
                    else:
                        # Show timing
                        st.caption(f"Found {len(recommendations)} recommendations in {rec_time*1000:.0f} ms")
                        
                        # Display recommendations in a grid-like layout
                        for idx, (_, rec_row) in enumerate(recommendations.iterrows()):
                            rec_card = format_recommendation_card(rec_row)
                            display_recommendation_card(rec_card, idx)
                else:
                    st.info("Recommendations will be available once the recommendations.parquet file is uploaded to S3.")
            
            # Genre visualizations - run ONLY in "details_visuals" phase
            if st.session_state.phase == "details_visuals":
                st.markdown("## How This Game Compares to Its Genre")
                plot_game_vs_genre_radar(game_info)   # ← NEW RADAR CHART HERE

                st.markdown("---")
                st.markdown("## Genre Popularity Over Time")
                plot_genre_scatter(game_info)

                st.markdown("---")
                st.markdown("## Rating Distribution in Genre")
                plot_genre_rating_distribution(game_info)

                st.markdown("---")
                st.markdown("## Genre Output Over Time")
                plot_genre_output_over_time(game_info)

        else:
            st.error("Could not load game details. Please try another game.")
    
    # =========================================================================
    # FEATURED GAMES SECTION (Phase: "search" only, when no search has been done)
    # =========================================================================
    if st.session_state.phase == "search" and not st.session_state.search_results:
        st.markdown("---")
        st.markdown("### Featured Games")
        st.caption("Here are some highly-rated games to explore:")
        
        featured = get_random_games(6)
        if not featured.empty:
            cols = st.columns(3)
            for idx, (_, game) in enumerate(featured.iterrows()):
                with cols[idx % 3]:
                    cover_url = game.get("cover_url", "")
                    if pd.isna(cover_url) or not cover_url:
                        cover_url = "https://via.placeholder.com/264x374?text=No+Cover"
                    
                    st.image(cover_url, width=180)
                    st.markdown(f"**{game['name']}**")
                    
                    rating = game.get("total_rating")
                    if pd.notna(rating):
                        st.caption(f"Rating: {rating:.1f}/100")
                    
                    # Button to select this featured game - sets phase to details
                    if st.button("View Details", key=f"featured_{game['id']}"):
                        # For featured games, we create a minimal search result entry
                        st.session_state.search_results = [game.to_dict()]
                        st.session_state.search_query_committed = game['name']
                        st.session_state.selected_game_id = game["id"]
                        st.session_state.confirmed_game_id = game["id"]
                        st.session_state.phase = "details"
                        st.rerun()
    
    # =========================================================================
    # FOOTER
    # =========================================================================
    st.markdown("---")
    st.markdown(
        "<center><small>Data sourced from IGDB | "
        "Recommendations powered by Sentence Transformers | "
        "Built with Streamlit</small></center>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
