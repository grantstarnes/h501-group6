"""
Streamlit App V2 Backup: Games Popularity Explorer - S3 DEMO VERSION.

This is the BACKUP/DEMO version using SMALLER dataset (~33K games with ratings).
Uses files with "_S3Implementation" suffix for reliable Streamlit Cloud deployment.

=============================================================================
TWO IMPLEMENTATIONS:
=============================================================================

1. DEMO (this app - streamlit_app_V2_backup.py):
   - Uses: s3_loader_V2_backup.py, content_recommender_V2_backup.py
   - Data: *_S3Implementation.parquet (~33K games with ratings)
   - Purpose: Demonstrate S3 bucket integration on Streamlit Cloud
   - Small file sizes, reliable deployment

2. FULL (streamlit_app_V2.py):
   - Uses: s3_loader_V2.py, content_recommender_V2.py
   - Data: games_enriched.parquet, recommendations.parquet (~220K games)
   - Purpose: Full offline capabilities with all features
   - Larger files, may need local or powerful cloud

=============================================================================
DEPLOYMENT TO STREAMLIT COMMUNITY CLOUD:
=============================================================================

1. Push this repo to GitHub

2. Go to https://share.streamlit.io

3. Click "New app" and select:
   - Repository: grantstarnes/h501-group6
   - Branch: Woods
   - Main file path: igdb_puller/streamlit_app/streamlit_app_V2_backup.py

4. No secrets needed (data is public on S3)

5. Click "Deploy"

=============================================================================
LOCAL DEVELOPMENT:
=============================================================================

To run locally with local data (after running data_pipeline_V2_backup.py):
    
    $env:USE_LOCAL_DATA = "true"
    streamlit run streamlit_app_V2_backup.py

To run locally with S3 data:

    streamlit run streamlit_app_V2_backup.py

=============================================================================
"""

import streamlit as st
import pandas as pd
import time
import plotly.express as px

# Import BACKUP V2 modules (uses _S3Implementation files)
from s3_loader_V2_backup import (
    load_games,
    search_games,
    check_data_availability,
    get_data_stats,
)
from content_recommender_V2_backup import (
    get_recommendations,
    format_recommendation_card,
    get_game_display_info,
    get_random_games,
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Everything Games! (S3 Demo)",
    page_icon="🎮",
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
        border: 1px solid #333;
    }
    
    .game-title {
        font-size: 1.5em;
        font-weight: bold;
        color: #ffffff;
    }
    
    .game-meta {
        color: #888;
        font-size: 0.9em;
    }
    
    .similarity-badge {
        background-color: #4CAF50;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        margin-left: 10px;
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
    
    /* Hide Streamlit branding */
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
            # Title with similarity badge
            title_html = f'<span class="game-title">{rec["name"]}</span>'
            if rec["similarity"]:
                title_html += f' <span class="similarity-badge">{rec["similarity"]} match</span>'
            st.markdown(title_html, unsafe_allow_html=True)
            
            # Meta info
            st.markdown(f'<span class="game-meta">{rec["year"]} | Rating: {rec["rating"]}</span>', 
                       unsafe_allow_html=True)
            
            # Genres
            if rec["genres"]:
                genre_html = " ".join([f'<span class="genre-tag">{g}</span>' for g in rec["genres"]])
                st.markdown(genre_html, unsafe_allow_html=True)
            
            # Summary preview
            if rec["summary"]:
                st.caption(rec["summary"])
        
        st.markdown("---")


def display_top_games_by_year(game: dict):
    """
    Display a scatter plot with connected points showing top-rated games by year based on aggregated rating.
    Shows an overall representation of the available data.
    
    MEMORY OPTIMIZED: Uses vectorized operations and avoids unnecessary copying
    """
 

    # Load games data from S3 (loader chooses S3 implementation when the app runs in cloud)
    games_df = load_games()
    if games_df is None or games_df.empty:
        return

    # Keep only rows with release_year and aggregated_rating
    df = games_df[games_df["release_year"].notna() & games_df["aggregated_rating"].notna()].copy()
    if df.empty or len(df) < 5:
        st.caption("Not enough data to show top games by year.")
        return

    # Ensure types
    df["release_year"] = df["release_year"].astype(int)
    df["aggregated_rating"] = pd.to_numeric(df["aggregated_rating"], errors="coerce")

    # Filter for the notebook's range (1998+)
    df = df[df["release_year"] >= 1998]
    if df.empty:
        st.caption("No games found in the data range.")
        return

    # Compute the top-rated game per year (matching Highest_Rated_Games_by_Year logic)
    top_games_by_year = (
        df.sort_values("aggregated_rating", ascending=False)
        .groupby("release_year", as_index=False)
        .first()
        .sort_values("release_year")
    )

    # Ensure numeric for plotting
    top_games_by_year["aggregated_rating"] = pd.to_numeric(top_games_by_year["aggregated_rating"], errors="coerce")

    if top_games_by_year.empty:
        st.caption("No top games available after filtering.")
        return

    # Build plotly scatter plot that mirrors 'Highest_Rated_Games_by_Year.ipynb' scatter plot
    # Custom colors for scatter plot
    colorscale = [
        [0.0, "#000040"],
        [0.5, "#0000ff"],
        [1.0, "#00ff00"],
    ]

    fig = px.scatter(
        top_games_by_year,
        x="release_year",
        y="aggregated_rating",
        color="aggregated_rating",
        color_continuous_scale=colorscale,
        hover_data=["name"],
        labels={"release_year": "Release Year", "aggregated_rating": "Aggregated Rating"},
    )

    # Add thin gray line connecting points (visual flow like the notebook)
    fig.add_scatter(
        x=top_games_by_year["release_year"],
        y=top_games_by_year["aggregated_rating"],
        mode="lines",
        line=dict(color="lightgray", width=1.5),
        showlegend=False,
    )

    # Annotate the top 3 highest-rated points
    top3 = top_games_by_year.nlargest(3, "aggregated_rating")
    for _, row in top3.iterrows():
        fig.add_annotation(
            x=row["release_year"],
            y=row["aggregated_rating"] + 0.7,
            showarrow=False,
            font=dict(size=10, color="black", family="Arial", weight="bold"),
            xanchor="center",
        )

    # Layout tuning to match notebook style
    min_year = int(top_games_by_year["release_year"].min()) if "release_year" in top_games_by_year.columns else 1998
    max_year = int(top_games_by_year["release_year"].max()) if "release_year" in top_games_by_year.columns else min_year

    fig.update_layout(
        title="Top-Rated Games by Year (1998+)",
        xaxis=dict(
            title="Release Year",
            tickmode="linear",
            dtick=5,
            range=[min_year - 0.2, max_year + 0.2],
        ),
        yaxis=dict(title="Aggregated Rating", range=[70, 105]),
        coloraxis_colorbar=dict(title="Aggregated Rating")
    )

    # Show the Plotly figure in Streamlit
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
    
    # Chart visibility toggle (to avoid loading chart + recommendations together)
    if "show_genre_chart" not in st.session_state:
        st.session_state.show_genre_chart = False
    
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
                    st.session_state.show_genre_chart = False  # Reset chart visibility
                    st.rerun()
        
        # Show hint if in search_results phase (not yet confirmed)
        if st.session_state.phase == "search_results":
            st.info("Select a game from the dropdown and click View Game to see details and recommendations.")
    
    # =========================================================================
    # DETAILS + RECOMMENDATIONS SECTION (Phase: "details" ONLY)
    # This is the ONLY place where get_recommendations() is called
    # =========================================================================
    if st.session_state.phase == "details" and st.session_state.confirmed_game_id is not None:
        st.markdown("---")
        
        # Load game details (uses cached games df from s3_loader_V2)
        with st.spinner("Loading game details..."):
            game_info = get_game_display_info(st.session_state.confirmed_game_id)
        
        if game_info:
            display_game_details(game_info)
            
            # Recommendations section - ONLY runs in "details" phase
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
            
            # Game Popularity scatter plot - OPT-IN to avoid memory spike
            # Running this with recommendations would cause memory crash on Streamlit Cloud
            st.markdown("---")
            
            if not st.session_state.show_genre_chart:
                # Show button to load chart (chart not loaded yet)
                if st.button("📊 Show Top Games by Year", use_container_width=True):
                    st.session_state.show_genre_chart = True
                    st.rerun()
            else:
                # Chart is shown - display hide button and the chart
                if st.button("❌ Hide Top Games Chart", use_container_width=True):
                    st.session_state.show_genre_chart = False
                    st.rerun()
                
                st.markdown("### Top-Rated Games by Year")
                display_top_games_by_year(game_info)
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
                        st.session_state.show_genre_chart = False  # Reset chart visibility
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