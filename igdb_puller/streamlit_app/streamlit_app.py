# streamlit_app/streamlit_app.py
from __future__ import annotations
import os
import math
from typing import List
from content_recommender import build_similarity, top_similar, top_similar_by_id
import streamlit as st
import pandas as pd
import plotly.express as px
from visuals import scatter_avg_rating_by

from igdb_helpers import (
    get_client, search_games, fetch_game_details, igdb_image_url,
    load_games_for_analytics, df_most_rated_genre, df_best_year,
    df_best_platform, df_best_publisher
)

st.set_page_config(page_title="Games Popularity Explorer", page_icon="🎮", layout="wide")

# ---------- Secrets / auth ----------
# Put these in .streamlit/secrets.toml (or set env vars in deployment):
# TWITCH_CLIENT_ID = "..."
# TWITCH_CLIENT_SECRET = "..."
CLIENT_ID = st.secrets.get("TWITCH_CLIENT_ID", os.getenv("TWITCH_CLIENT_ID", ""))
CLIENT_SECRET = st.secrets.get("TWITCH_CLIENT_SECRET", os.getenv("TWITCH_CLIENT_SECRET", ""))
if not CLIENT_ID or not CLIENT_SECRET:
    st.error("Missing Twitch credentials. Set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET in Streamlit secrets.")
    st.stop()

# ---------- Caching ----------
@st.cache_data(show_spinner=False, ttl=60*30)
def _client():
    return get_client(CLIENT_ID, CLIENT_SECRET)

@st.cache_data(show_spinner=True, ttl=60*10)
def _search(q: str):
    return search_games(_client(), q, limit=25)

@st.cache_data(show_spinner=True, ttl=60*60)
def _details(game_id: int):
    return fetch_game_details(_client(), game_id)

@st.cache_data(show_spinner=True, ttl=60*60)
def _analytics_df(max_rows: int = 20000):
    return load_games_for_analytics(_client(), max_rows=max_rows)
@st.cache_data(show_spinner=True, ttl=60*60)
def _ttb_for_ids(ids: list[int]):
    """Fetch game_time_to_beats for the given game ids, convert to hours."""
    if not ids:
        return pd.DataFrame(columns=["game_id","hastily","normally","completely","count"])
    client = _client()

    # IGDB WHERE has a size limit; chunk ids defensively
    def _chunks(seq, n=400):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

    rows = []
    for chunk in _chunks(sorted(set(int(x) for x in ids if pd.notna(x)))):
        where = f"game = ({','.join(map(str, chunk))})"
        # fields defined in the registry for this endpoint
        batch = list(client.paged(
            endpoint="game_time_to_beats",
            fields="game_id,normally,hastily,completely,count,created_at,updated_at",
            where=where,
            sort="id asc",
            limit=500,
            max_rows=None,
        ))
        rows.extend(batch)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Coerce numeric and convert likely-seconds → hours
    for c in ["hastily","normally","completely"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            q90 = df[c].quantile(0.90)
            if q90 > 2000:      # looks like seconds
                df[c] = df[c] / 3600.0
            elif 200 <= q90 <= 2000:  # looks like minutes
                df[c] = df[c] / 60.0
            # else: assume already hours
    return df
# ---------- UI ----------
# ---------- Header & centered search ----------
st.markdown(
    "<h1 style='text-align:center;margin:0.25rem 0 1rem;'>Games Popularity Explorer</h1>",
    unsafe_allow_html=True
)

c_l, c_mid, c_r = st.columns([1, 2, 1])
with c_mid:
    query = st.text_input(
        " ",  # empty label for clean look
        placeholder="Search a game (e.g., The Witcher 3, Celeste, Elden Ring)",
        key="main_query",
    )
    go = st.button("Search", type="primary", use_container_width=True, key="main_go")

st.markdown("---")

# Two-pane layout: details (left) | recommendations (right)
col_main, col_side = st.columns([9, 3])

selected = None
details = None

# ---------- LEFT: Game search & details ----------
with col_main:
    st.subheader("Game details")

    if go and query.strip():
        results = _search(query.strip())
        if not results:
            st.info("No matches. Try a different title.")
        else:
            # same selection UX as before
            def _label(r):
                year = ""
                if r.get("first_release_date"):
                    yr = pd.to_datetime(r["first_release_date"], unit="s", errors="coerce").year
                    year = yr if pd.notna(yr) else ""
                rating = r.get("total_rating")
                rtxt = f" ({rating:.1f})" if isinstance(rating, (int,float)) and not math.isnan(rating) else ""
                return f'{r.get("name","")}{f" [{year}]" if year else ""}{rtxt}'

            idx = st.selectbox(
                "Select a game",
                options=list(range(len(results))),
                format_func=lambda i: _label(results[i]),
                key="result_idx",
            )
            selected = results[idx]
            details = _details(int(selected["id"]))

            # images
            imgs = details.get("images", [])
            if imgs:
                st.image(imgs, use_column_width=True, caption=[details["name"]]*len(imgs))

            # metadata
            meta_cols = st.columns(3)
            with meta_cols[0]:
                st.metric("Total rating", f'{round(details["ratings"].get("total_rating"),0) or "—"}')
                st.metric("Total rating count", f'{round(details["ratings"].get("total_rating_count"),0) or "—"}')
            with meta_cols[1]:
                st.metric("Agg. rating", f'{round(details["ratings"].get("aggregated_rating"),2) or "—"}')
                st.metric("Agg. rating count", f'{round(details["ratings"].get("aggregated_rating_count"),0) or "—"}')
            with meta_cols[2]:
                year = pd.to_datetime(details.get("first_release_date"), unit="s", errors="coerce").year
                st.metric("First release year", year if pd.notna(year) else "—")

            st.markdown("**Genres:** " + (", ".join(details.get("genres", [])) or "—"))
            st.markdown("**Platforms:** " + (", ".join(details.get("platforms", [])) or "—"))
            if details.get("publishers"):
                st.markdown("**Publishers:** " + ", ".join(details["publishers"]))
            if details.get("developers"):
                st.markdown("**Developers:** " + ", ".join(details["developers"]))

            if details.get("summary"):
                with st.expander("Summary", expanded=True):
                    st.write(details["summary"])
            if details.get("storyline"):
                with st.expander("Storyline"):
                    st.write(details["storyline"])

            sites = details.get("websites", [])
            if sites:
                st.markdown("**Links:**")
                for url, cat in sites:
                    st.markdown(f"- [{url}]({url})")

    # --- Visual Insights (only after a game is searched) ---
        st.markdown("### Game Insights")

        if details and isinstance(details, dict) and details.get("name"):
            # Grouping control
            by_option = st.radio(
                "Group by",
                options=["Genre", "Decade", "Release Window", "Time-to-Beat"],
                index=0,
                horizontal=True,
                help="Each point represents a group's average rating vs playtime."
            )
            by_key = {
                "Genre": "genre",
                "Decade": "decade",
                "Release Window": "release_window",
                "Time-to-Beat": "time_to_beat",
            }[by_option]    

            # Time-to-Beat mode only when relevant (kept as radio)
            ttb_mode = "normally"
            if by_key == "time_to_beat":
                ttb_mode = st.radio(
                    "Time-to-Beat mode",
                    options=["hastily", "normally", "completely"],
                    index=1,
                    horizontal=True,
                    help="Select which time-to-beat column to use for the X-axis."
                )

            # Pull analytics data (your helper already exists)
            analytics_df = _analytics_df(max_rows=20000)
            ttb_df = _ttb_for_ids(analytics_df["id"].dropna().astype(int).tolist())
            if not ttb_df.empty:
                # keep only the columns we need
                ttb_df = ttb_df[["game_id","hastily","normally","completely","count"]]
                analytics_df = analytics_df.merge(ttb_df, left_on="id", right_on="game_id", how="left")

                # Derive total_playtime_hours as mean of available TTB modes
                ttb_cols = [c for c in ["hastily","normally","completely"] if c in analytics_df.columns]
                if ttb_cols:
                    analytics_df["total_playtime_hours"] = analytics_df[ttb_cols].astype(float).mean(axis=1, skipna=True)

            # (Optional) peek in debug expander
            with st.expander("Debug: Playtime merge (top 10)"):
                cols_show = ["id","name","total_rating","hastily","normally","completely","total_playtime_hours"]
                st.dataframe(analytics_df[[c for c in cols_show if c in analytics_df.columns]].head(10))
                
            # Build and render figure (searched game highlighted)
            fig, dbg = scatter_avg_rating_by(
                df=analytics_df,
                by=by_key,
                searched_game_name=details["name"],
                ttb_mode=ttb_mode,
                debug=True,
                # If your column names differ, override here:
                # rating_col="total_rating",
                # total_playtime_col="total_playtime_hours",
                # name_col="name",
                # release_date_col="first_release_date",
                # genres_col="genres",
                # ttb_cols={"hastily":"...", "normally":"...", "completely":"..."},
            )
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Debug: Visual inputs", expanded=False):
            # Basic counts
            st.write(
                f"Rows total: **{dbg.get('rows_total', 'NA')}** | "
                f"Rows after grouping: **{dbg.get('rows_used_for_grouping', 'NA')}** | "
                f"By: **{dbg.get('by', 'NA')}**"
            )
            # Non-null summary
            nn = dbg.get("nonnull_counts", {})
            if nn:
                st.write(
                    f"Non-null counts — Rating: **{nn.get('rating_nonnull', 0)}**, "
                    f"Playtime: **{nn.get('playtime_nonnull', 0)}**"
                )
                # Any TTB cols present
                ttb_present = dbg.get("ttb_present", [])
                if ttb_present:
                    st.write("TTB columns present:", ", ".join(ttb_present))
                    st.write({k: v for k, v in nn.items() if k.endswith("_nonnull")})
                else:
                    st.warning("No time-to-beat columns present to derive playtime (if missing).")

            # Show first 15 grouped rows (if any)
            gp = dbg.get("group_preview")
            if gp is not None and len(gp) > 0:
                st.caption("Grouped preview (top 15):")
                st.dataframe(gp)
            else:
                st.warning("Grouped table is empty. Likely causes:\n"
                        "- All rows have NaN for rating or derived playtime.\n"
                        "- `genres`/`release_date` missing when grouping by Genre/Decade/Release Window.\n"
                        "- After deriving playtime from TTB, still NaN because no TTB columns exist.")



# ---------- RIGHT: Recommendations for the selected game ----------
with col_side:
    st.subheader("Similar Games")
    #show_debug = st.toggle("Show debug", value=False, key="rec_debug")

    if details is None:
        st.info("Search a game to see recommendations.")
    else:
        client = _client()

        # helper
        def ensure_int_list(x):
            if x is None: return []
            if isinstance(x, list): return [int(v) for v in x]
            if isinstance(x, str): return [int(v) for v in x.split("|") if v]
            try: return [int(x)]
            except: return []

        raw = details.get("raw", {}) or {}
        ratings = details.get("ratings", {}) or {}
        rec_game_id = int(raw["id"]) if raw.get("id") is not None else int(selected["id"])
        rec_genres    = ensure_int_list(raw.get("genres"))
        rec_platforms = ensure_int_list(raw.get("platforms"))
        rec_rating    = ratings.get("total_rating") or 0.0

        if not rec_genres:
            st.info("Not enough metadata (genre IDs) on this title to compute similarity.")
        else:
            fields = ",".join(["id","name","genres","platforms","total_rating","total_rating_count","cover.image_id"])
            where = f"genres = ({','.join(map(str, rec_genres))}) & total_rating_count != null & total_rating_count > 25"

            try:
                rows = list(client.paged(
                    endpoint="games",
                    fields=fields,
                    where=where,
                    sort="total_rating_count desc",
                    limit=200,
                    max_rows=800
                ))
            except Exception as e:
                st.exception(e)
                rows = []

            cand = pd.DataFrame(rows)

            # build seed row + sanitize
            seed_row = {
                "id": rec_game_id,
                "name": details.get("name", selected.get("name", "Unknown")),
                "genres": rec_genres,
                "platforms": rec_platforms,
                "total_rating": float(rec_rating),
                "total_rating_count": int(raw.get("total_rating_count", 0)),
            }

            cand["genres"] = cand["genres"].apply(ensure_int_list)
            cand["platforms"] = cand["platforms"].apply(ensure_int_list)
            cand["normally"] = cand.get("normally", 0.0)
            cand["completely"] = cand.get("completely", 0.0)

            # keep seed + filter non-seed BEFORE featurization
            cand_nonseed = cand[cand["id"].astype(int) != rec_game_id]
            cand_nonseed = cand_nonseed[cand_nonseed["total_rating_count"].fillna(0) > 25]
            cand = pd.concat([pd.DataFrame([seed_row]), cand_nonseed], ignore_index=True)
            cand = cand.drop_duplicates(subset=["id"]).reset_index(drop=True)

            # if show_debug:
            #     st.caption(f"Candidates fetched: {len(cand)} (filtered by shared genres & rating_count>25)")
            #     if not cand.empty:
            #         st.dataframe(cand.head(25), use_container_width=True)

            if cand.empty:
                st.info("No similar games found from IGDB for this title.")
            else:
                try:
                    clean_df, sim = build_similarity(cand)
                    matched_id, recs = top_similar_by_id(clean_df, sim, int(rec_game_id), top_n=5)
                except Exception as e:
                    st.exception(e)
                    recs = pd.DataFrame()

                if recs is not None and not recs.empty:
                    for _, r in recs.iterrows():
                        cols = st.columns([5, 2])
                        with cols[0]:
                            st.markdown(f"**{r['name']}**")
                            st.caption(f"Similarity: {r['similarity']:.2f}")
                        with cols[1]:
                            tr = r.get("total_rating")
                            st.metric("Rating", f"{tr:.1f}" if isinstance(tr, (int, float)) else "—")
                else:
                    st.info("I couldn't compute any close matches from the candidate pool.")

# ---------- Footer notes ----------
st.markdown("---")
st.markdown(
    "<h2 style='margin-top:0.5rem;'>Notes</h2>",
    unsafe_allow_html=True
)

with st.expander("Current Issues", expanded=False):
    st.markdown("""
- **The search drop down**: The search results in a list of games in the dropdown but selecting those dropdown elements is unresposive.
- **Image/Artwork Missing**: Not able to pull relevent artwork for searched game. Could be a source issue or search reference issue.
- **Limited games**: A lot of searches end up unsuccessful. Need to confirm if this is a source limitation or search technique issue. 
- **Time to search**: Low end systems and weak internet connection will prolong the search and recommendations.
- **Bad recommendations**: Current logic is very rudimentary. This results in sub-optimal recommendations.
""")

with st.expander("Future Additions", expanded=False):
    st.markdown("""
- **Inline search suggestions** (type-ahead) and auto-select top match when only one strong hit exists.
- **Richer recommendations**: blend content-based features (genres/platforms) with popularity & time-to-beat signals.
- **Comparison view**: side-by-side stats (ratings, platforms, genres) for the selected game vs. recommendations.
- **Downloadables**: export current game details & rec list as CSV/JSON from the sidebar.
""")