# streamlit_app/streamlit_app.py
from __future__ import annotations
import os
import math
from typing import List
from content_recommender import build_similarity, top_similar, top_similar_by_id
import streamlit as st
import pandas as pd
import plotly.express as px

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
