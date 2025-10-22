# streamlit_app/streamlit_app.py
from __future__ import annotations
import os
import math
from typing import List

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
st.title("Games Popularity Explorer")

with st.sidebar:
    st.header("Search")
    query = st.text_input("Find a game", placeholder="e.g., The Witcher 3, Celeste, Elden Ring")
    go = st.button("Search", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Quick visuals powered by IGDB ratings. Increase sample size on the main page if needed.")

col_main, col_side = st.columns([7, 5])

with col_main:
    st.subheader("Game search & details")
    if go and query.strip():
        results = _search(query.strip())
        if not results:
            st.info("No matches. Try a different title.")
        else:
            # selection list with cover thumbs
            def _label(r):
                year = ""
                if r.get("first_release_date"):
                    year = pd.to_datetime(r["first_release_date"], unit="s", errors="coerce").year
                    if not (isinstance(year, int) and year > 0):
                        year = ""
                rating = r.get("total_rating")
                rtxt = f" ({rating:.1f})" if isinstance(rating, (int,float)) and not math.isnan(rating) else ""
                return f'{r.get("name","")}{f" [{year}]" if year else ""}{rtxt}'
            idx = st.selectbox("Select a game", options=list(range(len(results))), format_func=lambda i: _label(results[i]))
            selected = results[idx]
            details = _details(int(selected["id"]))

            # images
            imgs: List[str] = details.get("images", [])
            if imgs:
                st.image(imgs, use_column_width=True, caption=[details["name"]]*len(imgs))

            # metadata
            meta_cols = st.columns(3)
            with meta_cols[0]:
                st.metric("Total rating", f'{details["ratings"].get("total_rating") or "—"}')
                st.metric("Total rating count", f'{details["ratings"].get("total_rating_count") or "—"}')
            with meta_cols[1]:
                st.metric("Agg. rating", f'{details["ratings"].get("aggregated_rating") or "—"}')
                st.metric("Agg. rating count", f'{details["ratings"].get("aggregated_rating_count") or "—"}')
            with meta_cols[2]:
                year = pd.to_datetime(details.get("first_release_date"), unit="s", errors="coerce").year
                st.metric("First release year", year if pd.notna(year) else "—")

            # tags
            st.markdown("**Genres:** " + (", ".join(details.get("genres", [])) or "—"))
            st.markdown("**Platforms:** " + (", ".join(details.get("platforms", [])) or "—"))
            if details.get("publishers"):
                st.markdown("**Publishers:** " + ", ".join(details["publishers"]))
            if details.get("developers"):
                st.markdown("**Developers:** " + ", ".join(details["developers"]))

            # summary / storyline
            if details.get("summary"):
                with st.expander("Summary", expanded=True):
                    st.write(details["summary"])
            if details.get("storyline"):
                with st.expander("Storyline"):
                    st.write(details["storyline"])

            # websites
            sites = details.get("websites", [])
            if sites:
                st.markdown("**Links:**")
                for url, cat in sites:
                    st.markdown(f"- [{url}]({url})")

with col_side:
    st.subheader("Quick questions")
    sample_n = st.slider("Sample size (top by rating count)", min_value=5000, max_value=40000, step=5000, value=20000)
    df = _analytics_df(max_rows=sample_n)

    q = st.radio(
        "Choose a question",
        options=[
            "Which is the most rated genre?",
            "Which year has the highest rated games?",
            "Which platform has the best games?",
            "Which publisher has the best games?"
        ],
        index=0
    )

    if q == "Which is the most rated genre?":
        out = df_most_rated_genre(_client(), df).head(20)
        fig = px.bar(out, x="avg_rating", y="genre", orientation="h", title="Average rating by genre (top 20)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(out, use_container_width=True)

    elif q == "Which year has the highest rated games?":
        out = df_best_year(_client(), df)
        fig = px.line(out, x="year", y="avg_rating", markers=True, title="Average rating by release year (n≥20)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(out, use_container_width=True)

    elif q == "Which platform has the best games?":
        out = df_best_platform(_client(), df).head(30)
        fig = px.bar(out, x="avg_rating", y="platform", orientation="h", title="Average rating by platform (top 30)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(out, use_container_width=True)

    elif q == "Which publisher has the best games?":
        out = df_best_publisher(_client(), df).head(25)
        fig = px.bar(out, x="avg_rating", y="publisher", orientation="h", title="Average rating by publisher (n≥10)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(out, use_container_width=True)

st.markdown("---")
st.caption("Built on your IGDB client (OAuth + paging) and registry conventions for endpoints/fields.")
