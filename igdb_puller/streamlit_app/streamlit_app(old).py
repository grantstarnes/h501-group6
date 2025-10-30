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
                st.metric("Total rating", f'{round(details["ratings"].get("total_rating"),0) or "—"}')
                st.metric("Total rating count", f'{round(details["ratings"].get("total_rating_count"),0) or "—"}')
            with meta_cols[1]:
                st.metric("Agg. rating", f'{round(details["ratings"].get("aggregated_rating"),2) or "—"}')
                st.metric("Agg. rating count", f'{round(details["ratings"].get("aggregated_rating_count"),0) or "—"}')
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

# recommendation code

with col_side:
    with st.expander("Game Recommendations", expanded=False):
        client = _client()

        # Unique keys so these don't conflict with the sidebar search
        st.subheader("Find similar games")
        rec_query = st.text_input(
            "What is your favourite game?",
            placeholder="e.g., The Witcher 3, Celeste, Elden Ring",
            key="rec_query",
        )
        rec_go = st.button("Search", type="primary", use_container_width=True, key="rec_go")
        show_debug = st.toggle("Show debug", value=False, key="rec_debug")

        if rec_query.strip():
            # run immediately when text is present (button optional)
            results = _search(rec_query.strip())
            if not results:
                st.info("No matches. Try a different title.")
            else:
                # auto-pick top hit
                selected = results[0]
                details = _details(int(selected["id"]))
                ratings = details.get("ratings", {}) or {}
                raw = details.get("raw", {}) or {}

                # Fallback if RAW is absent
                rec_game_id = int(raw["id"]) if raw.get("id") is not None else int(selected["id"])

                def _to_int_list(v):
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return []
                    if isinstance(v, list):
                        return [int(x) for x in v]
                    if isinstance(v, str) and "|" in v:
                        return [int(x) for x in v.split("|") if x]
                    try:
                        return [int(v)]
                    except Exception:
                        return []

                rec_genres = _to_int_list(raw.get("genres"))
                rec_platforms = _to_int_list(raw.get("platforms"))
                rec_rating = ratings.get("total_rating") or 0.0

                if not rec_genres:
                    st.info("Not enough metadata (genre IDs) on this title to compute similarity.")
                else:
                    fields = ",".join([
                        "id","name","genres","platforms","total_rating","total_rating_count","cover.image_id"
                    ])
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
                    seed_row = {
                            "id": rec_game_id,
                            "name": details.get("name", selected.get("name", "Unknown")),
                            "genres": rec_genres,                 # list[int]
                            "platforms": rec_platforms,           # list[int]
                            "total_rating": rec_rating,           # float
                            "total_rating_count": raw.get("total_rating_count", 0),
                            #"cover.image_id": raw.get("cover", {}).get("image_id"),
                        }
                    cand = pd.concat([pd.DataFrame([seed_row]), cand], ignore_index=True)
                    
                    def ensure_int_list(x):
                        if x is None: return []
                        if isinstance(x, list): return [int(v) for v in x]
                        if isinstance(x, str): return [int(v) for v in x.split("|") if v]
                        try: return [int(x)]
                        except: return []

                    seed_row["genres"]    = ensure_int_list(seed_row.get("genres"))
                    seed_row["platforms"] = ensure_int_list(seed_row.get("platforms"))
                    seed_row["total_rating"] = float(seed_row.get("total_rating") or 0.0)
                    seed_row["total_rating_count"] = int(seed_row.get("total_rating_count") or 0)
                    cand["genres"] = cand["genres"].apply(ensure_int_list)
                    cand["platforms"] = cand["platforms"].apply(ensure_int_list)
                    cand["normally"] = cand.get("normally", 0.0)
                    cand["completely"] = cand.get("completely", 0.0)
                    cand_nonseed = cand[cand["id"].astype(int) != rec_game_id]
                    cand_nonseed = cand_nonseed[cand_nonseed["total_rating_count"].fillna(0) > 25]
                    cand = pd.concat([pd.DataFrame([seed_row]), cand_nonseed], ignore_index=True)
                    cand = cand.drop_duplicates(subset=["id"]).reset_index(drop=True)

                    if show_debug:
                        st.caption(f"Candidates fetched: {len(cand)} (filtered by shared genres & rating_count>25)")
                        if not cand.empty:
                            st.dataframe(cand.head(25), use_container_width=True)

                    if cand.empty:
                        st.info("No similar games found from IGDB for this title.")
                    else:
                        try:
                            clean_df, sim = build_similarity(cand)

                            if rec_game_id not in clean_df["id"].astype(int).tolist():
                                st.warning("Seed game not present after cleaning/feature build.")

                            seed_id = int(rec_game_id)
                            matched_id, recs = top_similar_by_id(clean_df, sim, seed_id, top_n=5)

                            seed_name = details.get("name", selected.get("name", "Selected game"))

                        except Exception as e:
                            st.exception(e)
                            clean_df, sim, matched, recs = None, None, None, pd.DataFrame()
                        
                        #st.caption(f"[debug] df rows: {len(clean_df)}, sim shape: {getattr(sim,'shape',None)}")

                        if recs is not None and not recs.empty:
                            st.subheader(f"Top 5 games similar to **{seed_name}**")
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
                            if show_debug and not cand.empty:
                                keep_cols = [c for c in ["name","total_rating","total_rating_count"] if c in cand.columns]
                                st.dataframe(cand[keep_cols].head(50), use_container_width=True)






# The recommendation part that is not working

# with col_side:        
#     with st.expander("Game Recommendations", expanded=False):
#         client=_client()
#         st.header("Search")
#         query = st.text_input("What is your favourite game?", placeholder="e.g., The Witcher 3, Celeste, Elden Ring")
#         go = st.button("Search", type="primary", use_container_width=True)
#         if query.strip():
#             results = _search(query.strip())
#             if not results:
#                 st.info("No matches. Try a different title.")
#             else:
#                 def _to_int_list(v):
#                     if v is None or (isinstance(v, float) and pd.isna(v)):
#                         return []
#                     if isinstance(v, list):
#                         return [int(x) for x in v]
#                     if isinstance(v, str) and "|" in v:
#                         return [int(x) for x in v.split("|") if x]
#                     try:
#                         return [int(v)]
#                     except Exception:
#                         return []
#                 def _label(r):
#                     year = ""
#                     if r.get("first_release_date"):
#                         year = pd.to_datetime(r["first_release_date"], unit="s", errors="coerce").year
#                         if not (isinstance(year, int) and year > 0):
#                             year = ""
#                     rating = r.get("total_rating")
#                     rtxt = f" ({rating:.1f})" if isinstance(rating, (int,float)) and not math.isnan(rating) else ""
#                     return f'{r.get("name","")}{f" [{year}]" if year else ""}{rtxt}'
#                 if results:
#                     selected = results[0]   # automatically pick the top result
#                 else:
#                     st.warning("No results found.")
#                     #st.stop()
#                 # idx = st.selectbox("Select a game", options=list(range(len(results))), format_func=lambda i: _label(results[i]))
#                 # selected = results[idx]
#                 details = _details(int(selected["id"]))
#                 raw = details.get("raw", {})
#                 # fallback to the selected hit if raw is missing id
#                 if not raw or raw.get("id") is None:
#                     st.warning("Game details missing raw fields; using selected result id.")
#                     rec_game_id = int(selected["id"])
#                 else:
#                     rec_game_id = int(raw["id"])

#                 # IMPORTANT: pull IDs from RAW for genres/platforms
#                 rec_genres    = _to_int_list(raw.get("genres"))
#                 rec_platforms = _to_int_list(raw.get("platforms"))
#                 rec_ratings   = details["ratings"].get("total_rating") or 0.0

#                 if not rec_genres:
#                     st.info("Not enough metadata (genre IDs) on this title to compute similarity.")
#                     #st.stop()
#                 # rec_game_id=int(details.get("id"))
#                 # rec_genres= _to_int_list(details.get("genres"))
#                 # rec_platforms= _to_int_list(details.get("platforms"))
#                 # rec_ratings=details["ratings"].get("total_rating") or 0.0

#                 if not rec_genres:
#                     st.info("Not enough metadata (genres) on this title to compute similarity.")
#                 else:
#                     # Pull a live candidate pool: games sharing ANY of the selected genres
#                     # and with reasonable rating count so we avoid ultra-noisy entries.
#                     fields = ",".join([
#                         "id","name","genres","platforms","total_rating","total_rating_count","cover.image_id"
#                     ])
#                     where = f"genres = ({','.join(map(str, rec_genres))}) & total_rating_count != null & total_rating_count > 25"
#                     rows = list(client.paged(
#                         endpoint="games",
#                         fields=fields,
#                         where=where,
#                         sort="total_rating_count desc",
#                         limit=200,
#                         max_rows=800
#                     ))
#                     cand = pd.DataFrame(rows)
#                     if cand.empty:
#                         st.info("No similar games found from IGDB for this title.")
#                     else:
#                         # (optional) try to enrich with avg completion time if you decide to fetch it live
#                         # For fully live operation you can skip 'normally'/'completely' and rely on ratings & genres.
#                         cand["normally"] = 0.0
#                         cand["completely"] = 0.0

#                         # remove the selected game from candidates
#                         cand = cand[cand["id"].astype(int) != rec_game_id].copy()

#                         clean_df, sim = build_similarity(cand)
#                         matched, recs, suggestions = top_similar(clean_df, sim, details["name"], top_n=5)

#                         if suggestions:
#                             st.warning("No exact match in candidate pool. Did you mean:")
#                             st.write(", ".join(suggestions))
#                         elif not recs.empty:
#                             st.subheader(f"Top 5 games similar to **{matched}**")
#                             for _, r in recs.iterrows():
#                                 cols = st.columns([1, 5, 2])
#                                 with cols[0]:
#                                     img_id = cand.loc[cand["name"] == r["name"], "cover.image_id"].dropna().astype(str)
#                                     if not img_id.empty:
#                                         st.image(igdb_image_url(img_id.iloc[0], "t_cover_big"), use_column_width=True)
#                                     else:
#                                         st.write("No image")
#                                 with cols[1]:
#                                     st.markdown(f"**{r['name']}**")
#                                     st.caption(f"Similarity: {r['similarity']:.2f}")
#                                 with cols[2]:
#                                     tr = r["total_rating"]
#                                     st.metric("Rating", f"{tr:.1f}" if isinstance(tr, (int,float)) else "—")



# the insights code that is currently not needed

# with col_side:
#     st.subheader("Quick questions")
#     sample_n = st.slider("Sample size (top by rating count)", min_value=5000, max_value=40000, step=5000, value=20000)
#     df = _analytics_df(max_rows=sample_n)

#     q = st.radio(
#         "Choose a question",
#         options=[
#             "Which is the most rated genre?",
#             "Which year has the highest rated games?",
#             "Which platform has the best games?",
#             "Which publisher has the best games?"
#         ],
#         index=0
#     )

#     if q == "Which is the most rated genre?":
#         out = df_most_rated_genre(_client(), df).head(20)
#         fig = px.bar(out, x="avg_rating", y="genre", orientation="h", title="Average rating by genre (top 20)")
#         st.plotly_chart(fig, use_container_width=True)
#         st.dataframe(out, use_container_width=True)

#     elif q == "Which year has the highest rated games?":
#         out = df_best_year(_client(), df)
#         fig = px.line(out, x="year", y="avg_rating", markers=True, title="Average rating by release year (n≥20)")
#         st.plotly_chart(fig, use_container_width=True)
#         st.dataframe(out, use_container_width=True)

#     elif q == "Which platform has the best games?":
#         out = df_best_platform(_client(), df).head(30)
#         fig = px.bar(out, x="avg_rating", y="platform", orientation="h", title="Average rating by platform (top 30)")
#         st.plotly_chart(fig, use_container_width=True)
#         st.dataframe(out, use_container_width=True)

#     elif q == "Which publisher has the best games?":
#         out = df_best_publisher(_client(), df).head(25)
#         fig = px.bar(out, x="avg_rating", y="publisher", orientation="h", title="Average rating by publisher (n≥10)")
#         st.plotly_chart(fig, use_container_width=True)
#         st.dataframe(out, use_container_width=True)

st.markdown("---")
st.caption("Built on your IGDB client (OAuth + paging) and registry conventions for endpoints/fields.")
