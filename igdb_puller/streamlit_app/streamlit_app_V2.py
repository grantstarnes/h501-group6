from __future__ import annotations
import os
from typing import Optional, Dict, Tuple

import pandas as pd
import streamlit as st

from content_recommender import (
    build_similarity,
    top_similar_by_id,
    top_similar_merlin_by_features,
    MerlinANN, MerlinConfig,
    to_query_features,
)

# ============================
# Page config & styles
# ============================

st.set_page_config(page_title="IGDB Game Finder + Recs (Baseline vs Merlin)", layout="wide")

CENTERED_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

div.block-container{
  padding-top: 1.2rem;
}

/* center the title + form */
.title-center { text-align: center; margin-bottom: 0.5rem; }
.form-wrap { display: flex; justify-content: center; }
.form-inner { width: min(900px, 95%); }
</style>
"""

st.markdown(CENTERED_CSS, unsafe_allow_html=True)

# ============================
# Data loading
# ============================

@st.cache_data(show_spinner=False)
def _load_games_df() -> pd.DataFrame:
    """
    Load your master items dataframe used by baseline + Merlin candidates.
    Expect columns: id, name, genres_lst, plats_lst, numeric cols.
    Replace below to point to your actual parquet/csv.
    """
    # Try common locations
    for pth in [
        "games_joined.parquet",
        "data/games_joined.parquet",
        "games_joined.csv",
        "data/games_joined.csv",
    ]:
        if os.path.exists(pth):
            if pth.endswith(".parquet"):
                return pd.read_parquet(pth)
            else:
                return pd.read_csv(pth)
    return pd.DataFrame(columns=["id","name","genres_lst","plats_lst","total_rating","follows","hypes","normally","completely"])  # empty fallback


@st.cache_resource(show_spinner=False)
def _baseline_model(games_df: pd.DataFrame):
    if games_df.empty:
        return games_df, None
    return build_similarity(games_df)


@st.cache_resource(show_spinner=False)
def _merlin_model() -> Optional[MerlinANN]:
    try:
        return MerlinANN(MerlinConfig(art_dir="merlin_artifacts"))
    except Exception as e:
        st.warning(f"Merlin not available: {e}")
        return None


# ============================
# Live search glue (replace with your IGDB functions)
# ============================

def live_search_game(query: str, games_df: pd.DataFrame) -> Optional[Dict]:
    """
    Session-based search: if you have an IGDB live client, replace this with that.
    For now, we try to find the best local fuzzy-ish match in games_df.
    """
    if not query:
        return None
    q = query.strip().lower()
    if games_df.empty:
        return None
    # simple contains matching, then fall back to startswith
    m = games_df[games_df["name"].astype(str).str.lower().str.contains(q, na=False)]
    if m.empty:
        m = games_df[games_df["name"].astype(str).str.lower().str.startswith(q, na=False)]
    if m.empty:
        return None
    # take highest rating as tie-breaker
    m = m.assign(_score=m.get("total_rating", pd.Series([0]*len(m))).fillna(0.0))
    row = m.sort_values(["_score","follows"], ascending=[False, False]).iloc[0].to_dict()
    return row


# ============================
# Layout: centered search + sidebar recs
# ============================

st.markdown("<h1 class='title-center'>🎮 Game Search & Recommendations</h1>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='form-wrap'><div class='form-inner'>", unsafe_allow_html=True)
    with st.form("search_form", clear_on_submit=False):
        q = st.text_input("Search a game", key="search_q")
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            submitted = st.form_submit_button("Search", type="primary")
    st.markdown("</div></div>", unsafe_allow_html=True)

# Sidebar: engine toggle + recs
with st.sidebar:
    st.header("Game Recommendations")
    engine = st.radio("Engine", ["Merlin (learned)", "Baseline (cosine)"])
    st.caption("Toggle to compare outputs.")
    rec_area = st.container()

# Load data/models
G = _load_games_df()
CLEAN, SIM = _baseline_model(G)
ANN = _merlin_model() if engine.startswith("Merlin") else None

# State for last result
if "last_game" not in st.session_state:
    st.session_state.last_game = None

# Handle search
if submitted and q:
    found = live_search_game(q, G)
    st.session_state.last_game = found

# Show details
game = st.session_state.last_game

if not game:
    st.info("Type a game name above to see details and recommendations.")
    st.stop()

# Details block (center column)
left, mid, right = st.columns([1,2,1])
with mid:
    st.subheader(game.get("name", "Selected game"))
    meta_cols = [c for c in ["total_rating","follows","hypes"] if c in game]
    if meta_cols:
        st.write({k: game.get(k) for k in meta_cols})

# Compute recommendations
with rec_area:
    try:
        if engine.startswith("Merlin") and ANN is not None:
            _, recs = top_similar_merlin_by_features(game, ANN, CLEAN, top_n=5)
        else:
            # try baseline by id if we have it, else baseline by fuzzy match id
            seed_id = int(game.get("id")) if game.get("id") is not None else None
            if seed_id is None and not G.empty:
                # map by name
                m = G[G["name"].astype(str) == game.get("name")]
                if not m.empty:
                    seed_id = int(m.iloc[0]["id"])  
            if seed_id is not None and SIM is not None:
                _, recs = top_similar_by_id(CLEAN, SIM, seed_id=seed_id, top_n=5)
            else:
                recs = pd.DataFrame(columns=["id","name","total_rating","follows","similarity"]) 
    except Exception as e:
        st.warning(f"Recommendation failed: {e}")
        recs = pd.DataFrame(columns=["id","name","total_rating","follows","similarity"]) 

    if recs is None or recs.empty:
        st.caption("No similar games found.")
    else:
        # Compact sidebar list
        for i, row in recs.head(5).iterrows():
            st.write(f"**{row.get('name','(unknown)')}**  ")
            st.caption(f"Score: {row.get('similarity', 0):.3f}  | Rating: {row.get('total_rating', float('nan'))}")
            st.divider()

# Optional: expanded table in main area
st.markdown("---")
st.subheader("Similar Games (table)")
st.dataframe(recs if recs is not None else pd.DataFrame())
