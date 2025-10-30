# streamlit_app/recommend_tab.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data(show_spinner=True)
def load_games_data():
    games = pd.read_csv("games.csv")
    ttb = pd.read_csv("game_time_to_beats.csv")

    # Basic clean-up
    games = games[["id", "name", "genres", "platforms", "total_rating", "follows", "hypes"]].fillna(0)
    games = games.dropna(subset=["name"])

    # Merge playtime data
    ttb_agg = ttb.groupby("game_id")[["normally", "completely"]].mean().reset_index()
    ttb_agg.rename(columns={"game_id": "id"}, inplace=True)
    games = games.merge(ttb_agg, on="id", how="left")

    # Process genres
    games["genres"] = games["genres"].fillna("").apply(lambda x: x.split("|") if isinstance(x, str) else [])
    mlb = MultiLabelBinarizer()
    genre_features = pd.DataFrame(mlb.fit_transform(games["genres"]), columns=mlb.classes_)

    # Scale numeric features
    num_cols = ["total_rating", "follows", "hypes", "normally", "completely"]
    num_scaled = pd.DataFrame(StandardScaler().fit_transform(games[num_cols]), columns=num_cols)

    # Combine features and compute similarity
    X = pd.concat([genre_features, num_scaled], axis=1)
    similarity_matrix = cosine_similarity(X, X)
    return games, similarity_matrix

def render_recommend_tab():
    st.header("Game Recommendations")

    games, sim_matrix = load_games_data()

    query = st.text_input("Search for a game", placeholder="Type part of a title (case-insensitive)")

    if query:
        matches = [g for g in games["name"] if query.lower() in g.lower()]
        if not matches:
            st.warning("No matching games found. Try a different keyword.")
            return

        selected = st.selectbox("Select a game", matches)
        if selected:
            idx = games[games["name"] == selected].index[0]
            sims = list(enumerate(sim_matrix[idx]))
            sims = sorted(sims, key=lambda x: x[1], reverse=True)
            top_idx = [i for i, _ in sims[1:6]]

            recs = games.iloc[top_idx][["name", "total_rating", "follows"]]
            st.subheader(f"Top 5 games similar to **{selected}**:")
            st.dataframe(recs, use_container_width=True)
