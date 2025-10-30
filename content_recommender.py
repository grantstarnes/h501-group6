# ----------------------------------------------------------
# Content-Based Game Recommender using IGDB data
# ----------------------------------------------------------

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----------------- 1. Load datasets -----------------
games = pd.read_csv("games.csv")
genres = pd.read_csv("genres.csv")
time_to_beat = pd.read_csv("game_time_to_beats.csv")

# Keep only necessary columns
games = games[[
    "id", "name", "genres", "platforms", 
    "total_rating", "follows", "hypes"
]].dropna(subset=["name"]).fillna(0)

# ----------------- 2. Merge playtime data -----------------
# Aggregate average completion time for each game_id
avg_playtime = time_to_beat.groupby("game_id")[["normally", "completely"]].mean().reset_index()
avg_playtime.rename(columns={"game_id": "id"}, inplace=True)

# Merge to games
games = games.merge(avg_playtime, on="id", how="left")

# ----------------- 3. Process genres -----------------
# Convert pipe-separated strings to lists
games["genres"] = games["genres"].fillna("").apply(lambda x: x.split("|") if isinstance(x, str) else [])

mlb = MultiLabelBinarizer()
genre_features = pd.DataFrame(mlb.fit_transform(games["genres"]), columns=mlb.classes_)

# ----------------- 4. Combine numerical features -----------------
num_features = games[["total_rating", "follows", "hypes", "normally", "completely"]].fillna(0)
scaler = StandardScaler()
num_scaled = pd.DataFrame(scaler.fit_transform(num_features), columns=num_features.columns)

# Final feature matrix
X = pd.concat([genre_features, num_scaled], axis=1)

# ----------------- 5. Compute cosine similarity -----------------
similarity_matrix = cosine_similarity(X, X)

# ----------------- 6. Define recommender function -----------------
def recommend(game_name, top_n=5):
    # Make comparison case-insensitive
    game_name_lower = game_name.lower()
    games_lower = games["name"].str.lower()

    if game_name_lower not in games_lower.values:
        print(f"Game '{game_name}' not found. Try checking your spelling.")
        # Optional: suggest closest matches
        close_matches = [g for g in games["name"] if game_name_lower in g.lower()]
        if close_matches:
            print("\nDid you mean:")
            for g in close_matches[:5]:
                print(f"  - {g}")
        return

    idx = games_lower[games_lower == game_name_lower].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i for i, _ in sim_scores[1:top_n+1]]
    recommendations = games.iloc[top_indices][["name", "total_rating", "follows"]]

    print(f"\nTop {top_n} similar games to '{games.iloc[idx]['name']}':\n")
    print(recommendations.to_string(index=False))

# ----------------- 7. Example usage -----------------
if __name__ == "__main__":
    sample_game = "The Witcher 3: Wild Hunt"
    recommend(sample_game, top_n=5)
