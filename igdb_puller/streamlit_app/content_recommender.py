# content_recommender.py
# Minimal, model-agnostic content-based similarity for games.
# Works with a DataFrame produced from live IGDB rows:
# required columns: id, name, genres, platforms, total_rating, follows, hypes
# optional columns: normally, completely (avg completion time)

from __future__ import annotations
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import re, numpy as np



def _norm_name(s: str) -> str:
    if s is None: return ""
    s = s.lower()
    s = re.sub(r"[™®©]", "", s)           # drop symbols
    s = re.sub(r"\s+", " ", s).strip()    # collapse spaces
    return s

def _to_list(v):
    # IGDB client flattens lists to pipe-strings; convert back
    if v is None or (isinstance(v, float) and pd.isna(v)): return []
    if isinstance(v, list): return v
    if isinstance(v, str) and "|" in v: return [x for x in v.split("|") if x]
    return [v] if v != "" else []

def build_similarity(games: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cleaned_games_df, cosine_sim_matrix)."""
    keep = ["id","name","genres","platforms","total_rating","follows","hypes","normally","completely"]
    for col in keep:
        if col not in games.columns:
            games[col] = 0 if col in ["total_rating","follows","hypes","normally","completely"] else None

    df = games.copy()
    df = df.dropna(subset=["name"])
    df["genres_lst"] = df["genres"].apply(_to_list)
    df["plats_lst"]  = df["platforms"].apply(_to_list)
    df = df.reset_index(drop=True).copy()
    # one-hot genres
    mlb = MultiLabelBinarizer()
    G = pd.DataFrame(mlb.fit_transform(df["genres_lst"]), columns=mlb.classes_, index=df.index)

    mlb_p = MultiLabelBinarizer()
    P = pd.DataFrame(mlb_p.fit_transform(df["plats_lst"]), columns=[f"p_{c}" for c in mlb_p.classes_], index=df.index)

    # numeric features (scaled)
    num_cols = ["total_rating","follows","hypes","normally","completely"]
    Xnum = df[num_cols].fillna(0.0)
    scaler = StandardScaler()
    Xnum = pd.DataFrame(scaler.fit_transform(Xnum), columns=num_cols, index=df.index)

    X = pd.concat([G,P, Xnum], axis=1)
    sim_mat = cosine_similarity(X)  # shape (n, n)
    sim = pd.DataFrame(sim_mat, index=df.index, columns=df.index)
    #sim = pd.DataFrame(cosine_similarity(X, X), index=df.index, columns=df.index)
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["name_norm"] = df["name"].astype(str).apply(_norm_name)
    return df, sim

def top_similar(df: pd.DataFrame, sim, game_name: str, top_n: int = 5) -> Tuple[str, pd.DataFrame, List[str]]:
    """Case-insensitive lookup. Returns (matched_name, recs_df, suggestions)."""
    names_lower = df["name"].str.lower()
    q = game_name.strip().lower()
    if q not in names_lower.values:
        # suggestions using substring containment
        sugg = [n for n in df["name"] if q in n.lower()]
        return "", pd.DataFrame(columns=["name","total_rating","follows"]), sugg[:10]

    idx = names_lower[names_lower == q].index[0]
    scores = sim.loc[idx].sort_values(ascending=False)
    scores = scores.drop(idx, errors="ignore")  # remove self
    top_idx = scores.head(top_n).index
    out = df.loc[top_idx, ["name","total_rating","follows"]].copy()
    out["similarity"] = scores.loc[top_idx].values
    return df.loc[idx, "name"], out.reset_index(drop=True), []

# def top_similar_by_id(df: pd.DataFrame, sim, seed_id: int, top_n: int = 5):
#     """ID-based lookup (safer than name). Returns (matched_id, recs_df)."""
#     if "id" not in df.columns:
#         raise ValueError("DataFrame missing 'id' column")
#     ids = df["id"].astype("Int64")
#     if seed_id not in ids.values:
#         return None, pd.DataFrame(columns=["id","name","total_rating","follows","similarity"])
#     idx = ids[ids == seed_id].index[0]
#     scores = sim.loc[idx].sort_values(ascending=False).drop(idx, errors="ignore")
#     top_idx = scores.head(top_n).index
#     out = df.loc[top_idx, ["id","name","total_rating","follows"]].copy()
#     out["similarity"] = scores.loc[top_idx].values
#     return int(df.loc[idx, "id"]), out.reset_index(drop=True)

import numpy as np
import pandas as pd

def top_similar_by_id(df: pd.DataFrame, sim, seed_id: int, top_n: int = 5):
    """
    Robust ID-based neighbor lookup aligned by POSITION.
    Requires that 'sim' was computed from *this same* df *in this order*.
    Returns (matched_id, recs_df) or (None, empty_df) if seed not present.
    """
    cols_out = [c for c in ["id", "name", "total_rating", "follows"] if c in df.columns]

    # 0) Normalize df index
    if not isinstance(df.index, pd.RangeIndex) or df.index.start != 0 or df.index.step != 1:
        df = df.reset_index(drop=True)

    n = len(df)
    if n == 0 or "id" not in df.columns:
        return None, pd.DataFrame(columns=cols_out + ["similarity"])

    ids = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    if seed_id not in ids.values:
        return None, pd.DataFrame(columns=cols_out + ["similarity"])

    seed_pos = int(ids[ids == seed_id].index[0])

    # 1) Get similarity row for the seed as a 1D Series indexed by 0..n-1
    if isinstance(sim, pd.DataFrame):
        # Validate sim shape and reindex by position
        if sim.shape[0] != n or sim.shape[1] != n:
            raise ValueError(f"Similarity matrix shape {sim.shape} does not match df length {n}")
        scores = sim.iloc[seed_pos].copy()
        # force positional index
        scores.index = pd.RangeIndex(n)
    else:
        sim_arr = np.asarray(sim)
        if sim_arr.ndim != 2 or sim_arr.shape[0] != n or sim_arr.shape[1] != n:
            raise ValueError(f"Similarity array shape {sim_arr.shape} does not match df length {n}")
        scores = pd.Series(sim_arr[seed_pos], index=pd.RangeIndex(n))

    # 2) Clean scores
    scores = scores.astype(float)
    scores.iloc[seed_pos] = -np.inf  # exclude seed itself
    scores.replace([np.inf, -np.inf], np.nan, inplace=True)
    scores = scores.dropna()

    # 3) Choose top-N *positions* and guard bounds
    k = min(top_n, max(0, len(scores)))
    if k == 0:
        return int(df.iloc[seed_pos]["id"]), pd.DataFrame(columns=cols_out + ["similarity"])

    top_pos = scores.nlargest(k).index.tolist()
    # filter any OOB just in case
    top_pos = [p for p in top_pos if 0 <= p < n]

    if not top_pos:
        return int(df.iloc[seed_pos]["id"]), pd.DataFrame(columns=cols_out + ["similarity"])

    # 4) Select rows by POSITION
    out = df.iloc[top_pos, :].loc[:, cols_out].copy()
    out["similarity"] = [float(scores.iloc[p]) for p in top_pos]

    matched_id = int(df.iloc[seed_pos]["id"])
    return matched_id, out.reset_index(drop=True)
