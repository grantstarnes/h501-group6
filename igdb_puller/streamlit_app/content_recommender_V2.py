from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# --- Baseline utilities ---
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity


# ============================
# Baseline (cosine) recommender
# ============================

def _ensure_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x) or x is None:
        return []
    # allow semicolon or comma separated strings
    if isinstance(x, str):
        if ";" in x:
            return [s.strip() for s in x.split(";") if s.strip()]
        if "," in x:
            return [s.strip() for s in x.split(",") if s.strip()]
        return [x.strip()] if x.strip() else []
    return []


BASELINE_NUM_COLS = [
    "total_rating", "follows", "hypes", "normally", "completely"
]
BASELINE_LIST_COLS = ["genres_lst", "plats_lst"]


def build_similarity(games_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare a clean feature DF and compute cosine similarity matrix.
    Expects columns: id, name, genres_lst, plats_lst, numeric cols.
    """
    df = games_df.copy()
    # Canonicalize columns
    for c in BASELINE_NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in BASELINE_LIST_COLS:
        if c not in df.columns:
            df[c] = [[] for _ in range(len(df))]

    df["genres_lst"] = df["genres_lst"].apply(_ensure_list)
    df["plats_lst"] = df["plats_lst"].apply(_ensure_list)

    # Build feature matrix
    mlb_gen = MultiLabelBinarizer(sparse_output=True)
    mlb_plat = MultiLabelBinarizer(sparse_output=True)

    gen_X = mlb_gen.fit_transform(df["genres_lst"])  # multi-hot
    plat_X = mlb_plat.fit_transform(df["plats_lst"])  # multi-hot

    num_mat = df[BASELINE_NUM_COLS].astype(float).fillna(0.0).values
    scaler = StandardScaler()
    num_X = scaler.fit_transform(num_mat)

    # Concatenate (sparse + dense)
    from scipy.sparse import hstack, csr_matrix
    feat_X = hstack([csr_matrix(gen_X), csr_matrix(plat_X), csr_matrix(num_X)])

    sim = cosine_similarity(feat_X, dense_output=False)

    # Pack encoders if you'd like to reuse later (not required here)
    df._baseline_encoders = {
        "mlb_gen": mlb_gen, "mlb_plat": mlb_plat, "scaler": scaler
    }
    return df.reset_index(drop=True), sim


def top_similar_by_id(clean_df: pd.DataFrame, sim: np.ndarray, seed_id: int, top_n: int = 5) -> Tuple[Optional[int], pd.DataFrame]:
    if "id" not in clean_df.columns or sim is None:
        return None, pd.DataFrame(columns=["id", "name", "total_rating", "follows", "similarity"]) 
    # locate row
    idx_lookup = {int(i): pos for pos, i in enumerate(clean_df["id"].astype(int))}
    if int(seed_id) not in idx_lookup:
        return None, pd.DataFrame(columns=["id", "name", "total_rating", "follows", "similarity"]) 

    i = idx_lookup[int(seed_id)]
    # sim is sparse matrix; get row
    row = sim.getrow(i)
    # top indices (excluding self)
    row.data[row.indices == i] = -1.0
    # argsort on sparse: use toarray for small K
    scores = row.toarray().ravel()
    top_idx = np.argpartition(scores, - (top_n + 1))[-(top_n + 1):]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    top_idx = [t for t in top_idx if t != i][:top_n]

    cols = [c for c in ["id", "name", "total_rating", "follows"] if c in clean_df.columns]
    out = clean_df.iloc[top_idx][cols].copy()
    out["similarity"] = scores[top_idx]
    out = out.reset_index(drop=True)
    return int(seed_id), out


# ============================
# Merlin two-tower (feature-based query) + FAISS ANN
# ============================

@dataclass
class MerlinConfig:
    art_dir: str = "merlin_artifacts"
    # names expected in items.parquet
    item_id_col: str = "id"
    # model input feature names for candidate/query towers
    cand_id_key: str = "cand_game_id"
    num_cols: Tuple[str, ...] = ("total_rating", "follows", "hypes", "normally", "completely")
    list_cols: Tuple[str, ...] = ("genres_lst", "plats_lst")


class MerlinANN:
    def __init__(self, cfg: MerlinConfig = MerlinConfig()):
        self.cfg = cfg
        try:
            import tensorflow as tf  # noqa: F401
            self.tf = tf
        except Exception as e:
            raise RuntimeError(f"TensorFlow not available: {e}")
        try:
            import faiss  # noqa: F401
            self.faiss = __import__("faiss")
        except Exception as e:
            raise RuntimeError(f"FAISS not available: {e}")

        from pathlib import Path
        art = Path(self.cfg.art_dir)
        items_path = art / "items.parquet"
        if not items_path.exists():
            raise RuntimeError(f"Missing items parquet at {items_path}")
        self.items = pd.read_parquet(items_path)
        if self.cfg.item_id_col not in self.items.columns:
            raise RuntimeError(f"items.parquet must contain '{self.cfg.item_id_col}'")
        self.items[self.cfg.item_id_col] = self.items[self.cfg.item_id_col].astype(int)

        # Load SavedModels
        self.cand_model = self.tf.saved_model.load(str(art / "saved_cand"))
        self.query_model = self.tf.saved_model.load(str(art / "saved_query"))

        # Precompute candidate embeddings using id + numerics
        self._build_candidate_index()

    def _build_candidate_index(self):
        ids = self.items[self.cfg.item_id_col].values.astype(np.int64)
        bs = 4096
        vecs = []
        for i in range(0, len(ids), bs):
            batch_ids = ids[i:i+bs]
            feed = {self.cfg.cand_id_key: self.tf.convert_to_tensor(batch_ids)}
            # Include numerics if the exported model expects them
            for c in self.cfg.num_cols:
                if c in self.items.columns:
                    feed[c] = self.tf.convert_to_tensor(self.items.iloc[i:i+bs][c].astype(float).fillna(0.0).values)
            out = self.cand_model(feed)
            v = list(out.values())[0].numpy().astype("float32")
            vecs.append(v)
        self.item_vecs = np.vstack(vecs)

        # Normalize & index (Inner Product ~= cosine after L2 norm)
        faiss = self.faiss
        faiss.normalize_L2(self.item_vecs)
        d = self.item_vecs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.item_vecs)

    # ---- Query embedding from live features ----
    def embed_query_features(self, features: Dict) -> np.ndarray:
        tf = self.tf
        # list features as ragged
        def _rag(name):
            vals = features.get(name, [])
            if not isinstance(vals, (list, tuple)):
                vals = [vals] if vals is not None else []
            # Cast to int if they look numeric
            if len(vals) and isinstance(vals[0], str):
                try:
                    vals = [int(v) for v in vals]
                except Exception:
                    pass
            return tf.ragged.constant([list(vals)], dtype=tf.int64)

        feed = {}
        for name in self.cfg.list_cols:
            feed[name] = _rag(name)
        for c in self.cfg.num_cols:
            feed[c] = tf.convert_to_tensor([float(features.get(c, 0.0))], dtype=tf.float32)

        out = self.query_model(feed)
        vec = list(out.values())[0].numpy().astype("float32")
        return vec

    def search_with_features(self, features: Dict, top_k: int = 5) -> pd.DataFrame:
        faiss = self.faiss
        qv = self.embed_query_features(features)
        faiss.normalize_L2(qv)
        D, I = self.index.search(qv, top_k)
        pos = I[0].tolist()
        scores = D[0].tolist()
        ids = self.items.iloc[pos][self.cfg.item_id_col].tolist()
        out = pd.DataFrame({"id": ids, "similarity": scores})
        return out


# ============================
# Bridges & helpers
# ============================

EXPECTED_META_COLS = ["id", "name", "total_rating", "follows"]


def to_query_features(live_game: Dict) -> Dict:
    """Map a live game JSON/dict (from your session-based search) into feature dict
    expected by Merlin query tower.
    """
    return {
        "genres_lst": _ensure_list(live_game.get("genres_lst") or live_game.get("genres") or []),
        "plats_lst": _ensure_list(live_game.get("plats_lst") or live_game.get("platforms") or []),
        "total_rating": float(live_game.get("total_rating", 0.0) or 0.0),
        "follows": float(live_game.get("follows", 0.0) or 0.0),
        "hypes": float(live_game.get("hypes", 0.0) or 0.0),
        "normally": float(live_game.get("normally", 0.0) or 0.0),
        "completely": float(live_game.get("completely", 0.0) or 0.0),
    }


def top_similar_merlin_by_features(live_game: Dict, ann: MerlinANN, meta_df: pd.DataFrame, top_n: int = 5) -> Tuple[Optional[int], pd.DataFrame]:
    try:
        qf = to_query_features(live_game)
        sims = ann.search_with_features(qf, top_k=top_n)
    except Exception:
        return None, pd.DataFrame(columns=["id", "name", "total_rating", "follows", "similarity"]) 

    cols = [c for c in EXPECTED_META_COLS if c in meta_df.columns]
    out = sims.merge(meta_df[cols], on="id", how="left")
    out = out[ [c for c in ["id","name","total_rating","follows","similarity"] if c in out.columns] ]
    return live_game.get("id"), out.reset_index(drop=True)


# ============================
# Light in-module smoke test hooks
# ============================

if __name__ == "__main__":
    print("This module provides baseline + Merlin ANN utilities. Import and use from your app.")
