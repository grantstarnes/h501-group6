from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple
import merlin.io
import merlin.models.tf as mm
import tensorflow as tf
import numpy as np
import pandas as pd

# Local imports (from your project)
from content_recommender import build_similarity, _ensure_list, BASELINE_NUM_COLS, BASELINE_LIST_COLS


def _load_games_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Games file not found: {path}")
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    # ensure expected columns
    for c in BASELINE_NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in BASELINE_LIST_COLS:
        if c not in df.columns:
            df[c] = [[] for _ in range(len(df))]
    df["genres_lst"] = df["genres_lst"].apply(_ensure_list)
    df["plats_lst"] = df["plats_lst"].apply(_ensure_list)
    return df


def make_pairs(clean_df: pd.DataFrame, sim, top_k_pos: int = 5, neg_per_pos: int = 3) -> pd.DataFrame:
    ids = clean_df["id"].dropna().astype(int).tolist()
    # build index lookup
    idx_lookup = {int(i): pos for pos, i in enumerate(clean_df["id"].astype(int))}

    data = []
    for seed in ids:
        i = idx_lookup.get(int(seed))
        if i is None:
            continue
        row = sim.getrow(i).toarray().ravel()
        row[i] = -1.0
        # top positives
        top_idx = np.argpartition(row, -(top_k_pos + 1))[-(top_k_pos + 1):]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        top_idx = [t for t in top_idx if t != i][:top_k_pos]
        pos_ids = clean_df.iloc[top_idx]["id"].astype(int).tolist()
        for cid in pos_ids:
            data.append((int(seed), int(cid), 1))
        # negatives
        pool = [x for x in ids if x not in pos_ids and x != int(seed)]
        if pool:
            k = min(len(pool), top_k_pos * neg_per_pos)
            negs = np.random.choice(pool, size=k, replace=False)
            for cid in negs:
                data.append((int(seed), int(cid), 0))

    pairs = pd.DataFrame(data, columns=["query_game_id", "cand_game_id", "label"]).drop_duplicates()
    return pairs


def export_artifacts(clean_df: pd.DataFrame, pairs: pd.DataFrame, out_dir: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # items table for candidates
    keep_cols = ["id", "name", "genres_lst", "plats_lst"] + [c for c in BASELINE_NUM_COLS if c in clean_df.columns]
    items = clean_df[keep_cols].copy()
    items.to_parquet(out / "items.parquet", index=False)

    # enrich pairs with query features (needed for feature-based query tower)
    qcols = ["genres_lst", "plats_lst"] + [c for c in BASELINE_NUM_COLS if c in clean_df.columns]
    df_q = clean_df[["id"] + qcols].rename(columns={"id": "query_game_id"})
    pairs_enriched = pairs.merge(df_q, on="query_game_id", how="left")
    pairs_enriched.to_parquet(out / "pairs.parquet", index=False)
    return out


def train_merlin(art_dir: str, embedding_dim: int = 64, batch_size: int = 4096, epochs: int = 3) -> None:


    art = Path(art_dir)
    items = merlin.io.Dataset(art / "items.parquet")
    pairs = merlin.io.Dataset(art / "pairs.parquet")

    num_cols = [c for c in BASELINE_NUM_COLS]

    # ---- Define input blocks ----
    # Query tower uses feature-based inputs (lists + numerics)
    gen_cfg = mm.CategoricalFeatureConfig("genres_lst", max_size=1 << 15, embedding_dim=embedding_dim, is_list=True)
    plt_cfg = mm.CategoricalFeatureConfig("plats_lst", max_size=1 << 15, embedding_dim=embedding_dim, is_list=True)

    qry_num = mm.InputBlockV2({c: mm.ContinuousFeatureConfig(c) for c in num_cols})
    qry_cat = mm.InputBlockV2({"genres_lst": gen_cfg, "plats_lst": plt_cfg})
    qry_inputs = qry_cat + qry_num
    qry_tower = mm.Encoder(qry_inputs, mm.MLPBlock([128, embedding_dim]))

    # Candidate tower uses ID + numerics
    cand_id_cfg = mm.CategoricalFeatureConfig("cand_game_id", embedding_dim=embedding_dim)
    cand_id = mm.InputBlockV2({"cand_game_id": cand_id_cfg})
    cand_num = mm.InputBlockV2({c: mm.ContinuousFeatureConfig(c) for c in num_cols})
    cand_inputs = cand_id + cand_num
    cand_tower = mm.Encoder(cand_inputs, mm.MLPBlock([128, embedding_dim]))

    task = mm.ItemRetrievalTask(sampling_uniform=True, metrics=[mm.RecallAt(10), mm.RecallAt(20)])
    model = mm.TwoTowerModel(qry_tower, cand_tower, task=task)
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4))

    # Fit
    model.fit(pairs, batch_size=batch_size, epochs=epochs)

    # Export SavedModels
    mm.ToSavedModel(qry_tower, schema=pairs.schema.select_by_name(["genres_lst", "plats_lst"] + num_cols)).save(str(art / "saved_query"))
    mm.ToSavedModel(cand_tower, schema=pairs.schema.select_by_name(["cand_game_id"] + num_cols)).save(str(art / "saved_cand"))


def build_faiss_index(art_dir: str) -> None:
    import tensorflow as tf
    import faiss

    art = Path(art_dir)
    items = pd.read_parquet(art / "items.parquet")
    cand = tf.saved_model.load(str(art / "saved_cand"))

    ids = items["id"].astype(np.int64).values
    num = items[[c for c in BASELINE_NUM_COLS if c in items.columns]].astype(float).fillna(0.0)

    vecs = []
    bs = 4096
    for i in range(0, len(ids), bs):
        batch_ids = ids[i:i+bs]
        feed = {"cand_game_id": tf.convert_to_tensor(batch_ids)}
        for c in BASELINE_NUM_COLS:
            if c in num.columns:
                feed[c] = tf.convert_to_tensor(num.iloc[i:i+bs][c].values, dtype=tf.float32)
        out = cand(feed)
        v = list(out.values())[0].numpy().astype("float32")
        vecs.append(v)
    item_vecs = np.vstack(vecs)

    faiss.normalize_L2(item_vecs)
    d = item_vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(item_vecs)

    faiss.write_index(index, str(art / "faiss_index.bin"))
    np.save(art / "item_ids.npy", ids)


def main():
    ap = argparse.ArgumentParser(description="Build Merlin artifacts: pairs ➜ train ➜ export ANN")
    ap.add_argument("--games", required=True, help="Path to items dataset (parquet or csv)")
    ap.add_argument("--out", default="merlin_artifacts", help="Output directory for artifacts")
    ap.add_argument("--top_k_pos", type=int, default=5)
    ap.add_argument("--neg_per_pos", type=int, default=3)
    ap.add_argument("--emb", type=int, default=64, help="Embedding dim")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    games_df = _load_games_df(args.games)
    clean_df, sim = build_similarity(games_df)
    pairs = make_pairs(clean_df, sim, top_k_pos=args.top_k_pos, neg_per_pos=args.neg_per_pos)
    export_artifacts(clean_df, pairs, args.out)
    train_merlin(args.out, embedding_dim=args.emb, batch_size=args.batch, epochs=args.epochs)
    build_faiss_index(args.out)
    print(f"Done. Artifacts in: {args.out}")


if __name__ == "__main__":
    main()
