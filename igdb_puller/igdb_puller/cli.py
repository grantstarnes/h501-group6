from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Optional
import argparse
import inspect
import ast
import re
import numpy as np
import pandas as pd
from requests import HTTPError

from .client import IGDBClient
from .exporter import CSVExporter, NDJSONExporter, ParquetExporter
from .registry import TABLES


EXPORTERS = {
    "csv": CSVExporter,
    "ndjson": NDJSONExporter,
    "parquet": ParquetExporter,
}

def pull_table(endpoint: str, fields: Optional[str] = None, where: Optional[str] = None, sort: str = "id asc",
               out: Optional[str] = None, fmt: str = "csv", max_rows: Optional[int] = None, rate_sleep: float = 0.35) -> Path:
    """Programmatic API: fetch an endpoint and export to a file."""
    client = IGDBClient(rate_sleep=rate_sleep)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    if endpoint in TABLES and not fields:
        fields = TABLES[endpoint].fields
        sort = sort or TABLES[endpoint].sort
    if not fields:
        fields = "*"  # caller really wants everything

    rows = client.paged(endpoint=endpoint, fields=fields, where=where, sort=sort, max_rows=max_rows)

    exporter_cls = EXPORTERS.get(fmt.lower())
    if not exporter_cls:
        raise ValueError(f"Unknown format: {fmt} (choose from {list(EXPORTERS)})")

    if out is None:
        out = f"igdb_{endpoint}_{ts}.{fmt.lower()}"
    out_path = Path(out)

    # Canonical column order (if specified explicitly)
    columns = [c.strip() for c in fields.split(",")] if fields and fields != "*" else None

    exporter = exporter_cls()
    return exporter.write(rows, out_path, columns=columns)


def pull_tables_as_globals(
    endpoints: list[str],
    *,
    fields_map: Optional[dict[str, str]] = None,
    where_map: Optional[dict[str, Optional[str]]] = None,
    rename_overrides: Optional[dict[str, str]] = None,
    max_rows_map: Optional[dict[str, int]] = None,
    rate_sleep: float = 0.35,
    add_df_prefix: bool = True,
    strip_first_token: bool = False,
    verbose: bool = True,
    target_ns: Optional[dict] = None,
    all_fields: bool = True,  
):
    """
    Pull endpoints straight into pandas and create variables like df_games.
    If target_ns is None, inject into the caller's globals (e.g., your notebook).
    """
    fields_map = fields_map or {}
    where_map = where_map or {}
    rename_overrides = rename_overrides or {}
    max_rows_map = max_rows_map or {}

    # NEW: resolve the namespace we’ll assign into
    if target_ns is None:
        # the frame of whoever called this function
        target_ns = inspect.currentframe().f_back.f_globals

    client = IGDBClient(rate_sleep=rate_sleep)
    created = []
    name_to_df = {}  # optional: return these for convenience

    for endpoint in endpoints:
        # Force 'fields *' for every call; ignore registry/table defaults.
        fields = "*"
        sort = "id asc"

        df = client.fetch_df(
            endpoint=endpoint,
            fields=fields,
            where=where_map.get(endpoint),
            sort=sort,
            max_rows=max_rows_map.get(endpoint),
        )

        base = rename_overrides.get(endpoint, endpoint)
        name_base = base.split("_", 1)[1] if (strip_first_token and "_" in base) else base
        safe_base = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name_base) or "df"
        var_name = f"df_{safe_base}" if add_df_prefix else safe_base

        # ensure uniqueness in the *target* namespace
        final_name = var_name
        i = 2
        while final_name in target_ns:
            final_name = f"{var_name}_{i}"
            i += 1

        # assign into the caller's namespace
        target_ns[final_name] = df
        name_to_df[final_name] = df

        created.append((final_name, endpoint, df.shape))
        if verbose:
            print(f"Created {final_name:<35} from '{endpoint}'  shape={df.shape}")

    # optionally return a dict you can use programmatically too
    return name_to_df


# ---------- helpers for ID extraction ----------
def _ensure_listish(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    if isinstance(x, (int, np.integer)):
        return [int(x)]
    if isinstance(x, float):
        if np.isnan(x):
            return []
        return [int(x)]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, (list, tuple, set)):
                    return list(val)
            except Exception:
                pass
        return [int(tok) for tok in re.findall(r"\d+", s)]
    return []

def ids_from_column(df: pd.DataFrame, col: str) -> list[int]:
    if col not in df.columns:
        return []
    out = []
    for v in df[col].tolist():
        out.extend(_ensure_listish(v))
    return sorted({int(i) for i in out if str(i).isdigit()})

def _chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

# ---------- dependency configuration ----------
DIRECT_BY_GAME = [
    "artworks","covers","external_games","game_localizations","involved_companies",
    "language_supports","multiplayer_modes","release_dates","screenshots","game_videos",
    "websites",
]
DIRECT_BY_GAME_ID = ["game_time_to_beats","popularity_primitives"]

LOOKUPS_FROM_GAMES = {
    # endpoint            column on games
    "age_ratings":        "age_ratings",
    "genres":             "genres",
    "keywords":           "keywords",
    "platforms":          "platforms",
    "player_perspectives":"player_perspectives",
    "game_modes":         "game_modes",
    "franchises":         "franchises",   # adjust to 'franchise' if your schema uses singular
    "collections":        "collection",   # games.collection is a single id
}

# second-order lookups we can optionally pull after first-order tables are loaded
SECOND_ORDER = {
    # from involved_companies -> companies, then company_logos, company_websites
    "companies": {"source_df": "df_involved_companies", "source_col": "company", "filter": "id"},
    "company_logos": {"source_df": "df_companies", "source_col": "logo", "filter": "id"},
    "company_websites": {"source_df": "df_companies", "source_col": "id", "filter": "company"},

    # from platforms -> platform_* (pull selectively as needed)
    "platform_logos": {"source_df": "df_platforms", "source_col": "logo", "filter": "id"},
    "platform_versions": {"source_df": "df_platforms", "source_col": "versions", "filter": "id"},
    "platform_version_release_dates": {"source_df": "df_platform_versions", "source_col": "id", "filter": "platform_version"},
    "platform_websites": {"source_df": "df_platforms", "source_col": "id", "filter": "platform"},
    "platform_families": {"source_df": "df_platforms", "source_col": "platform_family", "filter": "id"},
    "platform_types": {"source_df": "df_platforms", "source_col": "category", "filter": "id"},  # if needed

    # from release_dates -> release_date_* (regions/statuses/date_formats)
    "release_date_regions": {"source_df": "df_release_dates", "source_col": "region", "filter": "id"},
    "release_date_statuses": {"source_df": "df_release_dates", "source_col": "status", "filter": "id"},
    "date_formats": {"source_df": "df_release_dates", "source_col": "date_format", "filter": "id"},

    # from websites -> website_types
    "website_types": {"source_df": "df_websites", "source_col": "category", "filter": "id"},
}




def _pull_fk_batches(
    endpoints: list[str],
    fk_col: str,                 # e.g., "game_id", "game", "company", ...
    ids: list[int],
    *,
    batch_size: int,
    target_ns: dict,
    verbose: bool = True,
):
    """
    Pull endpoints filtered by fk_col = (ids...) in batches.
    If a batch triggers HTTP 413 (too large) or 400 (body too big/malformed),
    automatically halves the batch and retries.
    """
    if not ids:
        return
    cur = max(50, batch_size)

    i = 0
    while i < len(ids):
        chunk = ids[i:i+cur]
        ids_str = ",".join(map(str, chunk))
        wm = {t: f"{fk_col} = ({ids_str})" for t in endpoints}
        try:
            pull_tables_as_globals(endpoints, where_map=wm, target_ns=target_ns, verbose=verbose)
            i += cur
        except HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code in (400, 413) and cur > 50:
                cur = max(50, cur // 2)
                if verbose:
                    print(f"[{code}] Shrinking batch to {cur} and retrying…")
            else:
                raise

def pull_games_and_dependents(
    *,
    where_games: Optional[str] = None,
    all_fields: bool = False,
    batch_size: int = 8000,
    include_second_order: bool = True,
    target_ns: Optional[dict] = None,
    max_games: Optional[int] = None,
    verbose: bool = True,
):
    """
    One-call loader:
      - pulls `games` (unless df_games already exists),
      - pulls children with game or game_id,
      - pulls lookups referenced by games,
      - optionally pulls second-order lookups (companies/platforms/etc.).
    """
    if target_ns is None:
        target_ns = inspect.currentframe().f_back.f_globals


    if "df_games" not in target_ns:
        # if not all_fields:
        #     # a reasonable default list (edit as needed)
        #     fields = ",".join([
        #         "id","name","first_release_date","popularity","total_rating","total_rating_count",
        #         "age_ratings","genres","keywords","platforms","player_perspectives",
        #         "game_modes","franchises","collection","involved_companies","external_games"
        #     ])
        # fields_map = {"games": fields}
        where_map = {"games": where_games} if where_games else None
        max_rows_map = {"games": max_games} if max_games else None
        pull_tables_as_globals(["games"],  where_map=where_map,max_rows_map=max_rows_map,
                                target_ns=target_ns, verbose=verbose)

    df_games = target_ns["df_games"]
    game_ids = df_games["id"].dropna().astype(int).tolist()

    # 2) direct children by game_id
    _pull_fk_batches(
    DIRECT_BY_GAME_ID, fk_col="game_id", ids=game_ids,
    batch_size=batch_size, target_ns=target_ns, verbose=verbose)

    # 3) direct children by game
    _pull_fk_batches(
    DIRECT_BY_GAME, fk_col="game", ids=game_ids,
    batch_size=batch_size, target_ns=target_ns, verbose=verbose)

    # 4) lookups from columns on games
    for endpoint, col in LOOKUPS_FROM_GAMES.items():
        ids = ids_from_column(df_games, col)
        if not ids:
            continue
        for batch in _chunks(ids, batch_size):
            ids_str = ",".join(map(str, batch))
            wm = {endpoint: f"id = ({ids_str})"}
            pull_tables_as_globals([endpoint], where_map=wm,
                                    target_ns=target_ns, verbose=verbose)

    # 5) optional: second-order lookups
    if include_second_order:
        for endpoint, spec in SECOND_ORDER.items():
            src_name = spec["source_df"]
            if src_name not in target_ns:
                continue
            src_df = target_ns[src_name]
            ids = ids_from_column(src_df, spec["source_col"])
            if not ids:
                continue
            fk = spec["filter"]  # e.g., "company", "platform", or "id"
        _pull_fk_batches(
            [endpoint],
            fk_col=fk,
            ids=ids,
            batch_size=batch_size,       # adaptive shrink will handle 400/413
            target_ns=target_ns,
            verbose=verbose,
        )

    if verbose:
        print("Done pulling games and dependents.")



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="igdb-puller", description="Download IGDB tables to CSV/NDJSON/Parquet")
    p.add_argument("endpoint", help="IGDB endpoint (e.g., games, platforms, genres)")
    p.add_argument("--fields", help="Comma-separated field list (default: from registry or * if not found)")
    p.add_argument("--where", help="Optional IGDB where clause", default=None)
    p.add_argument("--sort", help="Sort clause (default: id asc)", default="id asc")
    p.add_argument("--out", help="Output file path (default: auto with timestamp)")
    p.add_argument("--fmt", help="csv | ndjson | parquet", default="csv")
    p.add_argument("--max-rows", type=int, help="Max rows to fetch (default: unlimited)")
    p.add_argument("--rate-sleep", type=float, default=0.35, help="Delay between pages to respect rate limits")
    return p


def main(argv: Optional[list] = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    out = pull_table(
        endpoint=args.endpoint,
        fields=args.fields,
        where=args.where,
        sort=args.sort,
        out=args.out,
        fmt=args.fmt,
        max_rows=args.max_rows,
        rate_sleep=args.rate_sleep,
    )
    print(out)

if __name__ == "__main__":
    main()
