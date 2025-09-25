# IGDB Puller

A small, modular package to fetch IGDB endpoints and export to CSV, NDJSON, or Parquet with fixed schemas and a simple CLI.

## Quick start

1. Create a `.env` (or set env vars):

```
TWITCH_CLIENT_ID=xxx
TWITCH_CLIENT_SECRET=yyy
```

2. Install (dev):

```bash
pip install -e .
```

3. Pull a table:

```bash
igdb-puller games --max-rows 10000 --fmt csv --out games.csv
```

Use defaults from the registry or override fields:

```bash
igdb-puller platforms --fields "id,name,slug,platform_family" --fmt parquet --out platforms.parquet
```

Filter with IGDB query language:

```bash
igdb-puller games --where "first_release_date >= 1609459200" --max-rows 20000
```

## Programmatic API

```python
from igdb_puller import pull_table

pull_table("genres", fmt="ndjson")
pull_table("games", fields="id,name,platforms,genres", max_rows=5000, out="games_small.csv")
```

## Extend the registry

Edit `igdb_puller/registry.py` and add a new `TableDef` entry. Then you can run:

```bash
igdb-puller new_endpoint
```

or programmatically:

```python
from igdb_puller import pull_table
pull_table("new_endpoint", fields="*")
```

## Notes
- CSV schema is fixed per run to avoid column drift.
- NDJSON avoids schema issues and is a great intermediate for later normalization.
- Parquet is columnar and fastest to load in analytics pipelines.