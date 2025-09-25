from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Optional
import argparse

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
