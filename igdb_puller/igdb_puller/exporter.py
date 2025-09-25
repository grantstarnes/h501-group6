from __future__ import annotations
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence
import csv
import pandas as pd

class Exporter:
    def write(self, rows: Iterable[Mapping], out_path: Path, columns: Optional[Sequence[str]] = None) -> Path:
        raise NotImplementedError

class CSVExporter(Exporter):
    def __init__(self, lineterminator: str = "\n"):
        self.lineterminator = lineterminator

    def write(self, rows: Iterable[Mapping], out_path: Path, columns: Optional[Sequence[str]] = None) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wrote_header = False
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = None
            for row in rows:
                if columns is None:
                    columns = list(row.keys())
                # enforce fixed schema
                for c in columns:
                    if c not in row:
                        row.setdefault(c, None)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(columns), lineterminator=self.lineterminator, quoting=csv.QUOTE_MINIMAL)
                    writer.writeheader()
                writer.writerow({c: row.get(c) for c in columns})
                wrote_header = True
        return out_path

class NDJSONExporter(Exporter):
    def write(self, rows: Iterable[Mapping], out_path: Path, columns: Optional[Sequence[str]] = None) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out_path

class ParquetExporter(Exporter):
    def write(self, rows: Iterable[Mapping], out_path: Path, columns: Optional[Sequence[str]] = None) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(list(rows))
        if columns is not None:
            for c in columns:
                if c not in df.columns:
                    df[c] = None
            df = df[list(columns)]
        df.to_parquet(out_path, index=False)
        return out_path