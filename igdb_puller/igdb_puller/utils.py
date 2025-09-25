from __future__ import annotations
import json
from typing import Any, Dict


def flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten lists to pipe-joined strings and dicts to compact JSON strings."""
    out = {}
    for k, v in rec.items():
        if isinstance(v, list):
            out[k] = "|".join(str(x) for x in v)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, separators=(",", ":"))
        else:
            out[k] = v
    return out