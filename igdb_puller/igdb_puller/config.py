from __future__ import annotations
import os
from pathlib import Path


try:
    from dotenv import load_dotenv # type: ignore
    load_dotenv() # project-local .env if present
except Exception:
    pass


TWITCH_CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")


if not TWITCH_CLIENT_ID or not TWITCH_CLIENT_SECRET:
    raise RuntimeError("Missing TWITCH_CLIENT_ID/TWITCH_CLIENT_SECRET. Set env vars or add them to a .env file.")


DATA_DIR = Path(os.getenv("IGDB_DATA_DIR", "igdb_data"))
DATA_DIR.mkdir(exist_ok=True)