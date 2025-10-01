from __future__ import annotations
import time
import requests
import pandas as pd
from typing import Dict, Iterable, Iterator, Optional, List
from .config import TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET
from .utils import flatten_record

TOKEN_URL = "https://id.twitch.tv/oauth2/token"
IGDB_BASE = "https://api.igdb.com/v4"

class IGDBClient:
    def __init__(self, rate_sleep: float = 0.35, session: Optional[requests.Session] = None):
        self.rate_sleep = rate_sleep
        self.session = session or requests.Session()
        self._token = None

    # ------------------------ Auth ------------------------
    def _get_token(self) -> str:
        if self._token:
            return self._token
        resp = self.session.post(
            TOKEN_URL,
            params={
                "client_id": TWITCH_CLIENT_ID,
                "client_secret": TWITCH_CLIENT_SECRET,
                "grant_type": "client_credentials",
            },
            timeout=30,
        )
        resp.raise_for_status()
        self._token = resp.json()["access_token"]
        return self._token

    def _headers(self) -> Dict[str, str]:
        return {
            "Client-ID": TWITCH_CLIENT_ID,
            "Authorization": f"Bearer {self._get_token()}",
            "Accept": "application/json",
        }

    # ------------------------ Core query ------------------------
    def paged(self, endpoint: str, fields: str, where: Optional[str] = None, sort: str = "id asc",
              limit: int = 500, max_rows: Optional[int] = None) -> Iterator[Dict]:
        offset = 0
        total = 0
        while True:
            body = f"fields {fields};"
            if where:
                body += f" where {where};"
            body += f" sort {sort}; limit {limit}; offset {offset};"

            r = self.session.post(f"{IGDB_BASE}/{endpoint}", headers=self._headers(), data=body.encode("utf-8"), timeout=90)
            if r.status_code == 429:
                time.sleep(max(self.rate_sleep, 1.5))
                continue
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break

            for row in batch:
                yield flatten_record(row)
                total += 1
                if max_rows and total >= max_rows:
                    return

            offset += limit
            time.sleep(self.rate_sleep)


    def fetch_df(
        self,
        endpoint: str,
        fields: str = "*",
        where: Optional[str] = None,
        sort: str = "id asc",
        max_rows: Optional[int] = None,) -> pd.DataFrame:
        """
        Fetch an endpoint into a pandas DataFrame using the existing paged() generator.
        """
        rows_iter = self.paged(endpoint=endpoint, fields=fields, where=where, sort=sort, max_rows=max_rows)
        return pd.DataFrame.from_records(list(rows_iter))

    # bind as instance method
#IGDBClient.fetch_df = fetch_df