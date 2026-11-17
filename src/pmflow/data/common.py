"""Shared helpers for data ingestion: DB connection, HTTP with backoff, checkpoint state."""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
import requests

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "data"
LOG_DIR = DATA_DIR / "logs"
ARTIFACTS = REPO / "artifacts" / "data"
DSN = os.environ.get("PMFLOW_DSN", "postgresql://pmflow:pmflow@localhost:5432/pmflow")
SCHEMA = Path(__file__).with_name("schema.sql")

UA = "pmflow-research/0.1 (public-data academic prototype)"


def setup_logging(name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s")
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        fh = logging.FileHandler(LOG_DIR / f"{name}.log")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(fh)
        root.addHandler(sh)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return logging.getLogger(name)


def connect(autocommit: bool = True) -> psycopg.Connection:
    return psycopg.connect(DSN, autocommit=autocommit)


def apply_schema() -> None:
    with connect() as c:
        c.execute(SCHEMA.read_text())


def ts_from_any(v: Any) -> datetime | None:
    """Parse epoch seconds / ISO strings into aware UTC datetimes."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    s = str(v).strip()
    try:
        if s.replace(".", "", 1).isdigit():
            return datetime.fromtimestamp(float(s), tz=timezone.utc)
        s = s.replace("Z", "+00:00")
        if " " in s and "T" not in s:
            s = s.replace(" ", "T", 1)
        if len(s) == 10:
            s += "T00:00:00+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def fnum(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def jload(v: Any) -> Any:
    """Gamma returns some arrays as JSON-encoded strings."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return None
    return v


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


class Http:
    """Thread-local requests sessions with retry + exponential backoff on 429/5xx/network errors."""

    def __init__(self, max_retries: int = 8, min_interval: float = 0.0):
        self._local = threading.local()
        self.max_retries = max_retries
        self.min_interval = min_interval
        self.log = logging.getLogger("http")

    def _session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip"})
            self._local.s = s
            self._local.last = 0.0
        return s

    def get_json(self, url: str, params: dict | None = None, timeout: float = 60) -> Any:
        s = self._session()
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            if self.min_interval:
                wait = self._local.last + self.min_interval - time.time()
                if wait > 0:
                    time.sleep(wait)
            self._local.last = time.time()
            try:
                r = s.get(url, params=params, timeout=timeout)
            except requests.RequestException as e:  # network error
                if attempt == self.max_retries:
                    raise
                self.log.warning("network error %s on %s; retry in %.1fs", e, url, delay)
                time.sleep(delay + random.random())
                delay = min(delay * 2, 120)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504, 520, 522, 524):
                if attempt == self.max_retries:
                    raise HttpError(r.status_code, r.text)
                ra = r.headers.get("retry-after")
                wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay
                self.log.warning("HTTP %s on %s; retry in %.1fs", r.status_code, url, wait)
                time.sleep(wait + random.random())
                delay = min(delay * 2, 120)
                continue
            raise HttpError(r.status_code, r.text)
        raise RuntimeError("unreachable")


# ---------------------------------------------------------------- checkpoint helpers

def state_get(conn: psycopg.Connection, source: str, key: str) -> dict | None:
    row = conn.execute(
        "SELECT status, cursor, n_rows, extra FROM ingest_state WHERE source=%s AND key=%s",
        (source, key),
    ).fetchone()
    if not row:
        return None
    return {"status": row[0], "cursor": row[1], "n_rows": row[2], "extra": row[3]}


def state_set(conn: psycopg.Connection, source: str, key: str, status: str, cursor: str | None = None,
              n_rows: int | None = None, error: str | None = None, priority: float | None = None,
              extra: dict | None = None) -> None:
    conn.execute(
        """INSERT INTO ingest_state (source, key, status, cursor, n_rows, error, priority, extra, updated_at)
           VALUES (%s,%s,%s,%s,coalesce(%s,0),%s,%s,%s,now())
           ON CONFLICT (source, key) DO UPDATE SET
             status=EXCLUDED.status, cursor=EXCLUDED.cursor,
             n_rows=coalesce(%s, ingest_state.n_rows), error=EXCLUDED.error,
             priority=coalesce(EXCLUDED.priority, ingest_state.priority),
             extra=coalesce(EXCLUDED.extra, ingest_state.extra), updated_at=now()""",
        (source, key, status, cursor, n_rows, error, priority,
         json.dumps(extra) if extra is not None else None, n_rows),
    )
