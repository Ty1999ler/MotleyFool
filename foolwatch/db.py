"""SQLite schema and connection helpers.

One row per article in `articles`, one row per (article, ticker) pair in
`coverage`. Keeping coverage separate is what makes "when was X first written
about, and how did the take change" a plain query rather than string parsing.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    path TEXT PRIMARY KEY,              -- /investing/2026/08/31/slug/ (url minus host)
    title TEXT,
    author TEXT,
    published_day TEXT,                 -- YYYY-MM-DD, from the URL date segment
    published_utc INTEGER,
    section TEXT,                       -- investing | retirement | ...
    stance TEXT,                        -- buy | sell | hold | warning | prediction | news | analysis
    stance_score INTEGER,               -- -2..+2, signed strength of the stance
    ticker_count INTEGER,
    fetched_utc INTEGER
);
CREATE INDEX IF NOT EXISTS idx_articles_day ON articles(published_day);
CREATE INDEX IF NOT EXISTS idx_articles_author ON articles(author);

CREATE TABLE IF NOT EXISTS coverage (
    ticker TEXT NOT NULL,
    path TEXT NOT NULL REFERENCES articles(path) ON DELETE CASCADE,
    published_day TEXT,
    is_primary INTEGER,                 -- 1 = headline subject or explicit (NASDAQ: X) ref
    source TEXT,                        -- meta | exchange_ref | headline | sitemap
    verified INTEGER,                   -- 1 if the ticker is in the exchange universe
    PRIMARY KEY (ticker, path)
);
CREATE INDEX IF NOT EXISTS idx_coverage_ticker_day ON coverage(ticker, published_day);
CREATE INDEX IF NOT EXISTS idx_coverage_day ON coverage(published_day);
CREATE INDEX IF NOT EXISTS idx_coverage_primary ON coverage(is_primary, ticker);

-- Crawl queue. Every URL the sitemaps offer lands here first, so a backfill
-- can be stopped and resumed without refetching, and failures are visible.
CREATE TABLE IF NOT EXISTS crawl_queue (
    path TEXT PRIMARY KEY,
    section TEXT,
    published_day TEXT,
    discovered_utc INTEGER,
    state TEXT NOT NULL DEFAULT 'pending',  -- pending | done | skipped | failed
    attempts INTEGER DEFAULT 0,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_queue_state ON crawl_queue(state, published_day);

-- Which archive months have been fully enumerated (not fetched — enumerated).
CREATE TABLE IF NOT EXISTS archive_months (
    month TEXT PRIMARY KEY,             -- YYYY/MM
    urls_found INTEGER,
    enumerated_utc INTEGER
);

CREATE TABLE IF NOT EXISTS universe (
    ticker TEXT PRIMARY KEY,
    name TEXT,
    exchange TEXT,
    is_etf INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tickers (
    ticker TEXT PRIMARY KEY,
    name TEXT,
    yahoo_symbol TEXT,
    yahoo_failed INTEGER DEFAULT 0,
    first_seen_day TEXT
);

CREATE TABLE IF NOT EXISTS prices (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    close REAL,
    volume INTEGER,
    PRIMARY KEY(symbol, date)
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT,                          -- daily | backfill | prices
    started_utc INTEGER,
    finished_utc INTEGER,
    articles_new INTEGER,
    coverage_new INTEGER,
    notes TEXT
);
"""


def get_conn(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def utc_now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def today_local(tz_name: str) -> str:
    return datetime.now(ZoneInfo(tz_name)).strftime("%Y-%m-%d")
