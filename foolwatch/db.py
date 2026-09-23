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
-- The primary key is (ticker, path), which cannot serve a lookup by path
-- alone. Without this, any join from articles to coverage scanned every
-- coverage row per article: ~10 billion row visits at 100k articles, which
-- hung the Authors tab outright.
CREATE INDEX IF NOT EXISTS idx_coverage_path ON coverage(path);

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
    first_seen_day TEXT,
    history_from TEXT                   -- earliest date Yahoo has been asked for
);

CREATE TABLE IF NOT EXISTS prices (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    close REAL,
    volume INTEGER,
    PRIMARY KEY(symbol, date)
);

-- One row per primary (article, ticker) pair: what the stock did after the
-- article, and what SPY did over the same sessions. Computed by the worker so
-- the dashboard reads a small table instead of millions of price rows.
-- Returns are percent; NULL where the price series does not reach that far.
-- Deliberately holds nothing from the article itself: stance is re-scored
-- whenever the classifier improves, so it is joined fresh from `articles`.
CREATE TABLE IF NOT EXISTS call_outcomes (
    path TEXT NOT NULL,
    ticker TEXT NOT NULL,
    published_day TEXT,
    entry_date TEXT,                    -- first session on/after publication
    entry_close REAL,
    ret_21 REAL,  spy_21 REAL,          -- ~1 month
    ret_63 REAL,  spy_63 REAL,          -- ~3 months
    ret_126 REAL, spy_126 REAL,         -- ~6 months
    ret_252 REAL, spy_252 REAL,         -- ~12 months
    computed_utc INTEGER,
    PRIMARY KEY (path, ticker)
);
CREATE INDEX IF NOT EXISTS idx_outcomes_day ON call_outcomes(published_day);

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
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created.

    CREATE TABLE IF NOT EXISTS never alters an existing table, so a column
    added to SCHEMA later has to be added to live databases here.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(tickers)")}
    if "history_from" not in cols:
        conn.execute("ALTER TABLE tickers ADD COLUMN history_from TEXT")
        conn.commit()


def utc_now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def today_local(tz_name: str) -> str:
    return datetime.now(ZoneInfo(tz_name)).strftime("%Y-%m-%d")
