"""Exchange ticker universe, used to verify that a ticker is a real listing."""

from __future__ import annotations

import logging
import re
import sqlite3

import requests

log = logging.getLogger(__name__)

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def download_universe(conn: sqlite3.Connection) -> int:
    """Load NASDAQ + NYSE/AMEX/Arca listings into the universe table.

    Network failures are tolerated when a cached universe already exists; the
    first run needs it and raises.
    """
    rows: list[tuple[str, str, str, int]] = []
    try:
        for url, is_nasdaq in ((NASDAQ_LISTED_URL, True), (OTHER_LISTED_URL, False)):
            resp = requests.get(url, timeout=60, headers={"User-Agent": "foolwatch/1.0"})
            resp.raise_for_status()
            lines = resp.text.splitlines()
            col = {name: i for i, name in enumerate(lines[0].split("|"))}
            sym_col = col.get("Symbol", col.get("ACT Symbol", 0))
            name_col = col.get("Security Name", 1)
            etf_col = col.get("ETF")
            test_col = col.get("Test Issue")
            for line in lines[1:]:
                if line.startswith("File Creation Time"):
                    continue
                parts = line.split("|")
                if len(parts) <= max(sym_col, name_col):
                    continue
                sym = parts[sym_col].strip().upper()
                if not re.fullmatch(r"[A-Z]{1,5}(\.[A-Z]{1,2})?", sym or ""):
                    continue
                if test_col is not None and len(parts) > test_col and parts[test_col].strip() == "Y":
                    continue
                is_etf = 1 if (etf_col is not None and len(parts) > etf_col
                               and parts[etf_col].strip() == "Y") else 0
                exch = "NASDAQ" if is_nasdaq else parts[col.get("Exchange", 0)].strip()
                rows.append((sym, parts[name_col].strip(), exch, is_etf))
    except requests.RequestException as e:
        existing = conn.execute("SELECT COUNT(*) c FROM universe").fetchone()["c"]
        if existing:
            log.warning("Universe download failed (%s); keeping cached %d symbols", e, existing)
            return existing
        raise

    conn.executemany(
        "INSERT OR REPLACE INTO universe (ticker, name, exchange, is_etf) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    log.info("Universe loaded: %d symbols", len(rows))
    return len(rows)


def backfill_names(conn: sqlite3.Connection) -> int:
    """Copy company names from the universe onto tracked tickers."""
    cur = conn.execute(
        "UPDATE tickers SET name = (SELECT name FROM universe u WHERE u.ticker = tickers.ticker) "
        "WHERE name IS NULL"
    )
    conn.commit()
    return cur.rowcount
