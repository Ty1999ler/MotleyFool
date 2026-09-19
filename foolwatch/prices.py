"""Price history via Yahoo Finance's public chart API (no API key required)."""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone

import requests

from .config import Config

log = logging.getLogger(__name__)

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) foolwatch/1.0"}
BENCHMARK = "SPY"


def fetch_daily(symbol: str, range_: str = "1y",
                session: requests.Session | None = None) -> list[tuple[str, float, int]]:
    """Fetch daily closes as [(YYYY-MM-DD, close, volume), ...].

    Raises requests.RequestException on network trouble, ValueError when the
    symbol simply has no data.
    """
    sess = session or requests
    resp = sess.get(
        CHART_URL.format(symbol=symbol),
        params={"range": range_, "interval": "1d", "events": "div,splits"},
        headers=HEADERS,
        timeout=30,
    )
    if resp.status_code in (404, 422):
        raise ValueError(f"No Yahoo data for {symbol}")
    resp.raise_for_status()
    result = (resp.json().get("chart") or {}).get("result")
    if not result:
        raise ValueError(f"No Yahoo data for {symbol}")
    r = result[0]
    timestamps = r.get("timestamp") or []
    quote = ((r.get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []
    tz_offset = (r.get("meta") or {}).get("gmtoffset", 0)
    out = []
    for i, ts in enumerate(timestamps):
        close = closes[i] if i < len(closes) else None
        if close is None:
            continue
        vol = volumes[i] if i < len(volumes) and volumes[i] is not None else 0
        day = datetime.fromtimestamp(ts + tz_offset, tz=timezone.utc).strftime("%Y-%m-%d")
        out.append((day, round(float(close), 4), int(vol)))
    if not out:
        raise ValueError(f"No usable price rows for {symbol}")
    return out


def resolve_yahoo_symbol(conn: sqlite3.Connection, ticker: str,
                         session: requests.Session) -> str | None:
    """Find and cache the Yahoo symbol for a ticker (BRK.B -> BRK-B)."""
    row = conn.execute(
        "SELECT yahoo_symbol, yahoo_failed FROM tickers WHERE ticker = ?", (ticker,)
    ).fetchone()
    if row and row["yahoo_symbol"]:
        return row["yahoo_symbol"]
    if row and row["yahoo_failed"]:
        return None

    base = ticker.replace(".", "-")
    for cand in (base, f"{base}.TO"):
        try:
            fetch_daily(cand, range_="5d", session=session)
            conn.execute("UPDATE tickers SET yahoo_symbol = ? WHERE ticker = ?", (cand, ticker))
            conn.commit()
            return cand
        except ValueError:
            time.sleep(0.2)
        except requests.RequestException as e:
            # Transient: don't blacklist, and don't risk caching a wrong candidate.
            log.warning("Network error probing %s for %s; retry next run: %s", cand, ticker, e)
            return None
    conn.execute("UPDATE tickers SET yahoo_failed = 1 WHERE ticker = ?", (ticker,))
    conn.commit()
    return None


def update_prices(conn: sqlite3.Connection, cfg: Config, range_: str = "1y",
                  min_articles: int = 1, only_missing: bool = False) -> dict:
    """Refresh closes for the benchmark plus every ticker worth charting.

    `min_articles` keeps a 3-year backfill from fanning out to thousands of
    one-off incidental tickers; `only_missing` skips symbols already stored.
    """
    session = requests.Session()
    rows = conn.execute(
        """
        SELECT c.ticker, COUNT(*) n
        FROM coverage c
        GROUP BY c.ticker
        HAVING COUNT(*) >= ?
        ORDER BY n DESC
        """,
        (min_articles,),
    ).fetchall()
    tickers = [r["ticker"] for r in rows]

    if only_missing:
        have = {r["symbol"] for r in conn.execute("SELECT DISTINCT symbol FROM prices")}
        resolved = {
            r["ticker"]: r["yahoo_symbol"]
            for r in conn.execute("SELECT ticker, yahoo_symbol FROM tickers")
        }
        tickers = [t for t in tickers if resolved.get(t) not in have]

    updated, failed = 0, 0
    log.info("Prices: %d tickers to refresh (range=%s)", len(tickers), range_)

    symbols: list[tuple[str, str]] = [(BENCHMARK, BENCHMARK)]
    for t in tickers:
        sym = resolve_yahoo_symbol(conn, t, session)
        if sym:
            symbols.append((t, sym))
        else:
            failed += 1

    for i, (ticker, sym) in enumerate(symbols, 1):
        try:
            rows_ = fetch_daily(sym, range_=range_, session=session)
            conn.executemany(
                "INSERT OR REPLACE INTO prices (symbol, date, close, volume) VALUES (?, ?, ?, ?)",
                [(sym, d, c, v) for d, c, v in rows_],
            )
            conn.commit()
            updated += 1
        except (ValueError, requests.RequestException) as e:
            failed += 1
            log.warning("Price fetch failed for %s (%s): %s", ticker, sym, e)
        if i % 100 == 0:
            log.info("  prices %d/%d", i, len(symbols))
        time.sleep(0.25)

    log.info("Prices updated: %d ok, %d failed", updated, failed)
    return {"updated": updated, "failed": failed}


def yahoo_symbol_for(conn: sqlite3.Connection, ticker: str) -> str | None:
    row = conn.execute("SELECT yahoo_symbol FROM tickers WHERE ticker = ?", (ticker,)).fetchone()
    return row["yahoo_symbol"] if row and row["yahoo_symbol"] else None


def close_on_or_after(conn: sqlite3.Connection, symbol: str, day: str) -> tuple[str, float] | None:
    row = conn.execute(
        "SELECT date, close FROM prices WHERE symbol = ? AND date >= ? ORDER BY date LIMIT 1",
        (symbol, day),
    ).fetchone()
    return (row["date"], row["close"]) if row else None


def latest_close(conn: sqlite3.Connection, symbol: str) -> tuple[str, float] | None:
    row = conn.execute(
        "SELECT date, close FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    return (row["date"], row["close"]) if row else None


def pct_return(conn: sqlite3.Connection, symbol: str, from_day: str) -> float | None:
    """Return from the first close on/after from_day to the latest close.

    Returns None when the earliest stored close is more than a week after
    from_day — a late baseline would silently misstate the return.
    """
    from datetime import date

    start = close_on_or_after(conn, symbol, from_day)
    end = latest_close(conn, symbol)
    if not start or not end or not start[1] or end[0] <= start[0]:
        return None
    if (date.fromisoformat(start[0]) - date.fromisoformat(from_day)).days > 7:
        return None
    return (end[1] - start[1]) / start[1] * 100.0
