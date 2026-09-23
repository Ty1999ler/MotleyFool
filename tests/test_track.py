"""Track record: call outcomes, their signs, and the price-history backfill.

Built on a synthetic market where every answer is known in advance: SPY is
flat, one stock compounds up 1% a session and another down 1%, so any return
the code produces can be checked against arithmetic.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

from foolwatch import prices, study
from foolwatch.config import Config
from foolwatch.db import get_conn

SESSIONS = pd.bdate_range("2024-01-02", periods=300)
UP_21 = (1.01 ** 21 - 1) * 100        # +23.24%: the rising stock over 1 month
DOWN_21 = (0.99 ** 21 - 1) * 100      # -19.03%: the falling stock over 1 month


def _add_prices(conn, symbol, factor):
    conn.executemany(
        "INSERT INTO prices (symbol, date, close, volume) VALUES (?, ?, ?, 0)",
        [(symbol, d.strftime("%Y-%m-%d"), 100 * factor ** i)
         for i, d in enumerate(SESSIONS)])


def _add_article(conn, path, title, day, score, ticker, author="Jane Fool"):
    stance = {2: "buy", 1: "buy_lean", 0: "analysis", -1: "warning", -2: "sell"}[score]
    conn.execute(
        "INSERT INTO articles (path, title, author, published_day, stance, "
        "stance_score) VALUES (?, ?, ?, ?, ?, ?)",
        (path, title, author, day, stance, score))
    conn.execute(
        "INSERT INTO coverage (ticker, path, published_day, is_primary, source, "
        "verified) VALUES (?, ?, ?, 1, 'primary_meta', 1)", (ticker, path, day))


@pytest.fixture
def market(tmp_path) -> sqlite3.Connection:
    conn = get_conn(tmp_path / "t.db")
    for t in ("UP", "DOWN"):
        conn.execute("INSERT INTO tickers (ticker, yahoo_symbol) VALUES (?, ?)", (t, t))
    _add_prices(conn, "SPY", 1.0)      # flat, so excess return == raw return
    _add_prices(conn, "UP", 1.01)
    _add_prices(conn, "DOWN", 0.99)
    day0 = SESSIONS[0].strftime("%Y-%m-%d")
    _add_article(conn, "/a/buy-up/", "3 Stocks to Buy Now", day0, 2, "UP")
    _add_article(conn, "/a/warn-down/", "Bad News for Holders", day0, -1, "DOWN")
    _add_article(conn, "/a/buy-down/", "2 Stocks to Buy Now", day0, 2, "DOWN")
    _add_article(conn, "/a/news-up/", "Why Up Stock Jumped", day0, 0, "UP")
    conn.commit()
    return conn


def test_outcomes_match_the_arithmetic(market):
    study.compute_call_outcomes(market)
    row = market.execute(
        "SELECT ret_21, spy_21, ret_252 FROM call_outcomes WHERE path = '/a/buy-up/'"
    ).fetchone()
    assert row["ret_21"] == pytest.approx(UP_21, rel=1e-6)
    assert row["spy_21"] == pytest.approx(0.0, abs=1e-9)
    assert row["ret_252"] is not None      # 300 sessions of history covers 12m


def test_a_bearish_call_pays_when_the_stock_falls(market):
    """The whole point of signing returns: 'caution' on a falling stock was right."""
    study.compute_call_outcomes(market)
    df = study.track_record_frame(market).set_index("path")
    assert df.loc["/a/warn-down/", "call_1m"] == pytest.approx(-DOWN_21, rel=1e-6)
    assert df.loc["/a/warn-down/", "hit_1m"] == 1.0


def test_a_bullish_call_on_a_falling_stock_is_a_miss(market):
    study.compute_call_outcomes(market)
    df = study.track_record_frame(market).set_index("path")
    assert df.loc["/a/buy-down/", "call_1m"] == pytest.approx(DOWN_21, rel=1e-6)
    assert df.loc["/a/buy-down/", "hit_1m"] == 0.0
    assert df.loc["/a/buy-up/", "hit_1m"] == 1.0


def test_articles_without_a_direction_are_not_calls(market):
    study.compute_call_outcomes(market)
    df = study.track_record_frame(market).set_index("path")
    assert np.isnan(df.loc["/a/news-up/", "call_1m"])
    assert np.isnan(df.loc["/a/news-up/", "hit_1m"])
    # ...but its raw outcome is still recorded
    assert df.loc["/a/news-up/", "excess_1m"] == pytest.approx(UP_21, rel=1e-6)


def test_scoring_is_incremental(market):
    """A call whose 12-month window has closed can never change again."""
    first = study.compute_call_outcomes(market)
    again = study.compute_call_outcomes(market)
    assert first["computed"] == 4
    assert again["computed"] == 0
    assert study.compute_call_outcomes(market, full=True)["computed"] == 4


def test_stale_stance_is_not_baked_into_outcomes(market):
    """Stance is re-scored when the classifier improves; outcomes must follow."""
    study.compute_call_outcomes(market)
    market.execute("UPDATE articles SET stance='sell', stance_score=-2 "
                   "WHERE path='/a/buy-up/'")
    market.commit()
    df = study.track_record_frame(market).set_index("path")
    assert df.loc["/a/buy-up/", "call_1m"] == pytest.approx(-UP_21, rel=1e-6)


def test_an_article_before_the_price_series_gets_no_fake_entry(market):
    _add_article(market, "/a/early/", "Stocks to Buy Now", "2023-06-01", 2, "UP")
    market.commit()
    study.compute_call_outcomes(market)
    row = market.execute(
        "SELECT entry_date, ret_21 FROM call_outcomes WHERE path='/a/early/'").fetchone()
    assert row["entry_date"] is None and row["ret_21"] is None


def test_scorecard_counts_and_rates(market):
    study.compute_call_outcomes(market)
    sc = study.scorecard(study.track_record_frame(market), "stance").set_index("stance")
    assert sc.loc["buy", "calls"] == 2
    assert sc.loc["buy", "hit_1m"] == pytest.approx(50.0)     # one up, one down
    assert sc.loc["warning", "hit_1m"] == pytest.approx(100.0)
    assert "analysis" not in sc.index                          # not a call


def test_follow_the_calls_counts_a_stock_once_per_month(market):
    day = SESSIONS[3].strftime("%Y-%m-%d")
    _add_article(market, "/a/buy-up-2/", "More Stocks to Buy Now", day, 2, "UP")
    market.commit()
    study.compute_call_outcomes(market)
    f = study.follow_the_calls(study.track_record_frame(market), "1m")
    assert len(f) == 1
    assert f.loc[0, "calls"] == 2          # UP and DOWN, UP not double-counted
    expected = (UP_21 + DOWN_21) / 2
    assert f.loc[0, "basket"] == pytest.approx(expected, rel=1e-6)
    assert f.loc[0, "follow_growth"] == pytest.approx(1 + expected / 100, rel=1e-6)
    assert f.loc[0, "spy_growth"] == pytest.approx(1.0, abs=1e-9)


def test_price_book_loads_from_the_database(market):
    book = study._PriceBook.from_db(market, ["UP", "SPY", "MISSING"])
    assert set(book.symbols) == {"UP", "SPY"}
    assert book.size("UP") == len(SESSIONS)


def test_history_backfill_fetches_from_a_year_before_first_coverage(market, monkeypatch):
    market.execute("DELETE FROM prices WHERE symbol != 'SPY'")
    market.commit()
    asked: dict[str, date] = {}

    def fake_fetch(sym, range_="1y", session=None, start=None):
        asked[sym] = start
        return [(start.isoformat(), 100.0, 0)]

    monkeypatch.setattr(prices, "fetch_daily", fake_fetch)
    monkeypatch.setattr(prices.time, "sleep", lambda s: None)
    res = prices.ensure_history(market, Config(), lead_days=400)
    need = date(2024, 1, 2) - pd.Timedelta(days=400).to_pytimedelta()
    assert asked["UP"] == need and asked["DOWN"] == need
    assert asked["SPY"] == need            # the benchmark must reach back too
    assert res["fetched"] == 3


def test_history_backfill_asks_each_ticker_only_once(market, monkeypatch):
    """A stock that listed late can never have history reaching back far
    enough; without the marker it would be re-downloaded every day."""
    market.execute("DELETE FROM prices WHERE symbol != 'SPY'")
    market.commit()
    calls = []

    def fake_fetch(sym, range_="1y", session=None, start=None):
        calls.append(sym)
        return [("2024-06-01", 100.0, 0)]   # history starts long after "need"

    monkeypatch.setattr(prices, "fetch_daily", fake_fetch)
    monkeypatch.setattr(prices.time, "sleep", lambda s: None)
    prices.ensure_history(market, Config())
    first = len(calls)
    prices.ensure_history(market, Config())
    assert first > 0
    assert len(calls) == first + 1          # only SPY re-checked, no tickers


def test_migration_adds_history_marker_to_an_old_database(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE tickers (ticker TEXT PRIMARY KEY, name TEXT, "
                "yahoo_symbol TEXT, yahoo_failed INTEGER DEFAULT 0, "
                "first_seen_day TEXT)")
    old.commit()
    old.close()
    conn = get_conn(path)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(tickers)")}
    assert "history_from" in cols


def test_last_daily_run_survives_a_restart(market):
    from zoneinfo import ZoneInfo
    from foolwatch.scheduler import _last_daily_day

    tz = ZoneInfo("America/New_York")
    assert _last_daily_day(market, tz) is None
    ts = int(pd.Timestamp("2026-09-23 18:45", tz=tz).timestamp())
    market.execute("INSERT INTO runs (kind, started_utc, finished_utc) "
                   "VALUES ('daily', ?, ?)", (ts - 60, ts))
    market.commit()
    assert _last_daily_day(market, tz) == "2026-09-23"
