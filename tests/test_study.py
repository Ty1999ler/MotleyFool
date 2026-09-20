"""Statistics behind the analysis tabs.

The Wilson interval and the verdict wording are what stop a three-for-four
record reading as a 75% skill estimate.
"""

from __future__ import annotations

import pandas as pd
import pytest

from foolwatch.db import get_conn
from foolwatch.study import MIN_PAIRS, bucket_of, verdict, wilson


def test_wilson_is_wide_for_tiny_samples():
    lo, hi = wilson(3, 4)
    assert lo < 40 and hi > 90          # 75% "record" spans most of the range


def test_wilson_tightens_as_the_sample_grows():
    narrow = wilson(600, 1000)
    wide = wilson(6, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_handles_the_degenerate_cases():
    assert wilson(0, 0) != wilson(0, 1)  # n=0 returns NaNs, not a crash
    lo, hi = wilson(1, 1)
    assert lo < 100 and hi <= 100        # a single hit is never 100% certain


def test_wilson_stays_inside_zero_to_one_hundred():
    for hits, n in [(0, 5), (5, 5), (1, 3), (99, 100)]:
        lo, hi = wilson(hits, n)
        assert 0 <= lo <= 100 and 0 <= hi <= 100


@pytest.mark.parametrize("n,expected", [
    (1, "1st article"), (2, "2nd article"), (3, "3rd article"),
    (4, "4th-5th"), (5, "4th-5th"), (7, "6th-10th"), (99, "11th+"),
])
def test_ordinal_buckets(n, expected):
    assert bucket_of(n) == expected


def _paired(tickers, mean_diff, median_diff, wins, horizon="21d"):
    return pd.DataFrame([{
        "horizon": horizon, "tickers": tickers,
        "first_mean": 0.0, "second_mean": 0.0,
        "first_median": 0.0, "second_median": 0.0,
        "mean_diff": mean_diff, "median_diff": median_diff,
        "first_wins_pct": wins,
    }])


def test_verdict_refuses_a_small_sample():
    out = verdict(_paired(MIN_PAIRS - 1, 5.0, 4.0, 70), pd.DataFrame())
    assert "too few" in out.lower()


def test_verdict_calls_out_disagreement_between_mean_median_and_winrate():
    """A positive mean on a flat median and a sub-50% win rate is outliers,
    not an edge — the exact shape the real data first produced."""
    out = verdict(_paired(100, 1.18, 0.0, 36), pd.DataFrame())
    assert "no evidence" in out.lower()


def test_verdict_dismisses_a_coin_flip():
    out = verdict(_paired(100, 0.2, 0.1, 50), pd.DataFrame())
    assert "coin flip" in out.lower()


def test_verdict_reports_a_consistent_signal_but_still_hedges():
    out = verdict(_paired(200, 4.0, 3.0, 62), pd.DataFrame())
    assert "association" in out.lower()


def test_verdict_handles_no_pairs_at_all():
    assert "nothing to compare" in verdict(pd.DataFrame(), pd.DataFrame()).lower()


def test_schema_creates_cleanly_on_a_fresh_database(tmp_path):
    conn = get_conn(tmp_path / "test.db")
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"articles", "coverage", "crawl_queue", "prices", "tickers"} <= tables
    conn.close()


def test_coverage_is_unique_per_article_and_ticker(tmp_path):
    conn = get_conn(tmp_path / "test.db")
    conn.execute("INSERT INTO articles (path, title) VALUES ('/a/', 't')")
    for _ in range(2):
        conn.execute(
            "INSERT OR REPLACE INTO coverage (ticker, path, published_day, "
            "is_primary, source, verified) VALUES ('NVDA','/a/','2026-01-01',1,'meta',1)")
    assert conn.execute("SELECT COUNT(*) c FROM coverage").fetchone()["c"] == 1
    conn.close()
