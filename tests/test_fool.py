"""Article-head parsing, url handling and the rate limiter.

No network: parse_head takes the head text directly, which is the whole point
of splitting fetch from parse.
"""

from __future__ import annotations

import time

from foolwatch.fool import (
    RateLimited,
    RateLimiter,
    _day_from_path,
    _norm_ticker,
    parse_head,
    path_of,
)

UNIVERSE = {"NVDA", "AAPL", "BRK.A", "BRK.B", "GOOGL", "TSM", "AXP"}


def in_universe(t: str) -> bool:
    return t in UNIVERSE


HEAD = """
<html><head>
<title>Some Headline | The Motley Fool</title>
<meta name="author" content="Geoffrey Seiler"/>
<meta name="author" content="Duplicate Ignored"/>
<meta name="description" content="A summary."/>
<meta name="tickers" content="NVDA,AAPL,GOOGL"/>
<meta name="primary_tickers" content="NVDA"/>
<meta property="og:title" content="5 AI Stocks to Buy | The Motley Fool"/>
</head>
"""


def test_parse_head_extracts_metadata():
    art = parse_head("/investing/2026/09/16/five-ai-stocks/", HEAD, in_universe)
    assert art.title == "5 AI Stocks to Buy"      # site suffix stripped
    assert art.author == "Geoffrey Seiler"        # first author tag wins
    assert art.section == "investing"
    assert art.published_day == "2026-09-16"


def test_primary_tickers_are_distinguished_from_incidental_ones():
    """The whole first-coverage story depends on this: a round-up tags a dozen
    tickers but is about one."""
    art = parse_head("/investing/2026/09/16/five-ai-stocks/", HEAD, in_universe)
    assert art.primary == {"NVDA"}
    assert set(art.tickers) == {"NVDA", "AAPL", "GOOGL"}
    assert "AAPL" not in art.primary


def test_primary_ticker_absent_from_meta_tickers_is_still_recorded():
    head = HEAD.replace('content="NVDA,AAPL,GOOGL"', 'content="AAPL"')
    art = parse_head("/investing/2026/09/16/x/", head, in_universe)
    assert "NVDA" in art.tickers
    assert art.primary == {"NVDA"}


def test_class_shares_get_their_dot_restored():
    """fool.com renders BRK.A as BRKA; the dot comes back when that resolves."""
    assert _norm_ticker("BRKA", in_universe) == "BRK.A"
    assert _norm_ticker("BRKB", in_universe) == "BRK.B"


def test_normalisation_leaves_real_tickers_alone():
    assert _norm_ticker("NVDA", in_universe) == "NVDA"
    assert _norm_ticker(" tsm ", in_universe) == "TSM"


def test_normalisation_rejects_non_tickers():
    assert _norm_ticker("", in_universe) is None
    assert _norm_ticker("TOOLONGTICKER", in_universe) is None
    assert _norm_ticker("12345", in_universe) is None


def test_dated_paths_yield_section_and_day():
    assert _day_from_path("/investing/2026/08/31/some-slug/") == (
        "investing", "2026-08-31")
    assert _day_from_path("/retirement/2025/01/02/other/") == (
        "retirement", "2025-01-02")


def test_undated_paths_are_rejected():
    assert _day_from_path("/investing/stock-market/") is None
    assert _day_from_path("/quote/nasdaq/nvda/") is None


def test_path_of_strips_host_and_query():
    assert path_of("https://www.fool.com/investing/2026/09/16/x/?a=1#frag") == (
        "/investing/2026/09/16/x/")


def test_rate_limiter_halves_on_penalty_and_has_a_floor():
    limiter = RateLimiter(4.0)
    assert limiter.penalize() == 2.0
    assert limiter.penalize() == 1.0
    for _ in range(20):
        limiter.penalize()
    assert limiter.rate >= RateLimiter.MIN_RATE


def test_rate_limiter_only_recovers_after_a_long_clean_run():
    """A quick recovery would walk straight back into the limiter."""
    limiter = RateLimiter(4.0)
    limiter.penalize()
    for _ in range(199):
        limiter.reward()
    assert limiter.rate == 2.0          # still halved
    limiter.reward()                    # 200th
    assert limiter.rate > 2.0


def test_rate_limiter_spaces_requests():
    limiter = RateLimiter(20.0)         # 50ms apart
    start = time.monotonic()
    for _ in range(3):
        limiter.wait()
    assert time.monotonic() - start >= 0.08


def test_rate_limited_carries_retry_after():
    err = RateLimited(79675.0)
    assert err.retry_after == 79675.0
    assert "429" in str(err)
