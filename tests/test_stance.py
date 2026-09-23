"""Headline classification.

Every case here is one the classifier got wrong at some point. The ordering
of the rules is the fragile part: several headline shapes contain buy/sell
wording while making no call at all, and they have to be caught first.
"""

from __future__ import annotations

import pytest

from foolwatch.stance import classify


@pytest.mark.parametrize("headline", [
    "Abercrombie & Fitch Chief HR Officer Sells 5,000 Shares",
    "Five Below CHRO Maureen Gellerman Sells 1,610 Shares for $401,000",
    "Ouster's CEO Sells 30,385 Shares Worth $1 Million",
    "BorgWarner Director Michael Hanley Sells 5,000 Shares for $327,550",
])
def test_insider_filings_are_not_sell_ratings(headline):
    """A Form 4 report contains "Sells" but expresses no opinion."""
    label, score = classify(headline)
    assert label == "insider"
    assert score == 0


@pytest.mark.parametrize("headline", [
    "Nvidia vs. AMD: Which Is the Better AI Chip Stock?",
    "Better Buy: Palantir Stock vs. Microsoft Stock",
    "Rivian vs. Lucid: This Stock Is Clearly the Better Buy",
    "SpaceX or Tesla: Which Elon Musk Stock Should Investors Buy?",
    "Better eVTOL Stock: Archer Aviation vs. Joby Aviation",
])
def test_head_to_head_comparisons_make_no_call(headline):
    """A comparison is tagged to *both* tickers, so a bullish score would
    credit a buy call to the loser as well."""
    label, score = classify(headline)
    assert label == "comparison"
    assert score == 0


@pytest.mark.parametrize("headline", [
    "Why Corning Plunged Today",
    "Why ServiceTitan Stock Is Crashing Today",
    "Why Dave & Buster's Stock Tumbled Today",
    "Why Super Micro Computer Stock Tumbled Monday Morning",
    "Why Rocket Lab Stock Soared Today",
])
def test_move_explainers_are_news_in_both_directions(headline):
    """Explaining yesterday's move is not a call. Downward moves used to hit
    the warning vocabulary while upward ones fell to news — an asymmetry that
    mislabelled about a quarter of all warnings."""
    assert classify(headline) == ("news", 0)


def test_a_forward_looking_why_is_still_a_warning():
    """The explainer rule must not swallow genuine warnings."""
    assert classify("Why a Stock Market Crash Could Be Coming")[0] == "warning"


def test_buying_opportunity_leans_buy_even_beside_crash_wording():
    label, score = classify(
        "Why Is CAVA Stock Crashing, and Is It a Buying Opportunity?")
    assert label == "buy_lean"
    assert score > 0


def test_enumerated_list_with_both_directions_is_mixed():
    label, score = classify(
        "Semiconductor Sell-Off: 1 AI Chip Stock to Buy, 1 to Hold, and 1 to Sell")
    assert label == "mixed"
    assert score == 0


@pytest.mark.parametrize("headline,expected", [
    ("2 High-Yield Dividend Stocks to Buy Now", "buy"),
    ("3 Reasons to Buy This Stock Right Now", "buy"),
    ("Should You Buy Oracle Stock While It's Down 53%?", "buy_lean"),
    ("Revolution Medicines Just Hit a Major Milestone. Is the Stock a Buy?", "buy_lean"),
    ("Should You Avoid Wendy's Stock, Even After a 66% Decline?", "sell_lean"),
    ("SpaceX Investors Just Got Some Bad News", "warning"),
    ("Prediction: Tesla Stock Will Be Worth This Much in 2031", "prediction"),
])
def test_directional_labels(headline, expected):
    assert classify(headline)[0] == expected


def test_scores_are_signed_and_bounded():
    for headline in [
        "2 High-Yield Dividend Stocks to Buy Now",
        "Should You Avoid Wendy's Stock?",
        "Prediction: Tesla Stock Will Be Worth This Much in 2031",
    ]:
        _, score = classify(headline)
        assert -2 <= score <= 2


def test_bullish_and_bearish_scores_have_opposite_signs():
    assert classify("2 Dividend Stocks to Buy Now")[1] > 0
    assert classify("Should You Avoid Wendy's Stock?")[1] < 0


def test_empty_and_untyped_headlines_do_not_raise():
    assert classify("") == ("analysis", 0)
    assert classify("   ")[1] == 0


def test_unremarkable_headline_falls_through_to_analysis():
    label, score = classify(
        "What Would It Take for Investors to Pay More for Toast Stock?")
    assert label == "analysis"
    assert score == 0
