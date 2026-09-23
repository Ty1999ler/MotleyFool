"""Headline -> editorial stance.

An article is an opinion, so counting articles undersells what the Fool is
saying. This turns the headline into a coarse stance label plus a signed score
(-2..+2) so the dashboard can chart how the take on a ticker moved over time.

Rules are ordered and the first match wins. Order matters a lot: "Chief HR
Officer Sells 5,000 Shares" is an insider filing, not a sell rating, so the
insider rule has to run before any buy/sell wording is considered.

Labels
    insider     - Form 4 style transaction report; carries no recommendation
    buy         - explicit "stocks to buy" / "reasons to buy"
    sell        - explicit "time to sell" / "stocks to avoid"
    buy_lean    - buy framed as a question ("Is X a Buy?")
    sell_lean   - sell framed as a question ("Should You Avoid X?")
    mixed       - both buy and sell language ("1 to Buy, 1 to Sell")
    hold        - explicit hold / wait
    warning     - bad news, risk, bubble, red flags
    prediction  - forward-looking projection
    news        - move explainers, earnings recaps, announcements
    analysis    - everything else
"""

from __future__ import annotations

import re

# --- signal vocabularies -----------------------------------------------------

# Insider/institutional transaction reports. These name a person or fund doing
# the buying/selling, which is a fact, not the Fool's opinion.
# Head-to-head comparisons: "AMD vs. Texas Instruments: Which Is the Better
# Buy?". These carry buy/sell wording but make no call on either company - and
# the article is tagged to both tickers, so scoring it +1 would credit a
# bullish call to the loser as well. Treated as non-directional.
# "Why Corning Plunged Today" explains a move that already happened; it is news,
# not a bearish call. Without this, a downward move hit the warning vocabulary
# first while an upward one ("Why X Soared Today") correctly fell to news — an
# asymmetry that filed about a quarter of all warnings under the wrong label.
# Stocks that have just plunged often rebound, so it also skewed the track
# record against the Fool's cautious calls.
MOVE_EXPLAINER_RE = re.compile(
    r"^why\s+.{0,90}?\b(?:plunge[sd]?|plunging|sink(?:s|ing)?|sank|sunk|"
    r"tumble[sd]?|tumbling|crash(?:es|ed|ing)?|fell|falls|falling|"
    r"drop(?:s|ped|ping)?|slid(?:e|es|ing)?|slump(?:s|ed|ing)?|"
    r"skyrocket(?:s|ed|ing)?|soar(?:s|ed|ing)?|jump(?:s|ed|ing)?|"
    r"surge[sd]?|surging|rall(?:y|ies|ied|ying)|pop(?:s|ped|ping)?|"
    r"climb(?:s|ed|ing)?|spike[sd]?|spiking|rose|rising|gain(?:s|ed|ing)?)\b",
    re.I,
)
# ...unless it looks forward: "Why a Stock Market Crash Could Be Coming" is a
# genuine warning, not an explainer.
FORWARD_RE = re.compile(
    r"\b(?:could|may|might|will|would|about to|coming|set to)\b", re.I)

COMPARISON_RE = re.compile(
    r"\bvs\.?\s|\bversus\b"
    r"|\bwhich\s+(?:\w+\s+){0,3}?(?:stock|etf|company|one|is)\b.{0,40}\b"
    r"(?:better|best|stronger|smarter)\b"
    r"|\b(?:better|best)\s+(?:buy|bet|pick|choice)\s*[:?]?\s*$"
    r"|\bor\s+\w+[:?]\s*which\b",
    re.I,
)

INSIDER_RE = re.compile(
    r"\b(insider|form\s*4|13[fF]|chief\s+\w+\s+officer|cfo|ceo|coo|cto|chro|"
    r"director|evp|svp|president)\b.{0,60}\b(sell|sells|sold|buy|buys|bought|"
    r"acquires|acquired|disposes|unloads)\b"
    r"|\b(sells|sold|buys|bought|acquires|purchases)\b\s+[\d,]+\s+shares"
    r"|\bshares?\s+(sold|bought|purchased)\s+by\b",
    re.I,
)

BUY_STRONG = (
    r"\b(?:stocks?|etfs?|companies|coins?|picks?|names?|giants?|winners?)\s+"
    r"(?:\w+\s+){0,3}?to\s+buy\b",
    r"\breasons?\s+to\s+buy\b",
    r"\bto\s+buy\s+(?:right\s+)?now\b",
    r"\bbuy\s+(?:the\s+)?dip\b",
    r"\bscreaming\s+buys?\b",
    r"\bno[\s-]brainer\s+(?:buy|stock)",
    r"\bworth\s+buying\b",
    r"\bstart(?:ing)?\s+(?:a\s+)?position\b",
    r"\bloading\s+up\s+on\b",
    r"\bkeep\s+buying\b",
    # Enumerated list form: "1 AI Chip Stock to Buy, 1 to Hold, and 1 to Sell"
    r"\b\d+\s+to\s+buy\b",
)

BUY_QUESTION = (
    r"\bis\s+.{0,60}?\ba\s+buy\b",
    # "Why Is CAVA Stock Crashing, and Is It a Buying Opportunity?" leans buy,
    # but without this it fell through to the crash wording and scored bearish.
    r"\bbuying\s+opportunit(?:y|ies)\b",
    r"\bshould\s+you\s+buy\b",
    r"\bis\s+it\s+(?:too\s+late|still\s+time)\s+to\s+buy\b",
    r"\btime\s+to\s+buy\b",
    r"\bbetter\s+buy\b",
    r"\bworth\s+a\s+look\b",
    r"\bbuy\s+or\s+sell\b",
)

SELL_STRONG = (
    r"\b(?:stocks?|etfs?|companies|names?)\s+(?:\w+\s+){0,3}?to\s+(?:sell|avoid|dump)\b",
    r"\breasons?\s+to\s+sell\b",
    r"\btime\s+to\s+sell\b",
    r"\bsell\s+(?:right\s+)?now\b",
    r"\bget\s+out\s+(?:of|while)\b",
    r"\bdump(?:ing)?\s+this\b",
    r"\bsteer\s+clear\b",
    r"\bstay\s+away\s+from\b",
    # Enumerated list form, the other half of "1 to Buy ... 1 to Sell"
    r"\b\d+\s+to\s+(?:sell|avoid|dump)\b",
)

SELL_QUESTION = (
    r"\bshould\s+you\s+(?:sell|avoid|dump)\b",
    r"\bis\s+it\s+time\s+to\s+(?:sell|take\s+profits|worry)\b",
    r"\bshould\s+(?:investors\s+)?worry\b",
    r"\btake\s+profits\b",
    r"\bis\s+.{0,60}?\ba\s+(?:sell|trap|value\s+trap)\b",
)

HOLD_WORDS = (
    r"\bto\s+hold\b",
    r"\bhold(?:ing)?\s+(?:for|forever|on)\b",
    r"\bbuy\s+and\s+hold\b",
    r"\bwait\b",
    r"\bsit\s+tight\b",
    r"\bon\s+the\s+sidelines\b",
)

WARNING_WORDS = (
    r"\bbad\s+news\b",
    r"\bred\s+flags?\b",
    r"\bwarning\b",
    r"\bbubble\b",
    r"\bcrash(?:ing|es)?\b",
    r"\bsell[\s-]off\b",
    r"\bnosedive[ds]?\b",
    r"\bplunge[ds]?\b",
    r"\btumble[ds]?\b",
    r"\bsink(?:s|ing)?\b",
    r"\bbig\s+questions?\b",
    r"\btrouble\b",
    r"\brisk(?:s|y)?\b",
    r"\bbear\s+market\b",
    r"\bworst\b",
    r"\bmistake\b",
    r"\bdanger(?:ous)?\b",
    r"\bcollapse[ds]?\b",
    r"\bfalling\s+\d+%",
    r"\bdown\s+\d+%\s+from\b",
)

PREDICTION_WORDS = (
    r"^prediction\b",
    r"\bprediction:",
    r"\bi\s+predict\b",
    r"\bwill\s+be\s+worth\b",
    r"\bcould\s+be\s+worth\b",
    r"\bforecast\b",
    r"\bby\s+20[3-9]\d\b",
    r"\bhere'?s\s+what\s+history\s+says\b",
    r"\bhistory\s+says\b",
    r"\bcan\s+double\b",
    r"\bpoised\s+to\b",
    r"\bset\s+to\s+(?:soar|surge|double|thrive)\b",
    r"\bmillionaire[\s-]maker\b",
)

NEWS_WORDS = (
    r"^why\s+",
    r"\bjust\s+(?:got|hit|announced|reported|posted|said)\b",
    r"\bearnings\b",
    r"\bq[1-4]\s+(?:results|earnings)\b",
    r"\bskyrocket(?:ed|s|ing)?\b",
    r"\bsoar(?:ed|s|ing)?\b",
    r"\bjump(?:ed|s|ing)?\b",
    r"\bpop(?:ped|s|ping)?\b",
    r"\brally(?:ing|ied)?\b",
    r"\bannounce[ds]?\b",
    r"\bbreakfast\s+news\b",
    r"\bipo\b",
    r"\bstock\s+split\b",
    r"\bmerger\b|\bacquisition\b|\bacquires\b",
)


def _compile(patterns: tuple[str, ...]) -> list[re.Pattern]:
    return [re.compile(p, re.I) for p in patterns]


_BUY_STRONG = _compile(BUY_STRONG)
_BUY_Q = _compile(BUY_QUESTION)
_SELL_STRONG = _compile(SELL_STRONG)
_SELL_Q = _compile(SELL_QUESTION)
_HOLD = _compile(HOLD_WORDS)
_WARN = _compile(WARNING_WORDS)
_PRED = _compile(PREDICTION_WORDS)
_NEWS = _compile(NEWS_WORDS)


def _any(pats: list[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in pats)


def classify(title: str) -> tuple[str, int]:
    """Return (label, score) for a headline. Score is -2..+2, 0 = no direction."""
    if not title:
        return ("analysis", 0)
    t = " ".join(title.split())

    # Transaction reports first — they contain buy/sell words but no opinion.
    if INSIDER_RE.search(t):
        return ("insider", 0)

    # Then comparisons, for the same reason: buy/sell wording, no single call.
    if COMPARISON_RE.search(t):
        return ("comparison", 0)

    buy_strong, sell_strong = _any(_BUY_STRONG, t), _any(_SELL_STRONG, t)
    buy_q, sell_q = _any(_BUY_Q, t), _any(_SELL_Q, t)

    # "1 Stock to Buy, 1 to Hold, and 1 to Sell" — a single stance would be a lie.
    if (buy_strong or buy_q) and (sell_strong or sell_q):
        return ("mixed", 0)

    if buy_strong:
        return ("buy", 2)
    if sell_strong:
        return ("sell", -2)
    if buy_q:
        # A buy question sitting on top of bad news is a "should you catch this
        # falling knife" piece — still a buy question, but not an endorsement.
        return ("buy_lean", 1)
    if sell_q:
        return ("sell_lean", -1)
    # Explicit buy/sell wording above still wins; only a bare move explainer
    # is demoted to news.
    if MOVE_EXPLAINER_RE.search(t) and not FORWARD_RE.search(t):
        return ("news", 0)
    if _any(_HOLD, t):
        return ("hold", 0)
    if _any(_PRED, t):
        return ("prediction", 0)
    if _any(_WARN, t):
        return ("warning", -1)
    if _any(_NEWS, t):
        return ("news", 0)
    return ("analysis", 0)


#: Display order and colour intent for the dashboard.
LABELS = [
    "buy", "buy_lean", "hold", "mixed", "comparison", "analysis", "prediction",
    "news", "insider", "warning", "sell_lean", "sell",
]
