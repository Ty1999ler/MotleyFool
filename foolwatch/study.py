"""Does getting in early pay?

An event study over Fool coverage. For every primary article about a ticker,
in publication order, measure the forward return from buying at that day's
close - then compare the 1st article against the 2nd, 3rd and later ones.

Three choices do the heavy lifting, and the answer is meaningless without them:

**Market-adjusted returns.** Every return has SPY's return over the *same
trading-day window* subtracted. Without this you measure the market, not the
coverage: over a rising three years almost any entry looks good.

**Trading-day horizons.** Offsets step through the price series itself, so a
"+21 day" return is 21 sessions, not 21 calendar days spanning holidays.

**Censoring control.** A three-year archive cannot tell you the *first* time the
Fool wrote about Apple - it only knows the first time inside the window. Any
ticker whose earliest observed article sits within `censor_days` of the start of
the data is dropped, because its "1st article" is an artefact of where the
backfill happens to begin.

Known biases that remain, and cannot be fixed from this data:

* **Survivorship.** Delisted and acquired tickers have no Yahoo history, so they
  quietly leave the sample. Losers are under-counted and every bucket is
  flattered.
* **Coverage follows momentum.** The Fool tends to write about stocks that have
  already moved. A first article is not a random entry point, so this measures
  an association, never a causal edge.
* **Nothing here accounts for costs, slippage, taxes or position sizing.**

Read the output as "what happened after coverage", not as a strategy.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BENCHMARK = "SPY"

#: Trading-day horizons measured after the entry close.
HORIZONS = (1, 5, 21, 63)

#: Ordinal buckets. "Getting in early" is the 1st-article row.
ORDINAL_BUCKETS = [
    (1, 1, "1st article"),
    (2, 2, "2nd article"),
    (3, 3, "3rd article"),
    (4, 5, "4th-5th"),
    (6, 10, "6th-10th"),
    (11, 10**9, "11th+"),
]


def bucket_of(n: int) -> str:
    for lo, hi, label in ORDINAL_BUCKETS:
        if lo <= n <= hi:
            return label
    return "11th+"


BUCKET_ORDER = [label for _, _, label in ORDINAL_BUCKETS]


@dataclass
class StudyResult:
    events: pd.DataFrame          # one row per article, with forward returns
    by_ordinal: pd.DataFrame      # aggregated by ordinal bucket
    paired: pd.DataFrame          # within-ticker 1st vs 2nd comparison
    by_intensity: pd.DataFrame    # aggregated by how heavily a ticker is covered
    notes: list[str]              # what was excluded, and why


class _PriceBook:
    """Per-symbol arrays of trading dates and closes, for offset lookups."""

    def __init__(self, prices: pd.DataFrame):
        self._dates: dict[str, np.ndarray] = {}
        self._closes: dict[str, np.ndarray] = {}
        for sym, grp in prices.groupby("symbol", sort=False):
            g = grp.sort_values("date")
            self._dates[sym] = g["date"].to_numpy(dtype="datetime64[D]")
            self._closes[sym] = g["close"].to_numpy(dtype=float)

    @classmethod
    def from_db(cls, conn: sqlite3.Connection,
                symbols: "Iterable[str] | None" = None) -> "_PriceBook":
        """Load straight from SQLite, one symbol at a time.

        Walks the (symbol, date) primary-key index into compact numpy arrays
        and never builds a DataFrame of every price row. With history back to
        2019 that is millions of rows, and a pandas frame of them costs several
        hundred megabytes that a dashboard on a shared NAS cannot spare.
        """
        book = cls.__new__(cls)
        book._dates, book._closes = {}, {}
        if symbols is None:
            symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM prices")]
        for sym in dict.fromkeys(s for s in symbols if s):
            rows = conn.execute(
                "SELECT date, close FROM prices WHERE symbol = ? AND close IS NOT NULL "
                "ORDER BY date", (sym,)).fetchall()
            if rows:
                book._dates[sym] = np.array([r[0] for r in rows], dtype="datetime64[D]")
                book._closes[sym] = np.array([r[1] for r in rows], dtype=float)
        return book

    @property
    def symbols(self) -> list[str]:
        return list(self._dates)

    def has(self, sym: str) -> bool:
        return sym in self._dates

    def entry_index(self, sym: str, day: np.datetime64) -> int | None:
        """Index of the first session on or after `day`."""
        dates = self._dates.get(sym)
        if dates is None:
            return None
        i = int(np.searchsorted(dates, day, side="left"))
        return i if i < len(dates) else None

    def forward_return(self, sym: str, entry_i: int, horizon: int) -> float | None:
        closes = self._closes[sym]
        exit_i = entry_i + horizon
        if exit_i >= len(closes):
            return None
        p0, p1 = closes[entry_i], closes[exit_i]
        if not p0 or np.isnan(p0) or np.isnan(p1):
            return None
        return float((p1 - p0) / p0 * 100.0)

    def entry_date(self, sym: str, entry_i: int) -> np.datetime64:
        return self._dates[sym][entry_i]

    def size(self, sym: str) -> int:
        return len(self._dates.get(sym, ()))

    def ret(self, sym: str, i0: int, i1: int) -> float | None:
        """Percent return between two session indices."""
        closes = self._closes.get(sym)
        if closes is None or i0 < 0 or i1 < 0 or i0 >= len(closes) or i1 >= len(closes):
            return None
        p0, p1 = closes[i0], closes[i1]
        if not p0 or np.isnan(p0) or np.isnan(p1):
            return None
        return float((p1 - p0) / p0 * 100.0)

    def ret_between_dates(self, sym: str, d0: np.datetime64,
                          d1: np.datetime64) -> float | None:
        """Return between two calendar dates, snapped to this symbol's sessions.

        Used for the benchmark leg: the stock and SPY share a calendar in
        principle, but a halted or newly listed name can have gaps, so matching
        on dates rather than on index keeps the two legs over the same span.
        """
        i0 = self.entry_index(sym, d0)
        i1 = self.entry_index(sym, d1)
        if i0 is None or i1 is None or i1 <= i0:
            return None
        return self.ret(sym, i0, i1)


def build_events(conn: sqlite3.Connection, *, primary_only: bool = True,
                 censor_days: int = 45,
                 min_articles_per_ticker: int = 2) -> StudyResult:
    """Run the study. Returns aggregates plus the event-level frame behind them."""
    notes: list[str] = []

    cov_filter = "AND c.is_primary = 1" if primary_only else ""
    cov = pd.read_sql_query(
        f"""
        SELECT c.ticker, c.published_day, a.stance_score, a.path, a.title
        FROM coverage c
        JOIN articles a ON a.path = c.path
        LEFT JOIN tickers t ON t.ticker = c.ticker
        WHERE c.verified = 1 {cov_filter}
          AND t.yahoo_symbol IS NOT NULL
        ORDER BY c.ticker, c.published_day
        """,
        conn,
    )
    if cov.empty:
        return StudyResult(cov, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                           ["No coverage with resolved price symbols yet."])

    symbols = pd.read_sql_query(
        "SELECT ticker, yahoo_symbol FROM tickers WHERE yahoo_symbol IS NOT NULL", conn)
    cov = cov.merge(symbols, on="ticker", how="inner")
    cov["published_day"] = pd.to_datetime(cov["published_day"])

    book = _PriceBook.from_db(conn, [*cov["yahoo_symbol"].unique(), BENCHMARK])
    if not book.symbols:
        return StudyResult(cov, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                           ["No price history stored — run `foolwatch prices`."])
    if not book.has(BENCHMARK):
        return StudyResult(cov, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                           [f"No {BENCHMARK} benchmark stored — market-adjusted "
                            "returns are impossible without it. Run "
                            "`foolwatch prices`."])

    # --- censoring -----------------------------------------------------------
    data_start = cov["published_day"].min()
    cutoff = data_start + pd.Timedelta(days=censor_days)
    first_seen = cov.groupby("ticker")["published_day"].min()
    censored = set(first_seen[first_seen <= cutoff].index)
    if censored:
        notes.append(
            f"Dropped {len(censored):,} tickers first seen before "
            f"{cutoff.date()} — with data starting {data_start.date()}, their "
            f"'1st article' is where the backfill begins, not where coverage did.")
        cov = cov[~cov["ticker"].isin(censored)]

    counts = cov.groupby("ticker")["path"].transform("size")
    thin = int((counts < min_articles_per_ticker).sum())
    if thin:
        notes.append(f"Dropped {thin:,} articles on tickers with fewer than "
                     f"{min_articles_per_ticker} articles — nothing to compare.")
        cov = cov[counts >= min_articles_per_ticker]

    if cov.empty:
        notes.append("Nothing survived the filters. The backfill is probably "
                     "still too thin — this needs years of archive, not days.")
        return StudyResult(cov, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), notes)

    cov = cov.sort_values(["ticker", "published_day"]).reset_index(drop=True)
    cov["ordinal"] = cov.groupby("ticker").cumcount() + 1
    cov["bucket"] = cov["ordinal"].map(bucket_of)
    # Days since the ticker's first observed article - "how early is early".
    cov["days_since_first"] = (
        cov["published_day"] - cov.groupby("ticker")["published_day"].transform("min")
    ).dt.days

    # --- forward returns -----------------------------------------------------
    rows: list[dict] = []
    no_price = 0
    for rec in cov.itertuples(index=False):
        sym = rec.yahoo_symbol
        day = np.datetime64(rec.published_day.date(), "D")
        ei = book.entry_index(sym, day)
        bi = book.entry_index(BENCHMARK, day)
        if ei is None or bi is None:
            no_price += 1
            continue
        # A price series that only starts long after the article gives a
        # meaningless entry price.
        if (book.entry_date(sym, ei) - day).astype(int) > 7:
            no_price += 1
            continue
        out = {
            "ticker": rec.ticker, "published_day": rec.published_day,
            "ordinal": rec.ordinal, "bucket": rec.bucket,
            "days_since_first": rec.days_since_first,
            "stance_score": rec.stance_score, "title": rec.title, "path": rec.path,
        }
        usable = False
        for h in HORIZONS:
            r = book.forward_return(sym, ei, h)
            b = book.forward_return(BENCHMARK, bi, h)
            if r is None or b is None:
                out[f"adj_{h}d"] = np.nan
            else:
                out[f"adj_{h}d"] = r - b
                usable = True
        if usable:
            rows.append(out)

    if no_price:
        notes.append(f"Skipped {no_price:,} articles with no usable price history "
                     "at publication.")

    events = pd.DataFrame(rows)
    if events.empty:
        notes.append("No article had enough price history around it to measure. "
                     "Run `foolwatch prices --range 5y` after the backfill.")
        return StudyResult(events, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), notes)

    notes.append("Returns are market-adjusted: SPY's move over the same trading "
                 "days is subtracted from each one.")
    notes.append("Survivorship bias remains — delisted tickers have no price "
                 "history, so they are absent and every bucket looks better "
                 "than reality.")

    return StudyResult(
        events=events,
        by_ordinal=_aggregate(events, "bucket", BUCKET_ORDER),
        paired=_paired_first_vs_second(events),
        by_intensity=_by_intensity(events, cov),
        notes=notes,
    )


#: Calendar months expressed in trading sessions, for the before/after windows.
MONTH_WINDOWS = {"1m": 21, "3m": 63, "12m": 252}


def coverage_timeline(conn: sqlite3.Connection, *, primary_only: bool = True,
                      market_adjusted: bool = True, censor_days: int = 45,
                      min_articles: int = 1) -> tuple[pd.DataFrame, list[str]]:
    """Per company: when coverage started, how fast it ramped, and the price
    path either side of that first article.

    The **before** columns are the point of this table. If a stock is already
    up sharply in the twelve months *before* the Fool's first article, then
    coverage is following the move, not leading it — which is the single
    biggest confounder in "does getting in early pay", made measurable instead
    of just disclaimed.
    """
    notes: list[str] = []
    cov_filter = "AND c.is_primary = 1" if primary_only else ""
    cov = pd.read_sql_query(
        f"""
        SELECT c.ticker, c.published_day, t.name, t.yahoo_symbol
        FROM coverage c
        LEFT JOIN tickers t ON t.ticker = c.ticker
        WHERE c.verified = 1 {cov_filter}
        ORDER BY c.ticker, c.published_day
        """,
        conn,
    )
    if cov.empty:
        return pd.DataFrame(), ["No verified coverage stored yet."]
    cov["published_day"] = pd.to_datetime(cov["published_day"])

    book = _PriceBook.from_db(conn, [*cov["yahoo_symbol"].dropna().unique(), BENCHMARK])
    if not book.symbols:
        return pd.DataFrame(), ["No price history — run `foolwatch prices`."]
    if market_adjusted and not book.has(BENCHMARK):
        notes.append(f"No {BENCHMARK} history stored, so returns are raw, not "
                     "market-adjusted. Run `foolwatch prices`.")
        market_adjusted = False

    data_start = cov["published_day"].min()
    cutoff = data_start + pd.Timedelta(days=censor_days)

    rows = []
    for ticker, g in cov.groupby("ticker", sort=False):
        g = g.sort_values("published_day")
        first = g["published_day"].iloc[0]
        total = len(g)
        if total < min_articles:
            continue
        sym = g["yahoo_symbol"].iloc[0]
        rec = {
            "ticker": ticker,
            "name": g["name"].iloc[0],
            "first_covered": first,
            "articles": total,
            "censored": bool(first <= cutoff),
            # "How fast were they covered at the beginning" — the initial ramp.
            "articles_30d": int((g["published_day"] <= first
                                 + pd.Timedelta(days=30)).sum()),
            "articles_90d": int((g["published_day"] <= first
                                 + pd.Timedelta(days=90)).sum()),
            "days_to_2nd": (int((g["published_day"].iloc[1] - first).days)
                            if total > 1 else None),
            "latest_article": g["published_day"].iloc[-1],
        }

        entry = book.entry_index(sym, np.datetime64(first.date(), "D")) if sym else None
        if entry is not None and (book.entry_date(sym, entry)
                                  - np.datetime64(first.date(), "D")).astype(int) > 7:
            entry = None   # price history starts too late to anchor anything

        if entry is None:
            for lab in MONTH_WINDOWS:
                rec[f"before_{lab}"] = np.nan
                rec[f"after_{lab}"] = np.nan
            rec["since_first"] = np.nan
        else:
            d_entry = book.entry_date(sym, entry)
            for lab, sessions in MONTH_WINDOWS.items():
                # Before: the window ENDING at the first article.
                i0 = entry - sessions
                rec[f"before_{lab}"] = _leg(book, sym, i0, entry, market_adjusted)
                # After: the window STARTING at the first article.
                i1 = entry + sessions
                rec[f"after_{lab}"] = _leg(book, sym, entry, i1, market_adjusted)
            rec["since_first"] = _leg(book, sym, entry, book.size(sym) - 1,
                                      market_adjusted)
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, notes + ["Nothing to show under these filters."]

    n_censored = int(df["censored"].sum())
    if n_censored:
        notes.append(
            f"{n_censored:,} companies are flagged **censored**: their first "
            f"article is within {censor_days} days of {data_start.date()}, the "
            "start of your data, so it is almost certainly not their real first "
            "coverage. Filter them out before drawing conclusions.")
    notes.append(
        "Returns are market-adjusted (SPY over the same sessions subtracted)."
        if market_adjusted else "Returns are raw, not market-adjusted.")
    notes.append("Months are trading sessions: 1m = 21, 3m = 63, 12m = 252.")
    notes.append(
        "Blank cells mean the price series does not reach that far. The 12m "
        "**before** column needs a year of history prior to the article — run "
        "`foolwatch prices --range 5y` (or `max`) to fill it in.")
    return df.sort_values("first_covered", ascending=False).reset_index(drop=True), notes


def _leg(book: "_PriceBook", sym: str, i0: int, i1: int,
         adjusted: bool) -> float:
    """One return leg, optionally with the benchmark's same-span move removed."""
    if i0 < 0 or i1 >= book.size(sym) or i1 <= i0:
        return np.nan
    r = book.ret(sym, i0, i1)
    if r is None:
        return np.nan
    if not adjusted:
        return r
    b = book.ret_between_dates(BENCHMARK, book.entry_date(sym, i0),
                               book.entry_date(sym, i1))
    return np.nan if b is None else r - b


def _aggregate(events: pd.DataFrame, key: str,
               order: list[str] | None = None) -> pd.DataFrame:
    """Mean, median and win rate per horizon. Medians matter: returns are skewed."""
    out = []
    for name, grp in events.groupby(key, sort=False, observed=True):
        row = {key: name, "articles": len(grp), "tickers": grp["ticker"].nunique()}
        for h in HORIZONS:
            col = grp[f"adj_{h}d"].dropna()
            row[f"n_{h}d"] = len(col)
            row[f"mean_{h}d"] = col.mean() if len(col) else np.nan
            row[f"median_{h}d"] = col.median() if len(col) else np.nan
            row[f"win_{h}d"] = (col > 0).mean() * 100 if len(col) else np.nan
        out.append(row)
    df = pd.DataFrame(out)
    if order is not None and not df.empty:
        df[key] = pd.Categorical(df[key], categories=order, ordered=True)
        df = df.sort_values(key)
    return df.reset_index(drop=True)


def _paired_first_vs_second(events: pd.DataFrame) -> pd.DataFrame:
    """Within-ticker 1st vs 2nd article.

    The sharpest form of the question: comparing two entries in the *same*
    stock removes every difference between stocks, which the bucket averages
    cannot. Paired differences plus a sign test on how often first beat second.
    """
    first = events[events["ordinal"] == 1].set_index("ticker")
    second = events[events["ordinal"] == 2].set_index("ticker")
    both = first.index.intersection(second.index)
    if not len(both):
        return pd.DataFrame()

    rows = []
    for h in HORIZONS:
        a = first.loc[both, f"adj_{h}d"]
        b = second.loc[both, f"adj_{h}d"]
        pair = pd.DataFrame({"first": a, "second": b}).dropna()
        if pair.empty:
            continue
        diff = pair["first"] - pair["second"]
        rows.append({
            "horizon": f"{h}d",
            "tickers": len(pair),
            "first_mean": pair["first"].mean(),
            "second_mean": pair["second"].mean(),
            "first_median": pair["first"].median(),
            "second_median": pair["second"].median(),
            "mean_diff": diff.mean(),
            "median_diff": diff.median(),
            "first_wins_pct": (diff > 0).mean() * 100,
        })
    return pd.DataFrame(rows)


def _by_intensity(events: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """Does *how often* the Fool writes about a ticker matter?

    Buckets tickers by total article count, then measures the forward return of
    their articles. Note the direction of causation is unknowable here: heavy
    coverage may follow a good run rather than precede one.
    """
    totals = cov.groupby("ticker")["path"].size().rename("total_articles")
    ev = events.merge(totals, left_on="ticker", right_index=True, how="left")
    bins = [0, 2, 5, 10, 25, 10**9]
    labels = ["2 articles", "3-5", "6-10", "11-25", "26+"]
    ev["intensity"] = pd.cut(ev["total_articles"], bins=bins, labels=labels,
                             right=True)
    ev = ev.dropna(subset=["intensity"])
    if ev.empty:
        return pd.DataFrame()
    return _aggregate(ev, "intensity", labels)


#: Sample size below which no bucket comparison is worth reading.
MIN_PAIRS = 30

#: Below this many calls, an author's hit rate is noise. 76 authors each with a
#: handful of calls will always produce an impressive-looking leader by chance.
MIN_CALLS = 20


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion, as percentages.

    Used instead of hits/n because a normal-approximation error bar is badly
    wrong at the sample sizes here (3 for 4 is not a 75% skill estimate).
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


def author_scores(conn: sqlite3.Connection, *, horizon: int = 21,
                  primary_only: bool = True, censor_days: int = 0,
                  min_calls: int = MIN_CALLS) -> tuple[pd.DataFrame, dict, list[str]]:
    """How often does each author's directional call go the right way?

    A "call" is an article whose headline stance has a direction: bullish
    (`buy`, `buy_lean`) or bearish (`sell`, `sell_lean`, `warning`). Articles
    scored 0 - news, predictions, insider filings, plain analysis - are not
    calls and are excluded rather than counted as wrong.

    A call is "right" when the market-adjusted return over `horizon` trading
    days moves the way the call pointed. Market-adjusted matters more here than
    anywhere else in this module: in a rising market every bullish call looks
    prescient, and since bullish calls outnumber bearish ones ~30:1, raw
    accuracy would mostly measure the market.

    Returns (per-author frame, base-rate summary, caveats).
    """
    res = build_events(conn, primary_only=primary_only, censor_days=censor_days,
                       min_articles_per_ticker=1)
    notes = list(res.notes)
    col = f"adj_{horizon}d"
    if res.events.empty or col not in res.events:
        return pd.DataFrame(), {}, notes + ["No measurable events yet."]

    ev = res.events.dropna(subset=[col]).copy()
    # Attach authors; build_events works at ticker/article level.
    authors = pd.read_sql_query("SELECT path, author FROM articles", conn)
    ev = ev.merge(authors, on="path", how="left")

    calls = ev[ev["stance_score"] != 0].copy()
    if calls.empty:
        return pd.DataFrame(), {}, notes + [
            f"No directional calls have {horizon} trading days of price history "
            "after them yet."]

    calls["bullish"] = calls["stance_score"] > 0
    calls["right"] = np.where(calls["bullish"], calls[col] > 0, calls[col] < 0)
    # Signed "did the call pay" return: bearish calls profit when price falls.
    calls["call_return"] = np.where(calls["bullish"], calls[col], -calls[col])

    base = {
        "calls": len(calls),
        "hit_rate": calls["right"].mean() * 100,
        "bullish_calls": int(calls["bullish"].sum()),
        "bearish_calls": int((~calls["bullish"]).sum()),
        "mean_call_return": calls["call_return"].mean(),
        "authors": calls["author"].nunique(),
        "horizon": horizon,
    }

    grp = calls.groupby("author", dropna=True)
    rows = []
    for author, g in grp:
        n, hits = len(g), int(g["right"].sum())
        lo, hi = wilson(hits, n)
        rows.append({
            "author": author,
            "calls": n,
            "hits": hits,
            "hit_rate": hits / n * 100,
            "ci_low": lo,
            "ci_high": hi,
            "vs_base": hits / n * 100 - base["hit_rate"],
            "mean_call_return": g["call_return"].mean(),
            "median_call_return": g["call_return"].median(),
            "bullish_share": g["bullish"].mean() * 100,
            "tickers": g["ticker"].nunique(),
        })
    df = pd.DataFrame(rows)
    # Only authors whose interval clears the base rate are distinguishable from
    # the crowd at all - flag it rather than ranking on raw hit rate.
    df["beats_base"] = df["ci_low"] > base["hit_rate"]
    df["enough_data"] = df["calls"] >= min_calls
    df = df.sort_values(["enough_data", "hit_rate", "calls"],
                        ascending=[False, False, False]).reset_index(drop=True)

    notes.append(
        f"A call is a headline with a direction. {base['calls']:,} of the "
        f"articles measured here are calls ({base['bullish_calls']:,} bullish, "
        f"{base['bearish_calls']:,} bearish); the rest state no view and are "
        "excluded rather than marked wrong.")
    notes.append(
        f"Base rate: {base['hit_rate']:.0f}% of all calls went the right way. "
        "Judge an author against that, not against 50% — and note the interval, "
        "not the headline number.")
    qualified = int(df["enough_data"].sum())
    notes.append(
        f"Only {qualified} of {len(df)} authors have {min_calls}+ calls. With "
        f"{len(df)} authors being ranked, the best-looking one will beat the "
        "base rate by luck alone; a Wilson interval that clears the base rate "
        "is the minimum bar for taking a name seriously.")
    return df, base, notes


def verdict(paired: pd.DataFrame, by_ordinal: pd.DataFrame) -> str:
    """One honest sentence about what the numbers do and do not support.

    Deliberately hard to get an exciting answer out of. The mean, the median and
    the win rate have to agree before this will call a result a signal: a
    positive mean sitting on a flat median and a sub-50% win rate is a handful
    of outliers, which is the single easiest way to fool yourself here.
    """
    if paired.empty:
        return ("No paired 1st/2nd articles with usable prices yet — nothing to "
                "compare. This needs the full backfill.")

    # Prefer the longest horizon that actually has data.
    row = paired.iloc[-1]
    for h in ("21d", "63d", "5d"):
        match = paired[paired["horizon"] == h]
        if not match.empty:
            row = match.iloc[0]
            break

    horizon = row["horizon"]
    n = int(row["tickers"])
    mean_d, med_d, wins = row["mean_diff"], row["median_diff"], row["first_wins_pct"]

    if n < MIN_PAIRS:
        return (f"Only {n} tickers have both a 1st and a 2nd article with prices "
                f"({horizon} horizon) — far too few to conclude anything. "
                "Finish the backfill.")

    # Do the three measures point the same way? A win rate inside the 45-55
    # band casts no directional vote: treating an exact 50% as bearish would
    # make a textbook coin flip look like disagreement.
    near_even = 45 <= wins <= 55
    win_vote = 0.0 if near_even else (1.0 if wins > 50 else -1.0)
    signs = {np.sign(mean_d), np.sign(med_d), win_vote}
    agree = len(signs - {0.0}) == 1 and np.sign(med_d) != 0

    body = (f"Over {horizon}, across {n:,} tickers: the 1st article beat the 2nd "
            f"by {mean_d:+.2f} pts on average, {med_d:+.2f} pts at the median, "
            f"and won {wins:.0f}% of the time.")

    if not agree:
        return (f"{body} Those disagree, which means a few large outliers are "
                "driving the average rather than a consistent edge — treat this "
                "as no evidence either way.")
    if abs(med_d) < 0.5 or near_even:
        return f"{body} That is too small and too close to a coin flip to act on."
    direction = "earlier was better" if mean_d > 0 else "earlier was worse"
    return (f"{body} All three agree, so {direction} in this sample. Still an "
            "association, not an edge: coverage follows momentum, delisted "
            "names are missing, and costs are not modelled.")


# --- track record ------------------------------------------------------------
#
# What happened after every call. The event study above asks a narrow question
# (did the first article beat later ones); this answers the broad one: when the
# Fool said buy, or sell, what did the stock actually do next?

#: Holding periods for the track record, in trading sessions.
TRACK_HORIZONS: dict[str, int] = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
_LONGEST = max(TRACK_HORIZONS.values())


def compute_call_outcomes(conn: sqlite3.Connection, *, full: bool = False) -> dict:
    """Score what happened after every primary article, and store it.

    One row per primary (article, ticker): the stock's return and SPY's return
    over the same sessions at 1, 3, 6 and 12 months. Stored raw, not
    market-adjusted, so the dashboard can show both and can draw a strategy
    line against SPY on the same axis.

    Incremental by default. A row whose 12-month legs are both filled can never
    change again, so the daily run only revisits calls whose windows are still
    maturing, plus new ones. Calls with no usable prices are still written, with
    NULL returns, so the page can say how many calls could not be scored rather
    than silently dropping them — which is where survivorship bias hides.
    """
    from .db import utc_now

    maturing = "" if full else (
        "AND (o.path IS NULL OR o.ret_252 IS NULL OR o.spy_252 IS NULL)")
    rows = conn.execute(
        f"""
        SELECT c.path, c.ticker, c.published_day, t.yahoo_symbol
        FROM coverage c
        JOIN tickers t ON t.ticker = c.ticker
        LEFT JOIN call_outcomes o ON o.path = c.path AND o.ticker = c.ticker
        WHERE c.is_primary = 1 AND c.verified = 1
          AND t.yahoo_symbol IS NOT NULL
          {maturing}
        """
    ).fetchall()
    if not rows:
        return {"computed": 0, "scored": 0}

    book = _PriceBook.from_db(conn, [*{r["yahoo_symbol"] for r in rows}, BENCHMARK])
    if not book.has(BENCHMARK):
        log.warning("No %s history; cannot score calls", BENCHMARK)
        return {"computed": 0, "scored": 0}

    now = utc_now()
    out, scored = [], 0
    for r in rows:
        sym, day = r["yahoo_symbol"], np.datetime64(r["published_day"], "D")
        rec = [r["path"], r["ticker"], r["published_day"], None, None]
        legs: list[float | None] = [None] * (2 * len(TRACK_HORIZONS))
        ei = book.entry_index(sym, day) if book.has(sym) else None
        # A series that only starts well after the article gives a fake entry.
        if ei is not None and (book.entry_date(sym, ei) - day).astype(int) <= 7:
            d0 = book.entry_date(sym, ei)
            rec[3] = str(d0)
            rec[4] = float(book._closes[sym][ei])
            for k, h in enumerate(TRACK_HORIZONS.values()):
                if ei + h < book.size(sym):
                    legs[2 * k] = book.ret(sym, ei, ei + h)
                    legs[2 * k + 1] = book.ret_between_dates(
                        BENCHMARK, d0, book.entry_date(sym, ei + h))
            scored += 1
        out.append((*rec, *legs, now))

    conn.executemany(
        "INSERT OR REPLACE INTO call_outcomes (path, ticker, published_day, "
        "entry_date, entry_close, ret_21, spy_21, ret_63, spy_63, ret_126, "
        "spy_126, ret_252, spy_252, computed_utc) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", out)
    conn.commit()
    log.info("Call outcomes: %d computed, %d with a usable entry price",
             len(out), scored)
    return {"computed": len(out), "scored": scored}


def track_record_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """Every scored call, joined to fresh article metadata, with outcomes.

    Adds, per horizon label (1m/3m/6m/12m):
      excess_<h>  stock return minus SPY over the same sessions
      call_<h>    excess signed by the call's direction, so positive always
                  means "the call paid": a bearish call scores when the stock
                  lags. NaN for articles that took no direction.
      hit_<h>     call_<h> > 0, NaN where there is no directional call
    """
    df = pd.read_sql_query(
        """
        SELECT o.*, a.title, a.author, a.stance, a.stance_score
        FROM call_outcomes o
        JOIN articles a ON a.path = o.path
        """,
        conn,
    )
    if df.empty:
        return df
    df["published_day"] = pd.to_datetime(df["published_day"])
    direction = np.sign(df["stance_score"]).replace(0, np.nan)
    for label, h in TRACK_HORIZONS.items():
        excess = df[f"ret_{h}"] - df[f"spy_{h}"]
        df[f"excess_{label}"] = excess
        df[f"call_{label}"] = excess * direction
        df[f"hit_{label}"] = np.where(df[f"call_{label}"].isna(), np.nan,
                                      (df[f"call_{label}"] > 0).astype(float))
    return df


def scorecard(df: pd.DataFrame, group: str = "stance") -> pd.DataFrame:
    """Hit rate and median/mean outcome per group at every horizon.

    Medians sit beside means because a handful of 10-baggers can carry a mean
    on their own; the Wilson interval says how much the hit rate can be trusted.
    """
    calls = df[df["stance_score"] != 0]
    rows = []
    for name, g in calls.groupby(group, sort=False):
        row = {group: name, "calls": len(g)}
        for label in TRACK_HORIZONS:
            col = g[f"call_{label}"].dropna()
            hits = int((col > 0).sum())
            lo, hi = wilson(hits, len(col))
            row.update({
                f"n_{label}": len(col),
                f"hit_{label}": hits / len(col) * 100 if len(col) else np.nan,
                f"lo_{label}": lo, f"hi_{label}": hi,
                f"median_{label}": col.median() if len(col) else np.nan,
                f"mean_{label}": col.mean() if len(col) else np.nan,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def follow_the_calls(df: pd.DataFrame, label: str = "1m") -> pd.DataFrame:
    """Growth of $1 from buying every bullish call, month by month, vs SPY.

    Each calendar month's bullish calls form an equal-weight basket, held for
    the horizon, and the baskets are chained. A ticker is counted once per
    month however many bullish articles it got — otherwise the stocks the Fool
    writes about constantly (Nvidia, Tesla) would dominate every basket, and no
    real portfolio buys the same stock twenty times in a month.

    Only meaningful at the 1-month horizon, where consecutive baskets barely
    overlap; longer holds overlap months and chaining them would overstate
    compounding. Approximate either way: entries fall on different days within
    the month, and costs, taxes and slippage are ignored.
    """
    h = TRACK_HORIZONS[label]
    bull = df[(df["stance_score"] > 0)
              & df[f"ret_{h}"].notna() & df[f"spy_{h}"].notna()].copy()
    if bull.empty:
        return pd.DataFrame()
    bull["month"] = bull["published_day"].dt.to_period("M").dt.to_timestamp()
    bull = bull.sort_values("published_day").drop_duplicates(["month", "ticker"])
    m = (bull.groupby("month")
             .agg(calls=("ticker", "size"),
                  basket=(f"ret_{h}", "mean"),
                  spy=(f"spy_{h}", "mean"))
             .reset_index())
    m["follow_growth"] = (1 + m["basket"] / 100).cumprod()
    m["spy_growth"] = (1 + m["spy"] / 100).cumprod()
    return m
