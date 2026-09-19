"""foolwatch Streamlit dashboard.

Four views over the same SQLite file: an overview of what the Fool is covering,
a per-ticker history (first coverage, how the take moved, price alongside), a
first-coverage scoreboard, and an author breakdown.

Chart conventions, applied deliberately:
  * Coverage volume and price never share a y-axis. Two measures of different
    scale get two charts stacked on a common date axis, never a dual axis.
  * Stance is a diverging measure around a real zero, so it gets the validated
    blue<->red diverging pair with a zero baseline - blue bullish, red bearish.
    (Blue/red rather than the usual green/red: green vs red is the least
    colourblind-safe pair there is, and this one is measured safe.)
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Streamlit executes this file as a script, not as a package member, so the
# relative imports a module would use are not available. Put the project root
# on the path and import absolutely.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foolwatch.config import DB_PATH, load  # noqa: E402
from foolwatch import study  # noqa: E402

BASE = "https://www.fool.com"

# Validated diverging pair (see references/palette.md); both modes pass all six
# checks at all-pairs strictness.
BULL = "#2a78d6"
BEAR = "#e34948"
NEUTRAL = "#898781"

#: stance -> bucket, so eleven labels collapse to a five-step diverging scale.
BUCKET = {
    "buy": "Buy", "buy_lean": "Buy (question)",
    "hold": "No call", "mixed": "No call", "analysis": "No call",
    "prediction": "No call", "news": "No call", "insider": "No call",
    "comparison": "No call",
    "warning": "Caution", "sell_lean": "Sell (question)", "sell": "Sell",
}
BUCKET_ORDER = ["Buy", "Buy (question)", "No call", "Caution",
                "Sell (question)", "Sell"]

st.set_page_config(page_title="foolwatch", page_icon="🃏", layout="wide")


# --- data access -------------------------------------------------------------

@st.cache_resource
def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        st.error(f"No database at {DB_PATH}.\n\n"
                 "Run `py -m foolwatch setup` then `py -m foolwatch daily` first.")
        st.stop()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True,
                           check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def as_dates(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """DateColumn needs real dates; every day column comes out of SQLite as text."""
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


@st.cache_data(ttl="10m")
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, get_conn(), params=params)


@st.cache_data(ttl="10m")
def db_bounds() -> tuple[str, str]:
    df = q("SELECT MIN(published_day) mn, MAX(published_day) mx FROM articles")
    if df.empty or pd.isna(df.at[0, "mn"]):
        return ("2020-01-01", date.today().isoformat())
    return (df.at[0, "mn"], df.at[0, "mx"])


def stance_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = df["stance"].map(BUCKET).fillna("No call")
    return df


def bucket_by_time(daily: pd.DataFrame, day_col: str) -> tuple[pd.DataFrame, str, str]:
    """Roll daily rows up to a period that suits the span.

    Returns (frame, axis label, Altair timeUnit). Monthly buckets are right for
    three years of archive and useless for a five-day-old database, where they
    collapse to a single sliver. The timeUnit matters as much as the resample:
    without it Altair treats the dates as continuous, draws hairline bars and
    labels the axis by hour.
    """
    if daily.empty:
        return daily, "Date", "yearmonthdate"
    d = daily.copy()
    d[day_col] = pd.to_datetime(d[day_col])
    span = (d[day_col].max() - d[day_col].min()).days
    freq, label, unit = (
        ("D", "Day", "yearmonthdate") if span <= 45 else
        ("W-MON", "Week", "yearweek") if span <= 400 else
        ("MS", "Month", "yearmonth")
    )
    out = (d.set_index(day_col)
            .resample(freq)[["articles", "net_stance"]].sum()
            .reset_index()
            .rename(columns={day_col: "period"}))
    # Resampling fills empty periods with zeros; drop leading/trailing padding.
    nz = out.index[out["articles"] > 0]
    if len(nz):
        out = out.loc[nz[0]:nz[-1]]
    return out, label, unit


# --- shared chart builders ---------------------------------------------------

def volume_chart(df: pd.DataFrame, x_label: str, unit: str):
    """Magnitude over time: single-series bars, so no legend is needed."""
    return (
        alt.Chart(df)
        .mark_bar(color=BULL, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("period:T", timeUnit=unit, title=x_label,
                    axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("articles:Q", title="Articles"),
            tooltip=[alt.Tooltip("period:T", timeUnit=unit, title=x_label),
                     alt.Tooltip("articles:Q", title="Articles")],
        )
        .properties(height=210)
    )


def net_stance_chart(df: pd.DataFrame, x_label: str, unit: str):
    """Polarity over time: diverging bars about a true zero baseline."""
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
                  cornerRadiusBottomLeft=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("period:T", timeUnit=unit, title=x_label,
                    axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("net_stance:Q", title="Net stance (bullish +)"),
            color=alt.condition(alt.datum.net_stance >= 0,
                                alt.value(BULL), alt.value(BEAR)),
            tooltip=[alt.Tooltip("period:T", timeUnit=unit, title=x_label),
                     alt.Tooltip("net_stance:Q", title="Net stance"),
                     alt.Tooltip("articles:Q", title="Articles")],
        )
        .properties(height=210)
    )


def price_chart(df: pd.DataFrame):
    """One series, named by the card heading, so it carries no legend box.

    Binned to the day the closes actually are, otherwise a short span gets an
    axis labelled by the hour.
    """
    return (
        alt.Chart(df)
        .mark_line(color=BULL, strokeWidth=2)
        .encode(
            x=alt.X("date:T", timeUnit="yearmonthdate", title="Date",
                    axis=alt.Axis(labelAngle=-40, format="%b %d, %Y")),
            y=alt.Y("close:Q", title="Close ($)",
                    scale=alt.Scale(zero=False, nice=True)),
            tooltip=[alt.Tooltip("date:T", timeUnit="yearmonthdate", title="Date"),
                     alt.Tooltip("close:Q", title="Close", format="$.2f")],
        )
        .properties(height=230)
    )


# --- sidebar -----------------------------------------------------------------

cfg = load()
lo, hi = db_bounds()

with st.sidebar:
    st.title("🃏 foolwatch")
    st.caption("What The Motley Fool covers, when it started, and how the take moved.")

    span = st.date_input(
        "Published between",
        value=(date.fromisoformat(lo), date.fromisoformat(hi)),
        min_value=date.fromisoformat(lo),
        max_value=date.fromisoformat(hi),
    )
    start, end = (span if isinstance(span, tuple) and len(span) == 2
                  else (date.fromisoformat(lo), date.fromisoformat(hi)))

    primary_only = st.toggle(
        "Primary coverage only", value=cfg.primary_only_default,
        help="fool.com tags every ticker an article touches. Primary coverage is "
             "the one company the article is actually about — without this, a "
             "portfolio round-up invents a dozen false first mentions.",
    )
    verified_only = st.toggle(
        "Listed tickers only", value=False,
        help="Drop tickers absent from the NASDAQ/NYSE listing files "
             "(crypto symbols, foreign OTC lines).",
    )

COV_FILTER = " AND c.is_primary = 1" if primary_only else ""
if verified_only:
    COV_FILTER += " AND c.verified = 1"
DAYS = (start.isoformat(), end.isoformat())


# --- views -------------------------------------------------------------------

def article_panel(ticker: str, name: str | None = None, limit: int = 25) -> None:
    """Recent articles for one ticker: headline, author, stance, clickable link.

    Used from the overview table so a company can be opened in place, rather
    than switching tab and hunting for it in a dropdown again.
    """
    rows = q(
        f"""
        SELECT c.published_day, a.title, a.author, a.stance, a.stance_score,
               c.is_primary, a.path
        FROM coverage c JOIN articles a ON a.path = c.path
        WHERE c.ticker = ?{COV_FILTER}
        ORDER BY c.published_day DESC, a.title
        LIMIT {int(limit)}
        """, (ticker,))
    label = f"{ticker} — {name}" if name else ticker
    if rows.empty:
        st.info(f"No articles for {label} under the current filters.")
        return

    total = q(f"SELECT COUNT(*) n FROM coverage c WHERE c.ticker = ?{COV_FILTER}",
              (ticker,)).at[0, "n"]
    st.markdown(f"**{label}** — {total} article(s); showing the "
                f"{min(limit, int(total))} most recent.")

    tbl = stance_frame(rows)
    tbl["url"] = BASE + tbl["path"]
    tbl["primary"] = tbl["is_primary"].astype(bool)
    st.dataframe(
        as_dates(tbl, "published_day")[
            ["published_day", "title", "author", "bucket", "primary", "url"]],
        hide_index=True,
        column_config={
            "published_day": st.column_config.DateColumn("Published"),
            "title": st.column_config.TextColumn("Headline", width="large"),
            "author": st.column_config.TextColumn("Author"),
            "bucket": st.column_config.TextColumn("Stance"),
            "primary": st.column_config.CheckboxColumn(
                "Primary", help="The company the article is actually about."),
            "url": st.column_config.LinkColumn("Link", display_text="read"),
        },
    )


def view_overview() -> None:
    kpi = q(
        f"""
        SELECT (SELECT COUNT(*) FROM articles WHERE published_day BETWEEN ? AND ?) AS articles,
               (SELECT COUNT(DISTINCT c.ticker) FROM coverage c
                 WHERE c.published_day BETWEEN ? AND ?{COV_FILTER}) AS tickers,
               (SELECT COUNT(DISTINCT author) FROM articles
                 WHERE published_day BETWEEN ? AND ?) AS authors
        """,
        DAYS * 3,
    )
    mix = stance_frame(q(
        "SELECT stance, COUNT(*) n FROM articles WHERE published_day BETWEEN ? AND ? "
        "GROUP BY stance", DAYS))
    bull = int(mix.loc[mix.bucket.isin(["Buy", "Buy (question)"]), "n"].sum())
    bear = int(mix.loc[mix.bucket.isin(["Sell", "Sell (question)", "Caution"]), "n"].sum())

    with st.container(horizontal=True):
        st.metric("Articles", f"{int(kpi.at[0, 'articles']):,}", border=True)
        st.metric("Tickers covered", f"{int(kpi.at[0, 'tickers']):,}", border=True)
        st.metric("Authors", f"{int(kpi.at[0, 'authors']):,}", border=True)
        st.metric("Bullish articles", f"{bull:,}", border=True)
        st.metric("Bearish or cautious", f"{bear:,}", border=True)

    if bear:
        st.caption(f"The Fool publishes roughly **{bull / bear:.1f} bullish articles "
                   f"for every cautious or bearish one** in this window — worth "
                   f"remembering before reading any single article as a signal.")

    daily = q(
        """
        SELECT published_day AS day,
               COUNT(*) AS articles,
               SUM(stance_score) AS net_stance
        FROM articles WHERE published_day BETWEEN ? AND ?
        GROUP BY 1 ORDER BY 1
        """, DAYS)
    if daily.empty:
        st.info("No articles in this window yet.")
        return
    periods, x_label, unit = bucket_by_time(daily, "day")

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Coverage volume")
            st.altair_chart(volume_chart(periods, x_label, unit))
    with right:
        with st.container(border=True):
            st.subheader("Net stance")
            st.caption("Sum of headline stance per period: blue bullish, red bearish.")
            st.altair_chart(net_stance_chart(periods, x_label, unit))

    with st.container(border=True):
        st.subheader("Most covered tickers")
        st.caption("Select a row to read that company's recent articles.")
        top = q(
            f"""
            SELECT c.ticker,
                   t.name,
                   COUNT(*) AS articles,
                   MIN(c.published_day) AS first_covered,
                   MAX(c.published_day) AS last_covered,
                   ROUND(AVG(a.stance_score), 2) AS avg_stance
            FROM coverage c
            JOIN articles a ON a.path = c.path
            LEFT JOIN tickers t ON t.ticker = c.ticker
            WHERE c.published_day BETWEEN ? AND ?{COV_FILTER}
            GROUP BY c.ticker ORDER BY articles DESC LIMIT 200
            """, DAYS)
        event = st.dataframe(
            as_dates(top, "first_covered", "last_covered"), hide_index=True,
            on_select="rerun", selection_mode="single-row",
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", pinned=True),
                "name": st.column_config.TextColumn("Company", width="medium"),
                "articles": st.column_config.NumberColumn("Articles"),
                "first_covered": st.column_config.DateColumn("First covered"),
                "last_covered": st.column_config.DateColumn("Last covered"),
                "avg_stance": st.column_config.NumberColumn(
                    "Avg stance", format="%.2f",
                    help="Mean headline stance, -2 (sell) to +2 (buy)."),
            },
        )
        picked = event.selection.rows
        if picked:
            row = top.iloc[picked[0]]
            article_panel(row["ticker"], row["name"])
        else:
            st.caption("Nothing selected.")

    with st.container(border=True):
        st.subheader("Stance mix")
        mix_b = (mix.groupby("bucket", as_index=False)["n"].sum()
                 .sort_values("n", ascending=False))
        st.altair_chart(
            alt.Chart(mix_b).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("n:Q", title="Articles"),
                y=alt.Y("bucket:N", title=None, sort=BUCKET_ORDER),
                color=alt.Color("bucket:N", sort=BUCKET_ORDER,
                                scale=alt.Scale(domain=BUCKET_ORDER,
                                                range=[BULL, "#86b6ef", NEUTRAL,
                                                       "#f0a3a2", "#e88a89", BEAR]),
                                legend=alt.Legend(title="Stance")),
                tooltip=["bucket:N", "n:Q"],
            ).properties(height=220)
        )


def view_ticker() -> None:
    opts = q(
        f"""
        SELECT c.ticker, COUNT(*) n, MAX(t.name) AS name
        FROM coverage c
        LEFT JOIN tickers t ON t.ticker = c.ticker
        WHERE 1=1{COV_FILTER} GROUP BY c.ticker ORDER BY n DESC, c.ticker
        """)
    if opts.empty:
        st.info("No coverage stored yet — run `py -m foolwatch daily` first.")
        return

    # Streamlit filters the dropdown on the rendered label, so putting the
    # company name in it makes "nvidia" find NVDA.
    labels = {
        r.ticker: (f"{r.ticker} — {r.name}  ({r.n} articles)" if r.name
                   else f"{r.ticker}  ({r.n} articles)")
        for r in opts.itertuples(index=False)
    }
    ticker = st.selectbox(
        "Ticker or company", opts["ticker"],
        format_func=lambda t: labels.get(t, t),
        help="Type a ticker or a company name to filter.",
    )

    hist = q(
        f"""
        SELECT c.published_day, a.title, a.author, a.stance, a.stance_score,
               c.is_primary, a.path
        FROM coverage c JOIN articles a ON a.path = c.path
        WHERE c.ticker = ?{COV_FILTER}
        ORDER BY c.published_day DESC
        """, (ticker,))
    if hist.empty:
        st.info("No coverage for that ticker under the current filters.")
        return

    first_day = hist["published_day"].min()
    last_day = hist["published_day"].max()
    meta = q("SELECT name, yahoo_symbol FROM tickers WHERE ticker = ?", (ticker,))
    name = (meta.at[0, "name"] if not meta.empty and meta.at[0, "name"] else ticker)
    symbol = meta.at[0, "yahoo_symbol"] if not meta.empty else None

    st.subheader(f"{ticker} — {name}")

    px = pd.DataFrame()
    ret = spy_ret = None
    if symbol:
        px = q("SELECT date, close FROM prices WHERE symbol = ? AND date >= ? ORDER BY date",
               (symbol, first_day))
        spy = q("SELECT date, close FROM prices WHERE symbol = 'SPY' AND date >= ? ORDER BY date",
                (first_day,))
        if len(px) > 1:
            ret = (px["close"].iloc[-1] - px["close"].iloc[0]) / px["close"].iloc[0] * 100
        if len(spy) > 1:
            spy_ret = (spy["close"].iloc[-1] - spy["close"].iloc[0]) / spy["close"].iloc[0] * 100

    with st.container(horizontal=True):
        st.metric("First covered", first_day, border=True)
        st.metric("Days of coverage",
                  f"{(date.fromisoformat(last_day) - date.fromisoformat(first_day)).days:,}",
                  border=True)
        st.metric("Articles", f"{len(hist):,}", border=True)
        st.metric("Avg stance", f"{hist['stance_score'].mean():+.2f}", border=True)
        if ret is not None:
            delta = f"{ret - spy_ret:+.1f} pts vs SPY" if spy_ret is not None else None
            st.metric("Since first coverage", f"{ret:+.1f}%", delta, border=True)

    if ret is not None and spy_ret is None:
        st.caption("No SPY baseline stored for this span — run `py -m foolwatch prices`.")

    per_day = (hist.groupby("published_day", as_index=False)
               .agg(articles=("path", "count"), net_stance=("stance_score", "sum"))
               .rename(columns={"published_day": "day"}))
    periods, x_label, unit = bucket_by_time(per_day, "day")

    if not px.empty:
        px = px.assign(date=pd.to_datetime(px["date"]))
        with st.container(border=True):
            st.subheader(f"{ticker} closing price since first coverage")
            st.altair_chart(price_chart(px))

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Articles over time")
            st.altair_chart(volume_chart(periods, x_label, unit))
    with right:
        with st.container(border=True):
            st.subheader("Net stance over time")
            if periods["net_stance"].abs().sum() == 0:
                st.caption(
                    f"Every {ticker} headline in this window is non-directional — "
                    "news, predictions and analysis, with no buy or sell call. "
                    "That is itself the finding, so there is no chart to draw.")
            else:
                st.caption("Blue bullish, red bearish. Flat means coverage "
                           "with no call.")
                st.altair_chart(net_stance_chart(periods, x_label, unit))

    with st.container(border=True):
        st.subheader("Every article")
        table = stance_frame(hist)
        table["url"] = BASE + table["path"]
        table["primary"] = table["is_primary"].astype(bool)
        st.dataframe(
            as_dates(table, "published_day")[
                ["published_day", "title", "author", "bucket", "stance_score",
                 "primary", "url"]],
            hide_index=True,
            column_config={
                "published_day": st.column_config.DateColumn("Published", pinned=True),
                "title": st.column_config.TextColumn("Headline", width="large"),
                "author": st.column_config.TextColumn("Author"),
                "bucket": st.column_config.TextColumn("Stance"),
                "stance_score": st.column_config.NumberColumn("Score", format="%+d"),
                "primary": st.column_config.CheckboxColumn(
                    "Primary", help="The company the article is actually about."),
                "url": st.column_config.LinkColumn("Link", display_text="open"),
            },
        )


@st.cache_data(ttl="30m", show_spinner="Building coverage timeline…")
def run_timeline(primary: bool, adjusted: bool, censor_days: int):
    return study.coverage_timeline(get_conn(), primary_only=primary,
                                   market_adjusted=adjusted,
                                   censor_days=censor_days)


def view_first_coverage() -> None:
    st.caption(
        "When each company was first covered, **how fast the coverage ramped**, "
        "and the price path on both sides of that first article. Select a row "
        "to read its articles."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        adjusted = st.toggle("Market-adjusted (vs SPY)", value=True,
                             help="Subtract SPY's move over the same sessions. "
                                  "Off gives raw price returns.")
    with c2:
        hide_censored = st.toggle(
            "Hide censored companies", value=True,
            help="A company whose first article sits at the very start of your "
                 "data was almost certainly covered before that.")
    with c3:
        censor = st.number_input("Censoring guard (days)", 0, 365, 45, 15,
                                 key="fc_censor")

    rows, notes = run_timeline(primary_only, adjusted, int(censor))
    if rows.empty:
        st.info(" ".join(notes) or "Nothing to show yet.")
        return

    shown = rows[~rows["censored"]] if hide_censored else rows
    if shown.empty:
        st.warning(
            f"Every company here is censored — all {len(rows):,} were first seen "
            "within the guard window, so none of these are real first-coverage "
            "dates. Expected until the backfill has depth; untick the filter to "
            "look anyway.")
        return

    before_cols = [f"before_{m}" for m in study.MONTH_WINDOWS]
    after_cols = [f"after_{m}" for m in study.MONTH_WINDOWS]

    med_before = shown["before_12m"].median()
    med_after = shown["after_12m"].median()
    with st.container(horizontal=True):
        st.metric("Companies", f"{len(shown):,}", border=True)
        st.metric("Median articles in first 30d",
                  f"{shown['articles_30d'].median():.0f}", border=True)
        st.metric("Median 12m BEFORE first article",
                  "—" if pd.isna(med_before) else f"{med_before:+.1f}%",
                  border=True,
                  help="Positive means the Fool tends to start covering a stock "
                       "that has already run.")
        st.metric("Median 12m AFTER first article",
                  "—" if pd.isna(med_after) else f"{med_after:+.1f}%",
                  border=True)

    if pd.notna(med_before) and pd.notna(med_after):
        if med_before > 5 and med_before > med_after:
            st.info(
                f"**Coverage follows the move.** These companies were already "
                f"{med_before:+.1f}% vs SPY in the year *before* the first "
                f"article, and {med_after:+.1f}% in the year after. On this "
                "sample a first article is closer to a lagging indicator than "
                "an entry signal.")
        elif med_after > med_before:
            st.info(
                f"Here the year *after* first coverage ({med_after:+.1f}%) beat "
                f"the year *before* ({med_before:+.1f}%) vs SPY. Suggestive, "
                "not conclusive — survivorship flatters the after column and "
                "not the before one.")

    display = [
        "ticker", "name", "first_covered", "days_to_2nd", "articles_30d",
        "articles_90d", "articles", *before_cols, *after_cols,
        "since_first", "latest_article", "censored",
    ]
    fmt = {
        "ticker": st.column_config.TextColumn("Ticker", pinned=True),
        "name": st.column_config.TextColumn("Company", width="medium"),
        "first_covered": st.column_config.DateColumn("First covered"),
        "days_to_2nd": st.column_config.NumberColumn(
            "Days to 2nd", help="Gap to the second article — the tightest "
                                "measure of how fast coverage ramped."),
        "articles_30d": st.column_config.NumberColumn("In 1st 30d"),
        "articles_90d": st.column_config.NumberColumn("In 1st 90d"),
        "articles": st.column_config.NumberColumn("Total"),
        "since_first": st.column_config.NumberColumn(
            "Since first", format="%+.1f%%"),
        "latest_article": st.column_config.DateColumn("Latest"),
        "censored": st.column_config.CheckboxColumn(
            "Censored", help="First article sits at the edge of your data."),
    }
    for m in study.MONTH_WINDOWS:
        fmt[f"before_{m}"] = st.column_config.NumberColumn(
            f"-{m}", format="%+.1f%%",
            help=f"Return over the {m} ending at the first article.")
        fmt[f"after_{m}"] = st.column_config.NumberColumn(
            f"+{m}", format="%+.1f%%",
            help=f"Return over the {m} starting at the first article.")

    with st.container(border=True):
        st.subheader("Coverage start, ramp speed, and returns either side")
        event = st.dataframe(
            as_dates(shown, "first_covered", "latest_article")[display],
            hide_index=True, on_select="rerun", selection_mode="single-row",
            column_config=fmt,
        )
        picked = event.selection.rows
        if picked:
            row = shown.iloc[picked[0]]
            article_panel(row["ticker"], row["name"])

    with st.container(border=True):
        st.subheader("Before vs after, by window")
        st.caption("Median market-adjusted return across the companies above. "
                   "The grey bars are the run-up into first coverage.")
        summary = pd.DataFrame([
            {"window": lab, "phase": phase,
             "median": shown[f"{pre}_{lab}"].median()}
            for lab in study.MONTH_WINDOWS
            for phase, pre in (("Before first article", "before"),
                               ("After first article", "after"))
        ]).dropna(subset=["median"])
        if summary.empty:
            st.caption("No window has enough price history yet.")
        else:
            phases = ["Before first article", "After first article"]
            st.altair_chart(
                alt.Chart(summary).mark_bar(cornerRadiusEnd=3).encode(
                    x=alt.X("median:Q", title="Median adj. return (%)"),
                    y=alt.Y("window:N", title=None,
                            sort=list(study.MONTH_WINDOWS)),
                    yOffset=alt.YOffset("phase:N", sort=phases),
                    color=alt.Color("phase:N", sort=phases,
                                    scale=alt.Scale(domain=phases,
                                                    range=[NEUTRAL, BULL]),
                                    legend=alt.Legend(title=None,
                                                      orient="bottom")),
                    tooltip=["window:N", "phase:N",
                             alt.Tooltip("median:Q", format="+.2f")],
                ).properties(height=240)
            )

    with st.expander("How to read this"):
        for n in notes:
            st.markdown(f"- {n}")
        st.markdown(
            "- **Days to 2nd** and **In 1st 30d** are the ramp-speed measures: "
            "a company written about five times in its first month is being "
            "pushed, not merely noted.\n"
            "- Survivorship applies to the **after** columns but not the "
            "**before** ones, which tilts the comparison in favour of 'after'. "
            "A company that delisted has no price history at all, so it is "
            "simply absent."
        )


@st.cache_data(ttl="30m", show_spinner="Scoring author calls…")
def run_author_scores(horizon: int, primary: bool, censor_days: int,
                      min_calls: int):
    df, base, notes = study.author_scores(
        get_conn(), horizon=horizon, primary_only=primary,
        censor_days=censor_days, min_calls=min_calls)
    return df, base, notes


def view_author_accuracy() -> None:
    st.caption(
        "A **call** is a headline that takes a direction — bullish (`buy`, "
        "`buy (question)`) or bearish (`sell`, `caution`). It is **right** when "
        "the stock moves that way over the holding period, after subtracting "
        "SPY. Articles that state no view are not counted as wrong; they are not "
        "calls at all."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        horizon = st.selectbox("Holding period (trading days)", study.HORIZONS,
                               index=2, key="auth_h")
    with c2:
        min_calls = st.number_input(
            "Minimum calls to rank", min_value=1, max_value=200,
            value=study.MIN_CALLS, step=5,
            help="Below ~20 calls a hit rate is indistinguishable from a coin "
                 "flip. Lower this only to explore, never to draw conclusions.")
    with c3:
        censor = st.number_input("Censoring guard (days)", min_value=0,
                                 max_value=365, value=0, step=15, key="auth_c",
                                 help="Author scoring does not depend on first "
                                      "coverage, so 0 is usually right here.")

    df, base, notes = run_author_scores(int(horizon), primary_only, int(censor),
                                        int(min_calls))

    if df.empty:
        st.warning("No directional call has enough price history after it yet. "
                   "Try a shorter holding period, or finish the backfill.")
        for n in notes:
            st.caption(f"• {n}")
        return

    with st.container(horizontal=True):
        st.metric("Calls measured", f"{base['calls']:,}", border=True)
        st.metric("Base rate", f"{base['hit_rate']:.0f}%", border=True,
                  help="Share of ALL calls that went the right way. This is the "
                       "bar an author has to clear — not 50%.")
        st.metric("Bullish / bearish",
                  f"{base['bullish_calls']:,} / {base['bearish_calls']:,}",
                  border=True)
        st.metric("Authors with calls", f"{len(df):,}", border=True)

    qualified = df[df["enough_data"]]
    if qualified.empty:
        st.error(
            f"**No author has {min_calls}+ calls yet, so there is no leaderboard "
            f"to show.** The table below is sorted by call count, deliberately "
            f"not by hit rate: with {len(df)} authors averaging "
            f"{base['calls'] / len(df):.1f} calls each, several sit at 100% off a "
            "single call. That is arithmetic, not skill. Finish the backfill — "
            "three years should give the prolific authors a few hundred calls "
            "each."
        )
        table = df.sort_values("calls", ascending=False)
    else:
        real = qualified[qualified["beats_base"]]
        if real.empty:
            st.info(
                f"**{len(qualified)} author(s) have {min_calls}+ calls, and none "
                f"of them beat the {base['hit_rate']:.0f}% base rate by more "
                "than their error bar.** That is the honest result: no evidence "
                "any individual is better than the crowd.")
        else:
            st.success(
                f"**{len(real)} of {len(qualified)} authors with {min_calls}+ "
                f"calls clear the {base['hit_rate']:.0f}% base rate even at the "
                "bottom of their confidence interval.** Still read this with "
                f"suspicion: ranking {len(df)} people guarantees a lucky leader.")
        table = qualified

    with st.container(border=True):
        st.subheader("Call accuracy by author")
        st.dataframe(
            table[["author", "calls", "hits", "hit_rate", "ci_low", "ci_high",
                   "vs_base", "mean_call_return", "bullish_share", "tickers",
                   "enough_data"]],
            hide_index=True,
            column_config={
                "author": st.column_config.TextColumn("Author", pinned=True),
                "calls": st.column_config.NumberColumn("Calls"),
                "hits": st.column_config.NumberColumn("Right"),
                "hit_rate": st.column_config.NumberColumn("Hit rate", format="%.0f%%"),
                "ci_low": st.column_config.NumberColumn(
                    "CI low", format="%.0f%%",
                    help="Wilson 95% interval. If this is below the base rate, "
                         "the author is not distinguishable from the crowd."),
                "ci_high": st.column_config.NumberColumn("CI high", format="%.0f%%"),
                "vs_base": st.column_config.NumberColumn(
                    "vs base", format="%+.0f pts"),
                "mean_call_return": st.column_config.NumberColumn(
                    "Mean call return", format="%+.2f",
                    help="Market-adjusted points earned per call, signed so a "
                         "correct bearish call scores positive."),
                "bullish_share": st.column_config.NumberColumn(
                    "Bullish", format="%.0f%%"),
                "tickers": st.column_config.NumberColumn("Tickers"),
                "enough_data": st.column_config.CheckboxColumn(
                    f"{min_calls}+ calls"),
            },
        )

    if not qualified.empty:
        with st.container(border=True):
            st.subheader("Hit rate vs. the base rate")
            st.caption("Bars show each author's hit rate minus the base rate. "
                       "Only authors clearing the minimum call count appear.")
            chart = (
                alt.Chart(qualified)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=alt.X("vs_base:Q", title="Hit rate minus base rate (pts)"),
                    y=alt.Y("author:N", title=None,
                            sort=alt.EncodingSortField("vs_base", order="descending")),
                    color=alt.condition(alt.datum.vs_base >= 0,
                                        alt.value(BULL), alt.value(BEAR)),
                    tooltip=["author:N", "calls:Q", "hit_rate:Q", "vs_base:Q"],
                )
                .properties(height=max(200, 22 * len(qualified)))
            )
            st.altair_chart(chart)

    with st.expander("Why this is harder than it looks"):
        for n in notes:
            st.markdown(f"- {n}")
        st.markdown(
            "- Bullish calls outnumber bearish ones heavily, so in a rising "
            "market raw accuracy would mostly measure the market. Every return "
            "here is SPY-adjusted for that reason.\n"
            "- An author does not choose which stocks to cover at random, and "
            "assignment may not be theirs at all. This measures the record of "
            "the calls under their byline, not their ability.\n"
            "- Survivorship still applies: delisted tickers have no prices, so "
            "calls that went to zero are missing entirely."
        )


def view_signals() -> None:
    st.caption("Two things the raw tables bury: when the Fool **changes its "
               "mind** about a company, and when it **suddenly starts writing** "
               "about one a lot.")

    ref = q("SELECT MAX(published_day) d FROM articles").at[0, "d"]

    with st.container(border=True):
        st.subheader("Stance flips")
        st.caption(
            "Consecutive directional articles on the same company where the "
            "direction reversed. Bearish headlines are ~2% of everything the "
            "Fool publishes, so a flip to caution is the rarest signal here — "
            "and the one most worth reading."
        )
        flips = q(
            f"""
            WITH d AS (
                SELECT c.ticker, c.published_day, a.stance, a.stance_score,
                       a.title, a.path, a.author,
                       LAG(a.stance_score) OVER w AS prev_score,
                       LAG(a.published_day) OVER w AS prev_day,
                       LAG(a.title)         OVER w AS prev_title,
                       LAG(a.path)          OVER w AS prev_path
                FROM coverage c
                JOIN articles a ON a.path = c.path
                WHERE a.stance_score <> 0{COV_FILTER}
                WINDOW w AS (PARTITION BY c.ticker
                             ORDER BY c.published_day, a.path)
            )
            SELECT ticker, prev_day, prev_title, prev_score, prev_path,
                   published_day, title, stance_score, path, author
            FROM d
            WHERE prev_score IS NOT NULL
              AND ((stance_score > 0) <> (prev_score > 0))
              AND published_day BETWEEN ? AND ?
            ORDER BY published_day DESC
            LIMIT 200
            """, DAYS)
        if flips.empty:
            st.info("No stance flips in this window. Expected while the archive "
                    "is thin — flips need two directional articles on the same "
                    "company, and only ~20% of articles take a direction.")
        else:
            flips["direction"] = np.where(
                flips["stance_score"] > 0, "turned bullish", "turned cautious")
            flips["days_between"] = (
                pd.to_datetime(flips["published_day"])
                - pd.to_datetime(flips["prev_day"])).dt.days
            flips["url"] = BASE + flips["path"]
            flips["prev_url"] = BASE + flips["prev_path"]
            st.dataframe(
                as_dates(flips, "published_day", "prev_day")[
                    ["ticker", "direction", "prev_day", "prev_title", "prev_url",
                     "published_day", "title", "url", "days_between", "author"]],
                hide_index=True,
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", pinned=True),
                    "direction": st.column_config.TextColumn("Flip"),
                    "prev_day": st.column_config.DateColumn("Was"),
                    "prev_title": st.column_config.TextColumn(
                        "Previous headline", width="medium"),
                    "prev_url": st.column_config.LinkColumn(
                        "Prev", display_text="read"),
                    "published_day": st.column_config.DateColumn("Now"),
                    "title": st.column_config.TextColumn(
                        "New headline", width="medium"),
                    "url": st.column_config.LinkColumn("New", display_text="read"),
                    "days_between": st.column_config.NumberColumn("Days apart"),
                    "author": st.column_config.TextColumn("New author"),
                },
            )

    with st.container(border=True):
        st.subheader("Coverage spikes")
        st.caption(
            "Articles in the last 30 days against the average 30 days over the "
            "90 before that. A company the Fool has suddenly started writing "
            "about — which may lead a move, or simply follow one."
        )
        spikes = q(
            f"""
            SELECT c.ticker,
                   MAX(t.name) AS name,
                   SUM(c.published_day >  date(?, '-30 days')) AS last_30d,
                   SUM(c.published_day <= date(?, '-30 days')
                       AND c.published_day > date(?, '-120 days')) / 3.0
                       AS prior_avg_30d,
                   MAX(c.published_day) AS latest
            FROM coverage c
            LEFT JOIN tickers t ON t.ticker = c.ticker
            WHERE 1=1{COV_FILTER}
            GROUP BY c.ticker
            HAVING last_30d >= 3
            ORDER BY last_30d DESC
            LIMIT 200
            """, (ref, ref, ref))
        if spikes.empty:
            st.info("Not enough history for a 30-day-vs-prior comparison yet. "
                    "This needs a few months of archive.")
        else:
            # A brand-new company has no baseline; that is newly-covered, not a
            # spike, so label it rather than dividing by zero.
            spikes["multiple"] = np.where(
                spikes["prior_avg_30d"] > 0,
                spikes["last_30d"] / spikes["prior_avg_30d"].replace(0, np.nan),
                np.nan)
            spikes["status"] = np.where(
                spikes["prior_avg_30d"] > 0, "accelerating", "newly covered")
            spikes = spikes.sort_values(
                ["multiple", "last_30d"], ascending=[False, False])
            st.dataframe(
                as_dates(spikes, "latest")[
                    ["ticker", "name", "status", "last_30d", "prior_avg_30d",
                     "multiple", "latest"]],
                hide_index=True,
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", pinned=True),
                    "name": st.column_config.TextColumn("Company", width="medium"),
                    "status": st.column_config.TextColumn("Status"),
                    "last_30d": st.column_config.NumberColumn("Last 30d"),
                    "prior_avg_30d": st.column_config.NumberColumn(
                        "Prior 30d avg", format="%.1f"),
                    "multiple": st.column_config.NumberColumn(
                        "Multiple", format="%.1fx",
                        help="Blank for companies with no prior coverage to "
                             "compare against."),
                    "latest": st.column_config.DateColumn("Latest article"),
                },
            )


def view_authors() -> None:
    rows = q(
        """
        SELECT a.author,
               COUNT(*) AS articles,
               ROUND(AVG(a.stance_score), 2) AS avg_stance,
               MIN(a.published_day) AS first_seen,
               MAX(a.published_day) AS last_seen,
               COUNT(DISTINCT c.ticker) AS tickers
        FROM articles a
        LEFT JOIN coverage c ON c.path = a.path AND c.is_primary = 1
        WHERE a.published_day BETWEEN ? AND ? AND a.author <> ''
        GROUP BY a.author
        ORDER BY articles DESC
        """, DAYS)
    if rows.empty:
        st.info("No authors in this window.")
        return
    with st.container(horizontal=True):
        st.metric("Authors", f"{len(rows):,}", border=True)
        st.metric("Median articles each", f"{rows.articles.median():.0f}", border=True)
        st.metric("Most prolific", rows.author.iloc[0], border=True)
    st.dataframe(
        as_dates(rows, "first_seen", "last_seen"), hide_index=True,
        column_config={
            "author": st.column_config.TextColumn("Author", pinned=True),
            "articles": st.column_config.NumberColumn("Articles"),
            "avg_stance": st.column_config.NumberColumn(
                "Avg stance", format="%.2f",
                help="Mean headline stance, -2 (sell) to +2 (buy)."),
            "tickers": st.column_config.NumberColumn("Primary tickers"),
            "first_seen": st.column_config.DateColumn("First article"),
            "last_seen": st.column_config.DateColumn("Latest article"),
        },
    )


@st.cache_data(ttl="30m", show_spinner="Running the event study…")
def run_study(primary: bool, censor_days: int, min_articles: int):
    res = study.build_events(get_conn(), primary_only=primary,
                             censor_days=censor_days,
                             min_articles_per_ticker=min_articles)
    return (res.events, res.by_ordinal, res.paired, res.by_intensity,
            res.notes, study.verdict(res.paired, res.by_ordinal))


def _return_chart(df: pd.DataFrame, key: str, stat: str, horizon: int,
                  order: list[str]):
    """Market-adjusted return by bucket: diverging about zero, so bars take sign."""
    col = f"{stat}_{horizon}d"
    d = df[[key, col, f"n_{horizon}d"]].dropna(subset=[col])
    if d.empty:
        return None
    return (
        alt.Chart(d)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(f"{key}:N", sort=order, title=None,
                    axis=alt.Axis(labelAngle=-30)),
            y=alt.Y(f"{col}:Q", title=f"{stat.title()} adj. return, {horizon}d (pts)"),
            color=alt.condition(alt.datum[col] >= 0,
                                alt.value(BULL), alt.value(BEAR)),
            tooltip=[alt.Tooltip(f"{key}:N", title="Bucket"),
                     alt.Tooltip(f"{col}:Q", title=f"{stat} (pts)", format="+.2f"),
                     alt.Tooltip(f"n_{horizon}d:Q", title="Events")],
        )
        .properties(height=260)
    )


def view_study() -> None:
    st.caption(
        "For every primary article, the return from buying at that day's close, "
        "**minus SPY over the same trading days** — grouped by whether it was the "
        "first article about that ticker or a later one."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        horizon = st.selectbox("Holding period (trading days)", study.HORIZONS,
                               index=2)
    with c2:
        stat = st.radio("Statistic", ["median", "mean"], horizontal=True,
                        help="Median first: a handful of huge winners can carry "
                             "a mean on its own.")
    with c3:
        censor = st.number_input(
            "Censoring guard (days)", min_value=0, max_value=365, value=45, step=15,
            help="Drop tickers whose earliest article is within this many days of "
                 "the start of your data — their '1st article' is just where the "
                 "backfill begins.")

    events, by_ordinal, paired, by_intensity, notes, the_verdict = run_study(
        primary_only, int(censor), 2)

    if events.empty:
        st.warning(the_verdict)
        for n in notes:
            st.caption(f"• {n}")
        return

    st.info(f"**{the_verdict}**")

    with st.container(horizontal=True):
        st.metric("Events measured", f"{len(events):,}", border=True)
        st.metric("Tickers", f"{events['ticker'].nunique():,}", border=True)
        st.metric("Paired 1st vs 2nd",
                  f"{0 if paired.empty else int(paired['tickers'].max()):,}",
                  border=True)

    with st.container(border=True):
        st.subheader("Return by how early the article was")
        chart = _return_chart(by_ordinal, "bucket", stat, horizon,
                              study.BUCKET_ORDER)
        if chart is None:
            st.caption(f"No article yet has {horizon} trading days of price "
                       "history after it. Pick a shorter holding period.")
        else:
            st.altair_chart(chart)
        show = ["bucket", "articles", "tickers", f"n_{horizon}d",
                f"mean_{horizon}d", f"median_{horizon}d", f"win_{horizon}d"]
        st.dataframe(
            by_ordinal[show], hide_index=True,
            column_config={
                "bucket": st.column_config.TextColumn("Which article"),
                "articles": st.column_config.NumberColumn("Articles"),
                "tickers": st.column_config.NumberColumn("Tickers"),
                f"n_{horizon}d": st.column_config.NumberColumn("Measured"),
                f"mean_{horizon}d": st.column_config.NumberColumn(
                    "Mean (pts)", format="%+.2f"),
                f"median_{horizon}d": st.column_config.NumberColumn(
                    "Median (pts)", format="%+.2f"),
                f"win_{horizon}d": st.column_config.NumberColumn(
                    "Beat SPY", format="%.0f%%"),
            },
        )

    if not paired.empty:
        with st.container(border=True):
            st.subheader("1st vs 2nd article, same ticker")
            st.caption("Comparing two entries in the *same* stock removes every "
                       "difference between stocks that the bucket averages above "
                       "cannot. This is the sharpest version of the question.")
            st.dataframe(
                paired, hide_index=True,
                column_config={
                    "horizon": st.column_config.TextColumn("Horizon"),
                    "tickers": st.column_config.NumberColumn("Tickers"),
                    "first_mean": st.column_config.NumberColumn("1st mean", format="%+.2f"),
                    "second_mean": st.column_config.NumberColumn("2nd mean", format="%+.2f"),
                    "first_median": st.column_config.NumberColumn("1st median", format="%+.2f"),
                    "second_median": st.column_config.NumberColumn("2nd median", format="%+.2f"),
                    "mean_diff": st.column_config.NumberColumn("Mean gap", format="%+.2f"),
                    "median_diff": st.column_config.NumberColumn("Median gap", format="%+.2f"),
                    "first_wins_pct": st.column_config.NumberColumn(
                        "1st wins", format="%.0f%%"),
                },
            )

    if not by_intensity.empty:
        with st.container(border=True):
            st.subheader("Does coverage frequency matter?")
            st.caption("Tickers grouped by how many articles they got in total. "
                       "Direction of causation is unknowable here — heavy "
                       "coverage may follow a good run rather than precede one.")
            chart = _return_chart(by_intensity, "intensity", stat, horizon,
                                  ["2 articles", "3-5", "6-10", "11-25", "26+"])
            if chart is not None:
                st.altair_chart(chart)

    with st.expander("What this does and does not show"):
        for n in notes:
            st.markdown(f"- {n}")
        st.markdown(
            "- Returns ignore costs, slippage, taxes and position sizing.\n"
            "- This is an observational study of past coverage, not a backtest "
            "of a tradeable strategy, and not advice."
        )


tabs = st.tabs(["Overview", "Ticker history", "First coverage", "Signals",
                "Early vs late", "Author accuracy", "Authors"])
with tabs[0]:
    view_overview()
with tabs[1]:
    view_ticker()
with tabs[2]:
    view_first_coverage()
with tabs[3]:
    view_signals()
with tabs[4]:
    view_study()
with tabs[5]:
    view_author_accuracy()
with tabs[6]:
    view_authors()
