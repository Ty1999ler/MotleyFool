"""fool.com discovery and extraction.

Discovery uses the sitemaps fool.com publishes in its own robots.txt:

  * ``/news-sitemap.xml``   - the last ~3 days, with titles and tickers inline.
  * ``/sitemap/``           - an index of month archives, back to 1995/03.
  * ``/sitemap/YYYY/MM``    - every URL published that month (~3,000 articles).

Extraction fetches each article but stops reading at ``</head>``. Everything we
need lives in the first ~17 KB of a ~300 KB page, including the two meta tags
that matter most:

    <meta name="tickers" content="NVDA,AVGO,GOOGL,AMZN,META,AMD,GOOG">
    <meta name="primary_tickers" content="NVDA">

That second tag is fool.com telling us which company the article is actually
about, so "primary vs incidental coverage" is read, not guessed.
"""

from __future__ import annotations

import html as html_mod
import logging
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from .config import BROWSER_UA, Config
from .db import utc_now
from .stance import classify

log = logging.getLogger(__name__)

SITEMAP_INDEX = "https://www.fool.com/sitemap/"
NEWS_SITEMAP = "https://www.fool.com/news-sitemap.xml"
MONTH_SITEMAP = "https://www.fool.com/sitemap/{month}"   # month = YYYY/MM
BASE = "https://www.fool.com"

LOC_RE = re.compile(r"<loc>([^<]+)</loc>")
MONTH_PATH_RE = re.compile(r"/sitemap/(\d{4})/(\d{2})")
DATED_PATH_RE = re.compile(r"^/([a-z0-9\-]+)/(\d{4})/(\d{2})/(\d{2})/[^/]+/?$")
NEWS_ENTRY_RE = re.compile(r"<url>(.*?)</url>", re.S)
NEWS_TITLE_RE = re.compile(r"<news:title>(.*?)</news:title>", re.S)
NEWS_TICKERS_RE = re.compile(r"<news:stock_tickers>(.*?)</news:stock_tickers>", re.S)

META_RE = re.compile(r'<meta\s+name="([a-z_]+)"\s+content="([^"]*)"', re.I)
OG_TITLE_RE = re.compile(r'<meta\s+property="og:title"\s+content="([^"]*)"', re.I)
TITLE_RE = re.compile(r"<title>([^<]*)</title>", re.I)
HEAD_END_RE = re.compile(rb"</head>", re.I)
TITLE_SUFFIX_RE = re.compile(r"\s*\|\s*The Motley Fool\s*$", re.I)


#: Longest the crawler holds the SQLite write lock between commits. Short, so
#: other writers (history backfill, outcome scoring, restance) never time out.
COMMIT_INTERVAL_SECONDS = 5.0


# --- politeness --------------------------------------------------------------

class RateLimited(Exception):
    """fool.com returned 429. Carries its Retry-After, in seconds, when given."""

    def __init__(self, retry_after: float | None):
        super().__init__(f"429 Too Many Requests (Retry-After={retry_after})")
        self.retry_after = retry_after


class RateLimiter:
    """Global token-bucket cap shared by every worker thread.

    Adaptive, because a fixed rate that looks safe can still trip the origin's
    limiter: every 429 halves the rate, and a long clean run creeps it back
    toward the configured base. Slowing down beats getting the IP blocked -
    6 req/s once earned a 23-hour cooldown.
    """

    MIN_RATE = 0.25

    def __init__(self, per_second: float):
        self.base_rate = per_second if per_second > 0 else 0.0
        self.rate = self.base_rate
        self._lock = threading.Lock()
        self._next = 0.0
        self._ok_since_penalty = 0

    @property
    def interval(self) -> float:
        return 1.0 / self.rate if self.rate > 0 else 0.0

    def wait(self) -> None:
        with self._lock:
            interval = self.interval
            if not interval:
                return
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)

    def penalize(self) -> float:
        """Halve the rate after a 429. Returns the new rate."""
        with self._lock:
            self._ok_since_penalty = 0
            if self.rate:
                self.rate = max(self.MIN_RATE, self.rate / 2)
            return self.rate

    def reward(self) -> None:
        """Creep back toward the configured rate after 200 clean fetches."""
        with self._lock:
            self._ok_since_penalty += 1
            if self._ok_since_penalty >= 200 and self.rate < self.base_rate:
                self.rate = min(self.base_rate, self.rate * 1.5)
                self._ok_since_penalty = 0


# --- parsed article ----------------------------------------------------------

@dataclass
class Article:
    path: str
    title: str = ""
    author: str = ""
    description: str = ""
    section: str = ""
    published_day: str = ""
    published_utc: int = 0
    tickers: list[str] = field(default_factory=list)
    primary: set[str] = field(default_factory=set)


def _norm_ticker(raw: str, in_universe) -> str | None:
    """Clean a raw ticker string; restore the dot fool.com drops from class shares."""
    tick = raw.strip().upper().replace("-", ".")
    if not tick:
        return None
    if not re.fullmatch(r"[A-Z][A-Z.]{0,6}", tick):
        return None
    if not 1 <= len(tick.split(".")[0]) <= 5:
        return None
    # fool.com renders BRK.A as BRKA; put the dot back when that resolves.
    if "." not in tick and len(tick) >= 3 and not in_universe(tick):
        dotted = f"{tick[:-1]}.{tick[-1]}"
        if in_universe(dotted):
            return dotted
    return tick


def path_of(url: str) -> str:
    return url.replace(BASE, "").split("?")[0].split("#")[0]


def _day_from_path(path: str) -> tuple[str, str] | None:
    """Return (section, YYYY-MM-DD) for a dated article path, else None."""
    m = DATED_PATH_RE.match(path)
    if not m:
        return None
    section, y, mo, d = m.groups()
    return section, f"{y}-{mo}-{d}"


def parse_head(path: str, head_text: str, in_universe) -> Article:
    art = Article(path=path)
    meta = {}
    for m in META_RE.finditer(head_text):
        key = m.group(1).lower()
        # Fool emits <meta name="author"> twice; first one wins.
        meta.setdefault(key, html_mod.unescape(m.group(2)).strip())

    title = ""
    og = OG_TITLE_RE.search(head_text)
    if og:
        title = html_mod.unescape(og.group(1))
    elif (tm := TITLE_RE.search(head_text)):
        title = html_mod.unescape(tm.group(1))
    art.title = TITLE_SUFFIX_RE.sub("", " ".join(title.split()))

    art.author = meta.get("author", "")
    art.description = meta.get("description", "")

    sec_day = _day_from_path(path)
    if sec_day:
        art.section, art.published_day = sec_day
        art.published_utc = int(
            datetime.strptime(art.published_day, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc).timestamp()
        )

    seen: list[str] = []
    for raw in (meta.get("tickers") or "").split(","):
        tick = _norm_ticker(raw, in_universe)
        if tick and tick not in seen:
            seen.append(tick)
    art.tickers = seen

    for raw in (meta.get("primary_tickers") or "").split(","):
        tick = _norm_ticker(raw, in_universe)
        if tick:
            art.primary.add(tick)
            if tick not in art.tickers:
                art.tickers.append(tick)
    return art


def _retry_after(resp: requests.Response) -> float | None:
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def fetch_head(session: requests.Session, path: str, cfg: Config,
               limiter: RateLimiter) -> str:
    """Download only as far as </head>. Saves ~95% of the transfer.

    A 429 raises RateLimited rather than burning retries: the caller has to
    decide whether to slow down or stop, and hammering a limiter that is
    already complaining is what turns a throttle into an IP ban.
    """
    last_err: Exception | None = None
    for attempt in range(cfg.retries):
        limiter.wait()
        try:
            resp = session.get(f"{BASE}{path}", timeout=30, stream=True,
                               headers={"User-Agent": BROWSER_UA})
            if resp.status_code == 429:
                wait = _retry_after(resp)
                resp.close()
                raise RateLimited(wait)
            if resp.status_code in (404, 410):
                resp.close()
                raise FileNotFoundError(f"{resp.status_code} for {path}")
            resp.raise_for_status()
            buf = bytearray()
            for chunk in resp.iter_content(8192):
                buf.extend(chunk)
                if HEAD_END_RE.search(buf) or len(buf) >= cfg.head_bytes:
                    break
            resp.close()
            limiter.reward()
            return buf.decode("utf-8", errors="replace")
        except (FileNotFoundError, RateLimited):
            raise
        except requests.RequestException as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise last_err or RuntimeError("fetch failed")


# --- discovery ---------------------------------------------------------------

def _queue_paths(conn: sqlite3.Connection, cfg: Config,
                 urls: list[str]) -> int:
    """Insert dated article paths into the crawl queue. Returns rows added."""
    rows = []
    for url in urls:
        path = path_of(url)
        sec_day = _day_from_path(path)
        if not sec_day:
            continue
        section, day = sec_day
        state = "pending" if cfg.section_ok(path) else "skipped"
        rows.append((path, section, day, utc_now(), state))
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO crawl_queue (path, section, published_day, "
        "discovered_utc, state) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn.total_changes - before


def available_months(session: requests.Session) -> list[str]:
    """Every YYYY/MM the archive index offers, oldest first."""
    resp = session.get(SITEMAP_INDEX, timeout=60, headers={"User-Agent": BROWSER_UA})
    resp.raise_for_status()
    months = []
    for loc in LOC_RE.findall(resp.text):
        m = MONTH_PATH_RE.search(loc)
        if m:
            months.append(f"{m.group(1)}/{m.group(2)}")
    return sorted(set(months))


def enumerate_month(conn: sqlite3.Connection, cfg: Config,
                    session: requests.Session, month: str,
                    limiter: RateLimiter) -> int:
    """Read one month archive sitemap into the crawl queue."""
    limiter.wait()
    resp = session.get(MONTH_SITEMAP.format(month=month), timeout=90,
                       headers={"User-Agent": BROWSER_UA})
    if resp.status_code == 404:
        log.warning("No archive sitemap for %s", month)
        return 0
    resp.raise_for_status()
    urls = LOC_RE.findall(resp.text)
    added = _queue_paths(conn, cfg, urls)
    conn.execute(
        "INSERT OR REPLACE INTO archive_months (month, urls_found, enumerated_utc) "
        "VALUES (?, ?, ?)",
        (month, len(urls), utc_now()),
    )
    conn.commit()
    log.info("Archive %s: %d urls, %d newly queued", month, len(urls), added)
    return added


def enumerate_recent(conn: sqlite3.Connection, cfg: Config,
                     session: requests.Session, limiter: RateLimiter) -> int:
    """Read the news sitemap (last ~3 days) into the crawl queue."""
    limiter.wait()
    resp = session.get(NEWS_SITEMAP, timeout=60, headers={"User-Agent": BROWSER_UA})
    resp.raise_for_status()
    added = _queue_paths(conn, cfg, LOC_RE.findall(resp.text))
    log.info("News sitemap: %d newly queued", added)
    return added


# --- extraction --------------------------------------------------------------

def _store(conn: sqlite3.Connection, art: Article) -> tuple[int, int]:
    """Write one article + its coverage rows. Returns (articles, coverage)."""
    label, score = classify(art.title)
    conn.execute(
        "INSERT OR REPLACE INTO articles (path, title, author, published_day, "
        "published_utc, section, stance, stance_score, ticker_count, fetched_utc) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (art.path, art.title, art.author, art.published_day, art.published_utc,
         art.section, label, score, len(art.tickers), utc_now()),
    )
    cov = 0
    for tick in art.tickers:
        is_primary = 1 if tick in art.primary else 0
        verified = conn.execute(
            "SELECT 1 FROM universe WHERE ticker = ?", (tick,)
        ).fetchone() is not None
        conn.execute(
            "INSERT OR REPLACE INTO coverage (ticker, path, published_day, "
            "is_primary, source, verified) VALUES (?, ?, ?, ?, ?, ?)",
            (tick, art.path, art.published_day, is_primary,
             "primary_meta" if is_primary else "meta", int(verified)),
        )
        cov += 1
        conn.execute(
            "INSERT INTO tickers (ticker, first_seen_day) VALUES (?, ?) "
            "ON CONFLICT(ticker) DO UPDATE SET "
            "first_seen_day = MIN(first_seen_day, excluded.first_seen_day)",
            (tick, art.published_day),
        )
    return 1, cov


def crawl_pending(conn: sqlite3.Connection, cfg: Config, limit: int | None = None,
                  progress_every: int = 250) -> dict:
    """Fetch every pending queue row. Resumable: re-running picks up where it stopped.

    Workers only do network + parsing; all SQLite writes happen on this thread,
    which keeps the single connection safe without locking games.
    """
    universe = {r["ticker"] for r in conn.execute("SELECT ticker FROM universe")}
    in_universe = universe.__contains__

    sql = ("SELECT path FROM crawl_queue WHERE state = 'pending' "
           "ORDER BY published_day DESC")
    if limit:
        sql += f" LIMIT {int(limit)}"
    paths = [r["path"] for r in conn.execute(sql)]
    if not paths:
        log.info("Nothing pending to crawl")
        return {"articles": 0, "coverage": 0, "failed": 0, "gone": 0}

    limiter = RateLimiter(cfg.requests_per_second)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=cfg.workers * 2,
                                            pool_maxsize=cfg.workers * 2)
    session.mount("https://", adapter)

    totals = {"articles": 0, "coverage": 0, "failed": 0, "gone": 0,
              "deferred": 0, "throttled": 0}
    started = time.monotonic()
    # Circuit breaker: once the origin is refusing, every further request makes
    # the cooldown worse. Trip out and leave the rest of the queue pending.
    abort = threading.Event()
    abort_reason: list[str] = []
    # How long the origin actually asked us to wait, so the caller can stand
    # down for that long rather than guessing.
    cooldowns: list[float] = []

    def work(path: str):
        if abort.is_set():
            return path, None, ("deferred", "aborted before fetch")
        try:
            head = fetch_head(session, path, cfg, limiter)
            return path, parse_head(path, head, in_universe), None
        except FileNotFoundError as e:
            return path, None, ("gone", str(e))
        except RateLimited as e:
            new_rate = limiter.penalize()
            wait = e.retry_after
            if wait is not None and wait > cfg.max_backoff_seconds:
                if not abort.is_set():
                    cooldowns.append(wait)
                    abort_reason.append(
                        f"origin asked for a {wait / 60:.0f} min cooldown "
                        f"(Retry-After={wait:.0f}s)")
                    abort.set()
                return path, None, ("deferred", f"429 Retry-After={wait:.0f}s")
            # Short, survivable throttle: sleep it off and let the row retry.
            time.sleep(min(wait or 30.0, cfg.max_backoff_seconds))
            log.warning("429 — rate now %.2f req/s", new_rate)
            return path, None, ("throttled", str(e))
        except Exception as e:                      # noqa: BLE001 - logged and queued
            return path, None, ("failed", f"{type(e).__name__}: {e}")

    last_commit = time.monotonic()
    log.info("Crawling %d articles at %.1f req/s across %d workers",
             len(paths), cfg.requests_per_second, cfg.workers)
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        for i, (path, art, err) in enumerate(pool.map(work, paths), 1):
            if err:
                kind, msg = err
                totals[kind] += 1
                if kind in ("deferred", "throttled"):
                    # Not the url's fault — leave it pending for the next run.
                    conn.execute(
                        "UPDATE crawl_queue SET last_error = ? WHERE path = ?",
                        (msg[:300], path))
                else:
                    conn.execute(
                        "UPDATE crawl_queue SET state = ?, attempts = attempts + 1, "
                        "last_error = ? WHERE path = ?",
                        ("skipped" if kind == "gone" else "failed", msg[:300], path))
            else:
                a, c = _store(conn, art)
                totals["articles"] += a
                totals["coverage"] += c
                conn.execute(
                    "UPDATE crawl_queue SET state = 'done', attempts = attempts + 1 "
                    "WHERE path = ?", (path,))
            # Commit by elapsed time as well as by count. Fifty articles was a
            # dozen seconds at 4 req/s but two minutes at 0.4 req/s, and holding
            # the write lock that long made every other writer — history
            # backfill, outcome scoring, restance — hit its timeout and fail.
            if i % 50 == 0 or time.monotonic() - last_commit >= COMMIT_INTERVAL_SECONDS:
                conn.commit()
                last_commit = time.monotonic()
            if i % progress_every == 0 and not abort.is_set():
                rate = i / max(time.monotonic() - started, 0.001)
                eta = (len(paths) - i) / max(rate, 0.001) / 60
                log.info("  %d/%d (%.1f/s, ~%.0f min left) articles=%d coverage=%d "
                         "failed=%d gone=%d throttled=%d", i, len(paths), rate, eta,
                         totals["articles"], totals["coverage"],
                         totals["failed"], totals["gone"], totals["throttled"])
    conn.commit()
    if abort_reason:
        log.error("Crawl stopped early: %s. %d urls left pending — rerun once the "
                  "cooldown expires.", abort_reason[0], totals["deferred"])
    log.info("Crawl finished in %.1f min: %s", (time.monotonic() - started) / 60, totals)
    totals["aborted"] = bool(abort_reason)
    totals["cooldown_seconds"] = max(cooldowns) if cooldowns else 0.0
    return totals


def requeue_failed(conn: sqlite3.Connection, max_attempts: int = 5) -> int:
    cur = conn.execute(
        "UPDATE crawl_queue SET state = 'pending' WHERE state = 'failed' "
        "AND attempts < ?", (max_attempts,))
    conn.commit()
    return cur.rowcount
