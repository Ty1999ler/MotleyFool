"""Long-running worker loop for container deployment.

Two jobs on one clock:

* **Daily catch-up** — once a day, read the news sitemap plus the current month's
  archive and crawl whatever is new (~250 articles, a couple of minutes).
* **Nightly backfill drip** — inside a quiet window, chip away at the archive
  queue in small batches.

The drip exists because the archive is ~100,000 articles and fool.com rate-limits:
one long greedy session earned a 23-hour 429 cooldown. Spread over a few nights at
a polite rate it finishes without ever annoying the origin. If a batch does trip a
long cooldown, the crawler reports it and the worker stands down until the next
night rather than retrying into a wall.

A plain loop rather than cron: no cron daemon in the image, logs go straight to
the container's stdout, and it reuses the same config.toml as the CLI.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import requests

from . import config as cfgmod
from . import fool, prices, study, universe
from .db import get_conn, utc_now

log = logging.getLogger("foolwatch.scheduler")


def _env_time(name: str, default: str) -> dtime:
    raw = os.environ.get(name, default).strip()
    hh, _, mm = raw.partition(":")
    return dtime(int(hh), int(mm or 0))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    def __init__(self, tz_name: str):
        self.tz = ZoneInfo(tz_name)
        self.daily_at = _env_time("FOOLWATCH_DAILY_AT", "18:30")
        self.backfill_start = _env_time("FOOLWATCH_BACKFILL_START", "01:00")
        self.backfill_end = _env_time("FOOLWATCH_BACKFILL_END", "06:00")
        self.batch = _env_int("FOOLWATCH_BACKFILL_BATCH", 1500)
        self.enabled = os.environ.get("FOOLWATCH_BACKFILL", "1") != "0"
        self.tick = _env_int("FOOLWATCH_TICK_SECONDS", 300)

    def in_backfill_window(self, now: datetime) -> bool:
        t = now.timetz().replace(tzinfo=None)
        if self.backfill_start <= self.backfill_end:
            return self.backfill_start <= t < self.backfill_end
        # Window wraps past midnight (e.g. 23:00 -> 05:00).
        return t >= self.backfill_start or t < self.backfill_end


def _last_daily_day(conn, tz) -> str | None:
    """Local date of the last completed daily run.

    Held in the database rather than in memory because Watchtower recreates the
    container on every push; an in-memory flag reset each time, so every deploy
    re-ran the whole daily job.
    """
    row = conn.execute(
        "SELECT MAX(finished_utc) AS t FROM runs WHERE kind = 'daily'").fetchone()
    if not row or not row["t"]:
        return None
    return datetime.fromtimestamp(row["t"], tz).strftime("%Y-%m-%d")


def _pending(conn) -> int:
    return conn.execute(
        "SELECT COUNT(*) c FROM crawl_queue WHERE state = 'pending'").fetchone()["c"]


def run_daily(conn, cfg) -> None:
    started = utc_now()
    session = requests.Session()
    limiter = fool.RateLimiter(cfg.requests_per_second)
    try:
        fool.enumerate_recent(conn, cfg, session, limiter)
        today = datetime.now(ZoneInfo(cfg.timezone))
        fool.enumerate_month(conn, cfg, session,
                             f"{today.year:04d}/{today.month:02d}", limiter)
    except requests.RequestException as e:
        log.warning("Discovery failed this run: %s", e)

    # Cap the daily crawl so a half-finished backfill queue cannot turn the
    # daily job into an all-day crawl.
    res = fool.crawl_pending(conn, cfg, limit=400)
    universe.backfill_names(conn)
    try:
        prices.update_prices(conn, cfg, range_="3mo", min_articles=3)
    except Exception as e:                     # noqa: BLE001 - never kill the loop
        log.error("Price refresh failed: %s", e)
    # The 3-month refresh keeps prices current but never reaches back to old
    # calls. Deepen history for any ticker that needs it (a no-op once each is
    # done), then rescore whatever calls are still maturing.
    try:
        prices.ensure_history(conn, cfg)
    except Exception as e:                     # noqa: BLE001
        log.error("Price history backfill failed: %s", e)
    try:
        study.compute_call_outcomes(conn)
    except Exception as e:                     # noqa: BLE001
        log.error("Call outcome scoring failed: %s", e)

    conn.execute(
        "INSERT INTO runs (kind, started_utc, finished_utc, articles_new, "
        "coverage_new, notes) VALUES ('daily', ?, ?, ?, ?, ?)",
        (started, utc_now(), res["articles"], res["coverage"],
         f"failed={res['failed']} gone={res['gone']} "
         f"aborted={res.get('aborted', False)}"))
    conn.commit()
    log.info("Daily run complete: %s", res)


def run_backfill_batch(conn, cfg, batch: int) -> tuple[bool, float]:
    """Crawl one batch.

    Returns (ok, cooldown_seconds). `cooldown_seconds` is what the origin
    actually asked for, so the caller can stand down for exactly that long —
    fool.com's cooldowns range from ten minutes to a full day, and treating a
    ten-minute throttle like a day-long ban wastes most of a night.
    """
    started = utc_now()
    res = fool.crawl_pending(conn, cfg, limit=batch)
    conn.execute(
        "INSERT INTO runs (kind, started_utc, finished_utc, articles_new, "
        "coverage_new, notes) VALUES ('backfill', ?, ?, ?, ?, ?)",
        (started, utc_now(), res["articles"], res["coverage"],
         f"failed={res['failed']} gone={res['gone']} "
         f"aborted={res.get('aborted', False)} "
         f"cooldown={res.get('cooldown_seconds', 0):.0f}s"))
    conn.commit()
    log.info("Backfill batch complete: %s (%d still pending)",
             res, _pending(conn))
    return not res.get("aborted", False), float(res.get("cooldown_seconds", 0.0))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = cfgmod.load()
    st = Settings(cfg.timezone)
    conn = get_conn()

    if not conn.execute("SELECT COUNT(*) c FROM universe").fetchone()["c"]:
        log.info("Universe empty — downloading listings")
        universe.download_universe(conn)

    log.info("Scheduler up. tz=%s daily_at=%s backfill=%s (%s-%s, batch=%d) "
             "rate=%.1f/s pending=%d",
             cfg.timezone, st.daily_at, "on" if st.enabled else "off",
             st.backfill_start, st.backfill_end, st.batch,
             cfg.requests_per_second, _pending(conn))

    last_daily: str | None = _last_daily_day(conn, st.tz)
    backfill_blocked_until: datetime | None = None

    while True:
        now = datetime.now(st.tz)
        today = now.strftime("%Y-%m-%d")

        try:
            if last_daily != today and now.timetz().replace(tzinfo=None) >= st.daily_at:
                log.info("Starting daily run for %s", today)
                run_daily(conn, cfg)
                last_daily = today

            elif (st.enabled and _pending(conn) > 0
                    and st.in_backfill_window(now)
                    and (backfill_blocked_until is None
                         or now >= backfill_blocked_until)):
                log.info("Backfill window — crawling a batch of %d", st.batch)
                ok, cooldown = run_backfill_batch(conn, cfg, st.batch)
                if not ok:
                    # Wait exactly as long as the origin asked, plus a small
                    # margin. A flat stand-down would throw away the rest of
                    # the night over a ten-minute throttle.
                    wait = cooldown if cooldown > 0 else 3600.0
                    backfill_blocked_until = now + timedelta(seconds=wait + 60)
                    log.warning("Origin asked for a %.0f min cooldown — pausing "
                                "backfill until %s", wait / 60,
                                backfill_blocked_until.strftime("%H:%M:%S"))
        except Exception as e:                 # noqa: BLE001 - the loop must survive
            log.exception("Tick failed, continuing: %s", e)

        time.sleep(st.tick)


if __name__ == "__main__":
    raise SystemExit(main())
