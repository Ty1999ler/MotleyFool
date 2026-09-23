"""foolwatch CLI."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import requests

from . import config as cfgmod
from . import fool, prices, universe
from .db import get_conn, utc_now


def setup_logging(verbose: bool = False) -> None:
    cfgmod.DATA_DIR.mkdir(exist_ok=True)
    # The Windows console defaults to cp1252, which mangles any non-ASCII
    # character in a headline or a log line.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    handlers: list[logging.Handler] = [logging.FileHandler(cfgmod.LOG_PATH, encoding="utf-8")]
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    handlers.append(console)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    for h in handlers:
        if isinstance(h, logging.FileHandler):
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))


log = logging.getLogger("foolwatch")


def _months_between(start: str, end: str) -> list[str]:
    """Inclusive list of YYYY/MM strings."""
    sy, sm = (int(x) for x in start.split("/"))
    ey, em = (int(x) for x in end.split("/"))
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}/{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _default_range(months_back: int) -> tuple[str, str]:
    today = date.today()
    end = f"{today.year:04d}/{today.month:02d}"
    y, m = today.year, today.month
    for _ in range(months_back - 1):
        m -= 1
        if m < 1:
            y, m = y - 1, 12
    return f"{y:04d}/{m:02d}", end


# --- commands ----------------------------------------------------------------

def cmd_setup(args, cfg, conn) -> int:
    n = universe.download_universe(conn)
    print(f"Ticker universe: {n} symbols")
    print(f"Database: {cfgmod.DB_PATH}")
    print("\nNext: `py -m foolwatch daily` for today, or "
          "`py -m foolwatch backfill --months 36` for three years of history.")
    return 0


def cmd_enumerate(args, cfg, conn) -> int:
    session = requests.Session()
    limiter = fool.RateLimiter(cfg.requests_per_second)
    if args.months:
        start, end = _default_range(args.months)
    else:
        start, end = args.start, args.end
    if not start or not end:
        print("Give either --months N or both --from YYYY/MM and --to YYYY/MM")
        return 2

    have = {r["month"] for r in conn.execute("SELECT month FROM archive_months")} \
        if not args.refresh else set()
    months = [m for m in _months_between(start, end) if m not in have]
    print(f"Enumerating {len(months)} month archives ({start} -> {end})")
    total = 0
    for m in months:
        try:
            total += fool.enumerate_month(conn, cfg, session, m, limiter)
        except requests.RequestException as e:
            log.error("Month %s failed: %s", m, e)
    pending = conn.execute(
        "SELECT COUNT(*) c FROM crawl_queue WHERE state = 'pending'").fetchone()["c"]
    print(f"Queued {total} new article urls. Pending now: {pending}")
    return 0


def cmd_crawl(args, cfg, conn) -> int:
    if args.requeue:
        n = fool.requeue_failed(conn)
        print(f"Requeued {n} previously failed urls")
    started = utc_now()
    res = fool.crawl_pending(conn, cfg, limit=args.limit)
    conn.execute(
        "INSERT INTO runs (kind, started_utc, finished_utc, articles_new, coverage_new, notes) "
        "VALUES ('crawl', ?, ?, ?, ?, ?)",
        (started, utc_now(), res["articles"], res["coverage"],
         f"failed={res['failed']} gone={res['gone']}"),
    )
    conn.commit()
    print(f"Articles stored: {res['articles']}  coverage rows: {res['coverage']}  "
          f"failed: {res['failed']}  gone: {res['gone']}")
    return 0


def cmd_backfill(args, cfg, conn) -> int:
    rc = cmd_enumerate(args, cfg, conn)
    if rc:
        return rc
    return cmd_crawl(args, cfg, conn)


def cmd_daily(args, cfg, conn) -> int:
    started = utc_now()
    session = requests.Session()
    limiter = fool.RateLimiter(cfg.requests_per_second)
    fool.enumerate_recent(conn, cfg, session, limiter)
    # The current month's archive catches anything the 3-day news window missed.
    today = date.today()
    try:
        fool.enumerate_month(conn, cfg, session,
                             f"{today.year:04d}/{today.month:02d}", limiter)
    except requests.RequestException as e:
        log.warning("Current-month enumeration failed: %s", e)

    res = fool.crawl_pending(conn, cfg, limit=args.limit)
    universe.backfill_names(conn)
    if not args.skip_prices:
        prices.update_prices(conn, cfg, range_="3mo",
                             min_articles=args.min_articles)
    conn.execute(
        "INSERT INTO runs (kind, started_utc, finished_utc, articles_new, coverage_new, notes) "
        "VALUES ('daily', ?, ?, ?, ?, ?)",
        (started, utc_now(), res["articles"], res["coverage"],
         f"failed={res['failed']} gone={res['gone']}"),
    )
    conn.commit()
    print(f"\nDaily done — {res['articles']} new articles, {res['coverage']} coverage rows.")
    return 0


def cmd_prices(args, cfg, conn) -> int:
    if args.history:
        res = prices.ensure_history(conn, cfg, limit=args.limit)
        print(f"Price history: {res['fetched']} fetched, {res['failed']} failed, "
              f"{res['rows']:,} rows added ({res['needed']} needed)")
        return 0
    res = prices.update_prices(conn, cfg, range_=args.range,
                               min_articles=args.min_articles,
                               only_missing=args.only_missing)
    print(f"Prices: {res['updated']} symbols updated, {res['failed']} failed")
    return 0


def cmd_outcomes(args, cfg, conn) -> int:
    from . import study

    res = study.compute_call_outcomes(conn, full=args.full)
    print(f"Call outcomes: {res['computed']:,} computed, "
          f"{res['scored']:,} with a usable entry price")
    return 0


def cmd_restance(args, cfg, conn) -> int:
    """Re-run the headline classifier over every stored article.

    Stance is written at crawl time, so any change to stance.py leaves the
    existing rows stale. This recomputes them in place — no refetching, which
    matters when the origin has us rate-limited.
    """
    from .stance import classify

    rows = conn.execute("SELECT path, title, stance, stance_score FROM articles").fetchall()
    changed = 0
    moves: dict[str, int] = {}
    for r in rows:
        label, score = classify(r["title"])
        if label != r["stance"] or score != r["stance_score"]:
            conn.execute(
                "UPDATE articles SET stance = ?, stance_score = ? WHERE path = ?",
                (label, score, r["path"]))
            moves[f"{r['stance']} -> {label}"] = moves.get(
                f"{r['stance']} -> {label}", 0) + 1
            changed += 1
    conn.commit()
    print(f"Re-scored {len(rows):,} articles; {changed:,} changed.")
    for move, n in sorted(moves.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {n:5}  {move}")
    return 0


def cmd_status(args, cfg, conn) -> int:
    q = lambda s, *a: conn.execute(s, a).fetchone()
    arts = q("SELECT COUNT(*) c, MIN(published_day) mn, MAX(published_day) mx FROM articles")
    print("=== foolwatch status ===")
    print(f"DB                {cfgmod.DB_PATH}")
    print(f"Articles          {arts['c']:,}  ({arts['mn']} -> {arts['mx']})")
    print(f"Coverage rows     {q('SELECT COUNT(*) c FROM coverage')['c']:,}")
    print(f"Tickers           {q('SELECT COUNT(DISTINCT ticker) c FROM coverage')['c']:,}"
          f"  (primary: {q('SELECT COUNT(DISTINCT ticker) c FROM coverage WHERE is_primary=1')['c']:,})")
    print(f"Authors           {q('SELECT COUNT(DISTINCT author) c FROM articles')['c']:,}")
    print(f"Months enumerated {q('SELECT COUNT(*) c FROM archive_months')['c']:,}")
    print(f"Universe          {q('SELECT COUNT(*) c FROM universe')['c']:,} symbols")
    print(f"Price rows        {q('SELECT COUNT(*) c FROM prices')['c']:,}"
          f" over {q('SELECT COUNT(DISTINCT symbol) c FROM prices')['c']:,} symbols")
    print("\nCrawl queue:")
    for r in conn.execute("SELECT state, COUNT(*) c FROM crawl_queue GROUP BY state ORDER BY c DESC"):
        print(f"  {r['state']:10} {r['c']:,}")
    print("\nStance mix:")
    for r in conn.execute("SELECT stance, COUNT(*) c FROM articles GROUP BY stance ORDER BY c DESC"):
        print(f"  {r['stance']:12} {r['c']:,}")
    return 0


def cmd_dashboard(args, cfg, conn) -> int:
    app = Path(__file__).resolve().parent / "dashboard.py"
    conn.close()
    cmd = [sys.executable, "-m", "streamlit", "run", str(app),
           "--server.port", str(args.port), "--server.headless", "false"]
    print(f"Starting dashboard at http://localhost:{args.port} — Ctrl+C to stop")
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="foolwatch",
                                description="Track Motley Fool coverage over time.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup", help="download the ticker universe and init the DB")

    def add_range(sp):
        sp.add_argument("--months", type=int, help="how many months back from today")
        sp.add_argument("--from", dest="start", help="start month YYYY/MM")
        sp.add_argument("--to", dest="end", help="end month YYYY/MM")
        sp.add_argument("--refresh", action="store_true",
                        help="re-read months already enumerated")

    en = sub.add_parser("enumerate", help="queue article urls from archive sitemaps")
    add_range(en)

    cr = sub.add_parser("crawl", help="fetch queued articles (resumable)")
    cr.add_argument("--limit", type=int, help="stop after N articles")
    cr.add_argument("--requeue", action="store_true", help="retry previously failed urls")

    bf = sub.add_parser("backfill", help="enumerate months then crawl them")
    add_range(bf)
    bf.add_argument("--limit", type=int)
    bf.add_argument("--requeue", action="store_true")

    dl = sub.add_parser("daily", help="pull the last few days and refresh prices")
    dl.add_argument("--limit", type=int)
    dl.add_argument("--skip-prices", action="store_true")
    dl.add_argument("--min-articles", type=int, default=3)

    pr = sub.add_parser("prices", help="refresh Yahoo closes")
    pr.add_argument("--range", default="2y", help="Yahoo range: 3mo, 1y, 2y, 5y, max")
    pr.add_argument("--min-articles", type=int, default=3,
                    help="skip tickers with fewer than N coverage rows")
    pr.add_argument("--only-missing", action="store_true")
    pr.add_argument("--history", action="store_true",
                    help="deepen history so every call can be scored")
    pr.add_argument("--limit", type=int, help="with --history: at most N symbols")

    oc = sub.add_parser("outcomes", help="score what happened after every call")
    oc.add_argument("--full", action="store_true",
                    help="rescore every call, not just maturing ones")

    sub.add_parser("restance",
                   help="re-run the headline classifier over stored articles")
    sub.add_parser("status", help="show what is in the database")

    db = sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    # 8501 is Streamlit's default and is already taken locally by another app.
    db.add_argument("--port", type=int, default=8531)

    args = p.parse_args(argv)
    setup_logging(args.verbose)
    cfg = cfgmod.load()
    conn = get_conn()

    handlers = {
        "setup": cmd_setup, "enumerate": cmd_enumerate, "crawl": cmd_crawl,
        "backfill": cmd_backfill, "daily": cmd_daily, "prices": cmd_prices,
        "status": cmd_status, "dashboard": cmd_dashboard,
        "restance": cmd_restance, "outcomes": cmd_outcomes,
    }
    try:
        return handlers[args.cmd](args, cfg, conn)
    except KeyboardInterrupt:
        print("\nInterrupted — progress is saved; rerun the same command to resume.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
