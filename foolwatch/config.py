"""Configuration loading: config.toml + environment variable overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config.toml"
DB_PATH = DATA_DIR / "foolwatch.db"
LOG_PATH = DATA_DIR / "foolwatch.log"

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


@dataclass
class Config:
    # [fool]
    # Sections of fool.com to keep. Articles under any other path are recorded
    # in the crawl queue but skipped — /investing/ is where tickers live.
    sections: list[str] = field(default_factory=lambda: ["investing"])
    # Politeness. requests_per_second is a global cap shared by all workers.
    requests_per_second: float = 4.0
    workers: int = 4
    # Metadata all sits in the first ~15 KB of an article; reading past that
    # wastes ~95% of the transfer (a full article page is ~300 KB).
    head_bytes: int = 24_000
    retries: int = 3
    # A 429 whose Retry-After exceeds this means the origin wants us gone for
    # hours; the crawl stops and leaves the queue pending rather than digging in.
    max_backoff_seconds: float = 120.0

    # [analysis]
    timezone: str = "America/New_York"
    price_lookback_days: int = 45
    # A ticker only listed in an article's meta tag (not the headline, not an
    # explicit "(NASDAQ: X)" reference) is incidental — a passing comparison.
    primary_only_default: bool = True

    def section_ok(self, path: str) -> bool:
        if not self.sections:
            return True
        head = path.lstrip("/").split("/", 1)[0]
        return head in self.sections


def load(path: Path = CONFIG_PATH) -> Config:
    cfg = Config()
    if path.exists():
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        fool = raw.get("fool", {})
        cfg.sections = fool.get("sections", cfg.sections)
        cfg.requests_per_second = float(fool.get("requests_per_second", cfg.requests_per_second))
        cfg.workers = int(fool.get("workers", cfg.workers))
        cfg.head_bytes = int(fool.get("head_bytes", cfg.head_bytes))
        cfg.retries = int(fool.get("retries", cfg.retries))
        cfg.max_backoff_seconds = float(
            fool.get("max_backoff_seconds", cfg.max_backoff_seconds))

        analysis = raw.get("analysis", {})
        cfg.timezone = analysis.get("timezone", cfg.timezone)
        cfg.price_lookback_days = int(analysis.get("price_lookback_days", cfg.price_lookback_days))
        cfg.primary_only_default = bool(analysis.get("primary_only_default", cfg.primary_only_default))

    # Environment overrides. config.toml is baked into the container image, so
    # the politeness knobs - the ones worth changing in a hurry - are settable
    # without a rebuild.
    def _env(name: str, cast, current):
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            return current
        try:
            return cast(raw)
        except ValueError:
            return current

    cfg.requests_per_second = _env("FOOLWATCH_RPS", float, cfg.requests_per_second)
    cfg.workers = _env("FOOLWATCH_WORKERS", int, cfg.workers)
    cfg.max_backoff_seconds = _env("FOOLWATCH_MAX_BACKOFF", float,
                                   cfg.max_backoff_seconds)
    cfg.timezone = os.environ.get("TZ") or cfg.timezone

    DATA_DIR.mkdir(exist_ok=True)
    return cfg
