# foolwatch

Tracks what **The Motley Fool** writes about: which tickers it covers, **when it
first covered each one**, and **how the editorial take moved over time** — with a
Streamlit dashboard over the whole history.

> Coverage tracking for context — not investment advice. The Fool publishes
> roughly 2.5 bullish articles for every cautious one, so the interesting signal
> is the *change*, not any single article.

## Why sitemaps, not page scraping

fool.com's own `robots.txt` advertises the sitemaps this tool uses:

| Source | What it gives |
|---|---|
| `/news-sitemap.xml` | the last ~3 days (~250 articles), for daily runs |
| `/sitemap/` | an index of month archives, **back to 1995/03** |
| `/sitemap/YYYY/MM` | every URL published that month (~3,000–6,000) |

Scraping the `/investing-news/` listing page instead only ever sees the ~13
newest links, which is why an article-listing approach silently misses most of
what gets published.

Each article is then fetched **head-only**: the reader stops at `</head>`, about
17 KB into a ~300 KB page. Everything needed is up there, including the two tags
that matter most:

```html
<meta name="tickers"          content="NVDA,AVGO,GOOGL,AMZN,META,AMD,GOOG">
<meta name="primary_tickers"  content="NVDA">
```

That second tag is fool.com stating which company the article is *actually*
about. It matters more than it looks: an article headlined
*"Greg Abel Has 75% of Berkshire's Portfolio in Just 8 Stocks"* meta-tags 15
tickers but is about one. Counting all 15 as mentions would invent 14 false
"first coverage" dates. The dashboard defaults to primary coverage only.

## Setup

```bash
py -m pip install -r requirements.txt
```

```bash
py -m foolwatch setup
```

`setup` downloads the NASDAQ/NYSE listing files (~12,800 symbols) used to verify
that a ticker is a real listing, and creates `data/foolwatch.db`.

## Rate limits — read this before backfilling

fool.com rate-limits hard. **6 req/s across 8 workers earned a 23-hour 429 ban**
after ~420 articles. The defaults are now 2 req/s across 4 workers, and:

- every 429 halves the request rate, which only creeps back after 200 clean fetches;
- a 429 whose `Retry-After` exceeds `max_backoff_seconds` **stops the crawl** and
  leaves the queue `pending`, instead of grinding through 100,000 urls marking
  them all failed;
- throttled urls are never marked `failed`, so nothing is lost.

If you see 429s, lower `requests_per_second`. Do not raise it.

## Backfill the history

History is the whole point, and it cannot be recovered later — the archive is
there now, so pull it once:

```bash
py -m foolwatch backfill --months 36
```

That enumerates 36 monthly archives (~110,000 `/investing/` articles) and
crawls them. At the shipped 6 req/s across 8 workers it takes **roughly 6
hours**. It is fully **resumable** — Ctrl+C and rerun the same command, or
`py -m foolwatch crawl` to keep draining the queue. Nothing is refetched.

Smaller bites work too:

```bash
py -m foolwatch backfill --from 2026/01 --to 2026/09
```

```bash
py -m foolwatch crawl --limit 5000
```

Then fetch prices for everything with real coverage:

```bash
py -m foolwatch prices --range 5y --min-articles 3
```

## Daily use

```bash
py -m foolwatch daily
```

Reads the news sitemap plus the current month's archive, crawls whatever is new,
and refreshes prices. Runs in a couple of minutes.

## The dashboard

```bash
py -m foolwatch dashboard
```

Opens at <http://localhost:8531> (not Streamlit's default 8501, which is already
taken by another app on this machine).

| View | What it answers |
|---|---|
| **Overview** | How much is being published, the bullish/bearish balance, most-covered tickers. **Select a row to read that company's articles in place.** |
| **Track record** | What actually happened after every call: hit rate with a confidence interval, median beside mean, which call types beat a coin flip, a growth-of-$1 line for buying every bullish call vs SPY, the outcome distribution, performance by quarter, best and worst calls, and every call with its link. Filter by holding period (1/3/6/12 months), direction and author. |
| **Ticker history** | When a ticker was first covered, every article since with links, net stance over time, price vs. SPY. The picker searches on company name as well as symbol. |
| **First coverage** | Coverage start date, **how fast coverage ramped** (days to 2nd article, articles in the first 30/90 days), and market-adjusted returns **1/3/12 months before and after** the first article |
| **Signals** | **Stance flips** — where the Fool changed its mind about a company — and **coverage spikes**, companies it has suddenly started writing about a lot |
| **Early vs late** | Whether the 1st article beat later ones, bucketed and paired within ticker |
| **Author accuracy** | How often each author's directional calls went the right way, with Wilson intervals and a base-rate comparison |
| **Authors** | Who writes the most, how many tickers they cover, their stance lean |

### Reading the Track record

Every call's return is measured from the first close on or after publication,
**minus SPY over the same trading sessions**, and signed so a positive number
always means the call paid — a bearish call scores when the stock lags. A
"call" is a headline that takes a direction; news, predictions and analysis
are excluded rather than counted as wrong.

Two things to hold onto when reading it:

- **Median and mean disagree, on purpose.** Across 2023–2026 the median call
  trailed SPY while the mean beat it, because a small minority of enormous
  winners carries the average. That is why the growth-of-$1 line can beat SPY
  even though most individual calls lost to it: a diversified basket catches
  the winners.
- **Survivorship flatters everything on the page.** Delisted and acquired
  stocks have no price history, so their calls can't be scored at all. Those
  skew toward losers. The page reports how many calls it couldn't score.

Outcomes are precomputed by the worker into a `call_outcomes` table after each
daily run, so the page reads one small table rather than millions of price
rows. Stance is joined fresh from `articles`, so a `restance` after a
classifier fix takes effect without recomputing anything.

### The before/after columns

The **First coverage** tab's `−12m / −3m / −1m` columns are the most useful thing
in the dashboard, and the reason is subtle: if a company is already up sharply
vs. SPY in the year *before* the Fool's first article, then coverage is
following the move rather than leading it. That is the biggest confounder in
"does getting in early pay", and these columns make it a number instead of a
disclaimer. Months are trading sessions (1m = 21, 3m = 63, 12m = 252).

Filling the `−12m` column needs a year of price history *before* each article,
so run `foolwatch prices --range 5y` (or `max`) after a deep backfill.

Note the asymmetry: survivorship affects the **after** columns and not the
**before** ones, which tilts the comparison in favour of "after". A company that
delisted has no price history and is simply absent.

Sidebar filters (date range, primary-coverage-only, listed-tickers-only) apply
across every view.

### A note on the charts

Coverage volume and price deliberately **never share a y-axis** — two measures
on different scales get two charts on a common date axis, never a dual axis.
Stance is diverging around a real zero, so it uses a blue↔red pair (blue
bullish) rather than the conventional green/red: green-vs-red is the least
colourblind-safe pair available, and this one is measured safe in both light and
dark mode.

## Stance classification

`stance.py` turns a headline into a label plus a signed score (−2…+2). Rules are
ordered, first match wins, and the ordering is load-bearing: *"Abercrombie &
Fitch Chief HR Officer Sells 5,000 Shares"* is an insider filing, not a sell
rating, so the insider rule runs before any buy/sell wording is considered.

| Label | Score | Example |
|---|---|---|
| `buy` | +2 | "2 High-Yield Dividend Stocks to Buy Now" |
| `buy_lean` | +1 | "Should You Buy Oracle Stock While It's Down 53%?" |
| `hold` / `mixed` | 0 | "1 AI Chip Stock to Buy, 1 to Hold, and 1 to Sell" |
| `prediction` | 0 | "Prediction: Tesla Stock Will Be Worth This Much in 2031" |
| `news` | 0 | "Why Baxter International Stock Skyrocketed" |
| `insider` | 0 | "Five Below CHRO Sells 1,610 Shares for $401,000" |
| `comparison` | 0 | "Nvidia vs. AMD: Which Is the Better AI Chip Stock?" |
| `analysis` | 0 | "What Would It Take for Investors to Pay More for Toast Stock?" |
| `warning` | −1 | "SpaceX Investors Just Got Some Bad News" |
| `sell_lean` | −1 | "Should You Avoid Wendy's Stock, Even After a 66% Decline?" |
| `sell` | −2 | "3 Reasons to Sell This Stock Now" |

Calibrated against 356 live headlines. Headlines are formulaic, so this is
accurate enough to chart; it is a keyword classifier, not a language model, and
it will occasionally misread an unusual headline.

Two rules run *before* any buy/sell wording is considered, because both forms
contain that wording while making no call:

- **`insider`** — "Chief HR Officer Sells 5,000 Shares" is a Form 4 filing.
- **`comparison`** — "Nvidia vs. AMD: Which Is the Better Buy?" is a relative
  call, and the article is tagged to *both* tickers, so scoring it bullish would
  credit a buy call to the loser too.

## Deploying to a Synology NAS (Docker Compose)

Two containers share one SQLite file: a **worker** that scrapes daily and drips
through the archive backfill overnight, and the **dashboard**.

1. Create the data folder on the NAS — e.g. `/volume1/docker/foolwatch/data`.
2. Copy this project there (File Station, or `git clone` over SSH).
3. `cp .env.example .env` and set `FOOLWATCH_DATA` to that folder.
4. **Container Manager → Project → Create**, point it at this folder, and let it
   read `docker-compose.yml`. Or over SSH:

```bash
docker compose up -d --build
```

The dashboard is then at `http://<nas-ip>:8531`, reachable from your LAN but not
the internet. To require an SSH tunnel instead, change the port mapping in
`docker-compose.yml` to `"127.0.0.1:8531:8501"`.

### What the worker does on its own

| When | What |
|---|---|
| `FOOLWATCH_DAILY_AT` (default 18:30) | News sitemap + current month, crawl new articles, refresh prices |
| `FOOLWATCH_BACKFILL_START`–`_END` (default 01:00–06:00) | Crawl a batch of the archive queue, then sleep |

At 2 req/s a five-hour window covers roughly 20,000 articles, so the ~103,000
article archive completes in about **four nights**. Once the queue is empty, set
`FOOLWATCH_BACKFILL=0` and the worker just does the daily run.

If fool.com asks for a long cooldown, the worker stands down for 24 hours and
logs it — it will not retry into a wall. Watch it with:

```bash
docker compose logs -f worker
```

```bash
docker compose exec worker python -m foolwatch status
```

### A note on SQLite across two containers

WAL mode handles one writer and one reader, which is exactly this shape. Keep
the bind mount on a **local** NAS volume — WAL misbehaves over SMB/NFS.

## Is getting in early better?

The **Early vs late** dashboard tab is an event study. For every primary
article it measures the return from buying at that day's close, minus SPY's
return over the same trading days, then groups by whether it was the 1st, 2nd,
3rd or later article about that ticker. There is also a within-ticker 1st-vs-2nd
comparison, which is the sharpest form of the question because it removes every
difference between stocks.

Three things make the answer trustworthy, and it is worthless without them:

- **Market-adjusted.** Over a rising three years, an unadjusted entry always
  looks good. SPY is subtracted from every return.
- **Trading-day horizons.** `+21d` is 21 sessions, stepped through the price
  series, not 21 calendar days.
- **Censoring control.** A three-year archive cannot know the first time the Fool
  wrote about Apple. Tickers whose earliest observed article sits near the start
  of your data are dropped, because their "1st article" is an artefact of where
  the backfill begins. This is why the tab looks empty until the backfill is
  well along.

Biases that remain and cannot be fixed from this data: **survivorship**
(delisted tickers have no price history, so losers quietly leave the sample and
every bucket is flattered), and the fact that **coverage follows momentum** — the
Fool writes about stocks that already moved, so a first article is not a random
entry point. Costs, slippage and taxes are not modelled.

The verdict line is deliberately hard to please: the mean, the median and the
win rate must all agree before it will call anything a signal. A positive mean
sitting on a flat median and a sub-50% win rate is a handful of outliers, which
is the easiest way to fool yourself here. Read the output as "what happened
after coverage", not as a strategy — and not as advice.

## Scheduling (Windows, without Docker)

```bash
schtasks /Create /TN "foolwatch daily" /TR "\"C:\Users\atp2txw\PycharmProjects\foolwatch\run_daily.bat\"" /SC DAILY /ST 18:30
```

## Commands

| Command | What it does |
|---|---|
| `setup` | Download the ticker universe, create the DB |
| `daily` | News sitemap + current month, crawl new, refresh prices |
| `backfill --months N` | Enumerate N months of archive, then crawl |
| `enumerate --from YYYY/MM --to YYYY/MM` | Queue urls without crawling |
| `crawl [--limit N] [--requeue]` | Drain the pending queue; resumable |
| `prices [--range 5y] [--min-articles N]` | Refresh Yahoo closes |
| `prices --history [--limit N]` | Deepen each ticker's price history to a year before its first call, so old calls can be scored. Idempotent — each ticker is fetched once. The daily run does this automatically. |
| `outcomes [--full]` | Score what happened after every call into `call_outcomes`. Incremental: only calls whose 12-month window is still open are revisited. The daily run does this automatically. |
| `restance` | Re-run the headline classifier over stored articles, in place. Run this after any change to `stance.py` — stance is written at crawl time, so existing rows go stale, and this needs no refetching. |
| `status` | Article/coverage/queue/stance counts |
| `dashboard [--port N]` | Launch the Streamlit app |

## Data layout

- `data/foolwatch.db` — SQLite. `articles` (one row per article, with stance),
  `coverage` (one row per article+ticker, with the primary flag), `crawl_queue`
  (resumability), `prices`, `universe`, `tickers`, `archive_months`, `runs`.
- `data/foolwatch.log` — run log.

Plain SQLite — open it with any browser and run your own queries:

```sql
-- Tickers the Fool turned cautious on: bullish early, bearish lately
SELECT c.ticker,
       ROUND(AVG(CASE WHEN c.published_day < '2026-01-01' THEN a.stance_score END), 2) AS before_2026,
       ROUND(AVG(CASE WHEN c.published_day >= '2026-01-01' THEN a.stance_score END), 2) AS since_2026,
       COUNT(*) AS articles
FROM coverage c JOIN articles a ON a.path = c.path
WHERE c.is_primary = 1
GROUP BY c.ticker
HAVING articles >= 20 AND before_2026 > 0.3 AND since_2026 < -0.1
ORDER BY (before_2026 - since_2026) DESC;
```

## Politeness and scope

Only article metadata is stored — url, headline, author, date, tickers, and a
stance label derived from the headline. No article bodies. The crawler reads
only the sitemaps fool.com publishes for crawlers, respects a global request
cap, and stays off everything `robots.txt` disallows (`/premium-reports`,
`/newsletters/`, `/investing/stocks/`, and the rest). Lower
`requests_per_second` in `config.toml` if you want it gentler.
