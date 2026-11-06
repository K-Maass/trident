# 02 — DATA SOURCES & CONNECTIONS

Every external connection in the system. Free tier first; paid upgrades at the bottom. Env var names match `.env.example`. **The live source registry for SENTINEL is `config/sentinel_sources.yaml`** — sources there can be flipped on/off without code changes; weights renormalize across whatever is enabled and healthy.

Free-tier limits below were verified July 2026 but change often — the `new-data-source` skill requires re-checking observed limits at onboarding time.

## Broker + market data

| Source | Used for | Limits (free) | Env | Notes |
|---|---|---|---|---|
| **Alpaca Trading API** | Paper+live orders, positions, account, clock, activities | 200 req/min | `ALPACA_API_KEY/SECRET` | Paper API identical to live. SDK: `alpaca-py` |
| **Alpaca Market Data** | Daily/minute bars, latest quotes | Free = IEX feed (fine for daily bars); SIP paid | same keys | Primary bar source AND the only price source at execution time |
| **yfinance** | Historical backfill (10+ yrs daily), splits | Unofficial Yahoo scraper — no SLA, breaks periodically | — | Backtest only. NEVER in the execution path (golden rule 8) |
| **Stooq** (stooq.com) | Redundant EOD backfill: free CSV download per ticker (`q/d/l/?s=aapl.us&i=d`), no key | Undocumented; be gentle | — | The fallback when yfinance breaks; cross-check adjusted closes between the two |
| **Twelve Data** | Optional third bar source | 800 calls/day, 8/min | `TWELVEDATA_API_KEY` (optional) | Only if Alpaca+yfinance+Stooq ever all disappoint |
| **pandas-market-calendars** | NYSE sessions, half-days, holidays | lib, no API | — | The scheduler's source of truth |

## News & social (SENTINEL) — see config/sentinel_sources.yaml

| Source | Used for | Limits (free) | Env | Status |
|---|---|---|---|---|
| **Finnhub** `/company-news` | Per-ticker news → own FinBERT scoring | 60 calls/min | `FINNHUB_API_KEY` | **Primary** |
| **RSS feeds** (Yahoo Finance per-ticker, CNBC, MarketWatch, Seeking Alpha) | Broad market context → own FinBERT | none | — (`feedparser`) | **Primary** |
| **ApeWisdom** (apewisdom.io/api) | Reddit **attention**: mentions + upvotes + 24h deltas across r/wallstreetbets, r/stocks, r/investing, r/options and more; paginated, refreshes ~2×/hr | Free, no key; undocumented limits — poll ≤ 2×/day | — | **Primary (attention_z)** |
| **Tradestie** (api.tradestie.com/v1/apps/reddit) | Top-50 WSB tickers with comment counts + their sentiment score, 15-min refresh; **`?date=MM-DD-YYYY` gives historical days → backfill** | Free, no key, 20 req/min/IP | — | **Primary (WSB sentiment)** |
| **Reddit Data API (PRAW)** | Raw posts/comments (own FinBERT) | 100 QPM OAuth — but since 2025 **all apps need manual pre-approval** (forms, ~2–4 wk review; account history matters) | `REDDIT_*` (optional) | **Optional — enable if/when approved** |
| **StockTwits** | Per-ticker stream + bull/bear self-labels | API **frozen to new developers** (2026) | — | **Verify-or-drop**: test unauth endpoints once in Phase 1; if dead, leave disabled |
| **Alpha Vantage** `NEWS_SENTIMENT` | Spot-check cross-validation only | **25 req/DAY** | `ALPHAVANTAGE_API_KEY` | Spot checks |
| **GDELT DOC 2.0** | Optional global news firehose (free, no key) | Generous but noisy; heavy filtering needed | — | Phase 7+ curiosity, not v1 |
| ~~X/Twitter~~ | — | $100+/mo | — | Skipped deliberately |

**Observed 2026-07-04 (Phase 1 ingestion live-probe, `src/trident/data/news.py`):**
- **Finnhub** `/company-news` ✅ live — banked real per-ticker news across the 40-name universe.
- **RSS** ✅ live — Yahoo per-ticker + CNBC + MarketWatch + Seeking Alpha all parse (note: the Yahoo per-ticker feed labels as `rss:finance` and MarketWatch/Dow Jones as `rss:feeds` from their hostnames — cosmetic; url_hash preserves true provenance).
- **ApeWisdom** ✅ live — attention snapshots land in `signals` (score=0, payload in metadata).
- **Tradestie** 🔴 **UNREACHABLE** — DNS resolves (`144.217.28.172`) but TCP connect times out on repeated attempts. It's the only backfillable social source (`?date=MM-DD-YYYY`); the collector flags it as skipped + WARN and renormalizes. Retry before relying on the historical backfill.
- **StockTwits** 🟢 unauth `streams/symbol/{T}.json` **alive** (config has it `disabled` as verify-or-drop → candidate to enable; caveat: tested from a residential IP, VPS behavior may differ).

## Fundamentals, events & macro (ORACLE + overlays)

| Source | Used for | Limits | Env |
|---|---|---|---|
| **SEC EDGAR** (companyfacts, full-text search, filings) | 10-K/10-Q XBRL facts, MD&A/risk text | Free, 10 req/s, **descriptive User-Agent required** | `EDGAR_USER_AGENT` — lib: `edgartools` |
| **SEC Form 4 via EDGAR** | Insider buys/sells (optional ORACLE tool, doc 05) | same | same |
| **API Ninjas** (api-ninjas.com) | One free key (no card) → earnings **calendar** (actual vs estimated EPS; some fields premium-masked on free), latest earnings-call **transcripts** (recent archive needs paid tier), pre-parsed **insider trading**, stock news | Free key; per-endpoint free limits — verify at signup | `API_NINJAS_KEY` (optional) |
| **yfinance** `.info/.financials` | Quick ratios, estimates | unofficial | — |
| **Financial Modeling Prep** | Clean ratios, peers | 250 req/day | `FMP_API_KEY` (optional) |
| **FRED** | Risk-free rate (DGS3MO), macro | 120 req/min | `FRED_API_KEY` (optional; fallback rf=0, noted in reports) |

## LLM

| Model | Used for | Est. cost |
|---|---|---|
| `claude-sonnet-4-6` | ORACLE reasoning/thesis writing | ~$5–20/mo |
| `claude-haiku-4-5` | SENTINEL relevance filter; bulk extraction | ~$2–8/mo |
| **FinBERT** (`ProsusAI/finbert`) | Primary sentiment scorer for raw-text sources — local, $0 | free |

## Scraping & integration platforms (evaluated July 2026)

- **Apify** — *adopted as optional fallback.* Marketplace of maintained scrapers ("actors": Yahoo Finance, SEC EDGAR, Finviz screener, earnings calendar, Reddit…), pay-per-use with a small free monthly platform credit (~$5 — verify current). Use only when a needed source has no sane API (e.g. raw Reddit text at scale without approval). `APIFY_TOKEN` in .env; wrap like any other client (limiter, cache, freshness). Note: scraping Reddit sits outside Reddit's ToS — personal-research judgment call, never a hard dependency.
- **Composio** — *evaluated, not adopted.* Agent tool-calling infra (managed OAuth to 1,000+ SaaS apps; free tier ≈ 20k tool calls/mo). Wrong shape for Trident: our needs are a handful of REST feeds on a schedule, where direct httpx clients are thinner and more reliable. Revisit only if ORACLE should ever *act* in SaaS apps (write theses to Notion, file GitHub issues).
- **OpenBB Platform** (open source) — *study, not a dependency.* Unified Python layer over many of the same free providers. Good reference for provider quirks; adds a layer without adding data for our narrow universe.

## Claude Code / Claude.ai connections (dev-time)

- **Postgres MCP** (`.mcp.json`): query fund state in natural language.
- **Alpaca MCP server** (github.com/alpacahq/alpaca-mcp-server): optional; verify install cmd; PAPER keys only.
- **Apify MCP**: available as a Claude.ai connector — handy for one-off "what actors exist for X" exploration before committing code.
- **GitHub MCP or `gh` CLI**: PRs/issues; CI in `.github/workflows/ci.yml`.

## Caching & rate-limit policy

All raw pulls cached: bars → parquet under `data/cache/bars/`, news/social/fundamentals → Postgres with `fetched_at`. Cache-first reads; token-bucket per client at 80% of documented limits (for undocumented free services: hard-coded conservative ceilings in sentinel_sources.yaml). Daily 07:30 ET freshness job asserts staleness bounds per source; a stale/dead source is auto-disabled for the cycle and weights renormalize — a WARN health event fires, never a crash.

## Terms-of-service honesty

yfinance and Stooq are unofficial (research/backfill only). Reddit now requires a valid OAuth token for any access from hosted-provider IP ranges — so no-auth `reddit.com/*.json` reads only work from residential IPs, which conflicts with a VPS deployment; that's why the aggregators are primary. ApeWisdom/Tradestie are free hobby-grade services: no SLA, no published commercial terms — fine for personal research, never load-bearing without the renormalization fallback. EDGAR is public domain — the one truly production-grade free source here.

## Paid upgrade path (only if free tiers pinch)

Polygon.io Starter $29/mo (bars + news), Tiingo ~$10/mo (clean EOD + fundamentals), Alpaca Algo Trader Plus $99/mo (SIP real-time — irrelevant at daily cadence), Quiver Quantitative from ~$30/mo (institutional-ish alt-data incl. Reddit/insider/congress if SENTINEL ever deserves a budget).
