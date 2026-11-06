# 04 — SENTINEL (media-context agent)

## Thesis
Aggregated shifts in media tone and attention lead or coincide with short-term price moves, especially around events. The edge is not "positive news → buy" (too fast, priced in) but **changes vs. a ticker's own baseline**: sentiment z-score spikes, attention surges, and bull/bear divergence.

## Source registry — config/sentinel_sources.yaml
All ingestion sources, their kinds, weights, and env requirements live in that file, not in code. Rules the pipeline must honor:
1. A source that is disabled, missing its env keys, or failing freshness checks contributes nothing and its weight is **renormalized away** for that cycle (WARN health event; never a crash, never a stale value silently reused).
2. `raw_text` sources are scored by our FinBERT; `pre_scored` sources (aggregators) contribute their provider score, stored alongside provenance in metadata.
3. `attention_only` sources feed `attention_z` but never `sent_raw`.

## Pipeline (agents/sentiment/)

**1. Ingest (`ingest.py`)** — 08:00 ET + hourly 10:00–15:00 ET:
- **Finnhub** `/company-news` per universe ticker (last 24h) — raw_text
- **RSS**: Yahoo Finance per-ticker, CNBC Markets, MarketWatch, Seeking Alpha — raw_text
- **ApeWisdom** (`/api/v1.0/filter/all-stocks/page/N`): mentions, upvotes, 24h deltas per ticker — attention_only; poll 2×/day
- **Tradestie** (`api.tradestie.com/v1/apps/reddit`, 20 req/min): top-50 WSB tickers, comment counts + provider sentiment_score — pre_scored + attention; poll 2×/day
- **Reddit (PRAW)** — raw_text, *optional*: enabled only once the operator's app is approved (2025 policy: manual review). Adds raw WSB/r/stocks text under our own FinBERT.
- **StockTwits** — *verify-or-drop* (API frozen to new devs): one Phase 1 probe of unauth per-ticker streams; enable only if alive.
- Dedupe raw_text by URL hash + fuzzy title (rapidfuzz > 90) → `news_items(source, ticker, title, body, url, published_at)`. Aggregator snapshots → `signals` metadata with `fetched_at`.

**2. Relevance filter** (raw_text only) — `claude-haiku-4-5`, batched 20 items/call, temperature 0: "Is this item materially about {ticker}'s business/stock? YES/NO + one-line reason." Kills ticker-symbol false positives ("A", "ALL") and meme noise. Cache verdicts by item hash.

**3. Score (`score.py`)** — raw_text: FinBERT (`ProsusAI/finbert`) on title+first 512 chars → p(pos) − p(neg) ∈ [−1, 1], local CPU, batch 32. pre_scored: clamp provider score to [−1, 1] and record provider name + raw value in metadata; during the first month of Phase 4, log the correlation between Tradestie's score and our FinBERT on overlapping tickers (calibration check — if they disagree wildly, down-weight Tradestie in the yaml and note it in the weekly report).

**4. Aggregate (`aggregate.py`)** — per ticker, daily:
- `sent_raw` = weight-normalized mean over enabled sentiment sources (defaults in yaml: finnhub_news 1.0, rss 0.8, tradestie 0.5, reddit_praw 0.4 when enabled, stocktwits 0.5 if verified) with exponential time decay (half-life 24h)
- `sent_z` = (sent_raw − 21d mean) / 21d std  ← the primary signal
- `attention_z` = z over 21d baseline of: raw item count + ApeWisdom mentions (each z-scored separately, then averaged over what's enabled)
- `disagreement` = std of item scores (high = contested story)
Persist to `signals(agent='sentinel', metadata={components, sources_used})` — `sources_used` makes every signal auditable against outages.

**5. Decide (`agent.py`)** — 08:30 ET:
- **Maturity gate (entries only):** a signal is `immature` until its baseline has 21 prior sessions (`base_count ≥ 21`; `min_periods=10` only lets `sent_z` *compute*, it does not make it trustworthy). **Never OPEN a position on an `immature` signal** — a z off a ~10-observation baseline is noise-dominated. This gate is **entry-only by design**: exits and the knife-guard are never gated on maturity, so a held name whose coverage regresses back to immature can always still be reduced or closed (risk-reducing actions are never blocked).
- **Enter** at 6% target weight when `not immature` AND `sent_z > +1.5` AND `attention_z > +1` AND evidence floor met: ≥ 5 relevant raw items, **or** ≥ 3 raw items + a top-20 ApeWisdom rank (aggregator corroboration) — never trade thin evidence
- **Exit** when `sent_z < 0` or 10 trading days elapsed (whichever first) — applies to held names regardless of maturity
- **Never-catch-knives rule**: `sent_z < −2` → do not buy the dip; flag only
- Hourly monitor: intraday flip to `sent_z < −2` on a held name → exit next cycle
- Max 6 concurrent positions (from `risk_limits.yaml: max_positions_sentinel`); rest cash. Rationale logged = top-5 contributing items (title, source, score, url) + component values + sources_used.

## Event overlay (Phase 7 nice-to-have)
Earnings calendar (Finnhub `/calendar/earnings`; API Ninjas calendar as cross-check with actual-vs-estimate EPS): freeze new entries 2 days pre-earnings (event risk ≠ sentiment edge); treat post-earnings sentiment+attention spikes as the highest-quality trigger.

## Backtest honesty
Free-tier news history is shallow (Finnhub ≈ 1 year) — but **Tradestie's `?date=` parameter provides historical daily WSB snapshots**: in Phase 1, backfill as far as it serves (respect 20 req/min) to give the WSB sentiment/attention legs real history. Everything else: backtest the *rules* on history accumulated from day one of our own ingestion (which starts in **Phase 1**, three phases before the agent trades). Third-party sentiment backfill inherits their model, not ours — say so in every SENTINEL report.

## Relevance validation (accumulating, not one-shot)
The Haiku filter is load-bearing, so the weekly spot-check **accumulates** rather than resetting: a 19/20 week proves very little on its own, and a one-shot 100–200-label study would only tell us the filter *was* right once, not that it *stays* right as the news mix drifts.

Each week, `scripts/relevance_review.py` draws a **stratified 10 predicted-YES + 10 predicted-NO** sample (stratified, not random: a random draw is ~58% NO and under-samples the YES class, which is the expensive failure). The reviewer labels it blind — **the model's verdict is hidden on the sheet** so the reviewer isn't anchored by it, and is recovered from `url_hash` at ingest.

Human labels persist to **`relevance_labels`**, never to `news_items.relevant`: the model's verdict is *history* and is never overwritten by a human (same never-edit-history rule as the rest of the schema). The PK is `(url_hash, reviewer)`, so two reviewers can independently label the same item.

The weekly report carries **cumulative YES-precision and NO-precision with Wilson 95% CIs** over every label ever collected. The two are tracked separately because they fail differently: a YES error lets noise reach FinBERT and move `sent_raw`; a NO error silently drops real signal before it can ever be traded. Wilson, not the normal approximation, because n is small and p is near 1 — exactly where the naive interval runs past 1.0 and implies a certainty we don't have (20/20 gives a lower bound near 0.84, not 1.0).

**Define relevance before scoring it.** Multi-company articles and administrative correction notices are useful boundary cases. Label them explicitly and measure how often a definition change affects the aggregate before adopting it.

## Failure modes
Ticker false positives (the Haiku filter is load-bearing — see the accumulating validation above); sarcasm on WSB (disagreement helps; accept noise); stale RSS duplicates (dedupe); **hobby-grade aggregator risk** (ApeWisdom/Tradestie have no SLA — freshness checks + renormalization make them droppable, and the calibration check catches silent drift); overfitting thresholds (1.5/1.0 are priors — grid-search only walk-forward on accumulated data, never the full sample at once).
