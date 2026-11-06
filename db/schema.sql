-- TRIDENT schema. Money: NUMERIC(18,6). Time: timestamptz UTC. Never edit history rows;
-- corrections are new rows (see reconciliation runbook, doc 09).

CREATE TABLE IF NOT EXISTS agents (
    id           TEXT PRIMARY KEY,                 -- 'quant' | 'sentinel' | 'oracle' | 'bench'
    kind         TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'active',   -- active | halted | disabled
    start_cash   NUMERIC(18,6) NOT NULL,
    high_water_mark NUMERIC(18,6) NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO agents (id, kind, start_cash) VALUES
    ('quant',    'statistical_ml',   30000),
    ('sentinel', 'media_sentiment',  30000),
    ('oracle',   'llm_fundamental',  30000),
    ('bench',    'spy_buy_and_hold', 10000)
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS decisions (
    id            TEXT PRIMARY KEY,
    agent_id      TEXT NOT NULL REFERENCES agents(id),
    asof          TIMESTAMPTZ NOT NULL,
    ticker        TEXT NOT NULL,
    target_weight NUMERIC(8,6) NOT NULL,
    horizon_days  INT NOT NULL,
    rationale     JSONB NOT NULL,                  -- golden rule 2: never empty
    inputs_digest TEXT NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_decisions_agent_ts ON decisions (agent_id, asof DESC);

CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,
    decision_id     TEXT REFERENCES decisions(id),
    agent_id        TEXT NOT NULL REFERENCES agents(id),
    client_order_id TEXT NOT NULL UNIQUE,          -- '{agent}-{uuid12}': attribution key
    broker_order_id TEXT,
    ticker          TEXT NOT NULL,
    side            TEXT NOT NULL,
    qty             NUMERIC(18,6) NOT NULL,
    limit_price     NUMERIC(18,6),
    status          TEXT NOT NULL,
    risk_notes      JSONB NOT NULL DEFAULT '[]',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_orders_agent_ts ON orders (agent_id, created_at DESC);

CREATE TABLE IF NOT EXISTS fills (
    id        BIGSERIAL PRIMARY KEY,
    order_id  TEXT NOT NULL,                       -- client_order_id
    agent_id  TEXT NOT NULL REFERENCES agents(id),
    ticker    TEXT NOT NULL,
    side      TEXT NOT NULL,
    qty       NUMERIC(18,6) NOT NULL,
    price     NUMERIC(18,6) NOT NULL,
    fees      NUMERIC(18,6) NOT NULL DEFAULT 0,
    realized_pnl NUMERIC(18,6),
    ts        TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_fills_agent_ts ON fills (agent_id, ts DESC);

CREATE TABLE IF NOT EXISTS positions (
    agent_id  TEXT NOT NULL REFERENCES agents(id),
    ticker    TEXT NOT NULL,
    qty       NUMERIC(18,6) NOT NULL,
    avg_cost  NUMERIC(18,6) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (agent_id, ticker)
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    agent_id  TEXT NOT NULL REFERENCES agents(id),
    ts        TIMESTAMPTZ NOT NULL,
    equity    NUMERIC(18,6) NOT NULL,
    cash      NUMERIC(18,6) NOT NULL,
    PRIMARY KEY (agent_id, ts)
);

CREATE TABLE IF NOT EXISTS signals (
    id       BIGSERIAL PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id),
    ts       TIMESTAMPTZ NOT NULL,
    ticker   TEXT NOT NULL,
    score    NUMERIC(8,6) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_signals_agent_ticker_ts ON signals (agent_id, ticker, ts DESC);

CREATE TABLE IF NOT EXISTS news_items (
    url_hash     TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    ticker       TEXT,
    title        TEXT NOT NULL,
    body         TEXT,
    url          TEXT NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    relevant     BOOLEAN,                          -- Haiku filter verdict (null = unscored)
    sentiment    NUMERIC(8,6)                      -- FinBERT p(pos)-p(neg)
);
CREATE INDEX IF NOT EXISTS ix_news_ticker_ts ON news_items (ticker, published_at DESC);

CREATE TABLE IF NOT EXISTS theses (
    id            BIGSERIAL PRIMARY KEY,
    asof          TIMESTAMPTZ NOT NULL,
    ticker        TEXT NOT NULL,
    rating        TEXT NOT NULL,
    conviction    NUMERIC(4,3) NOT NULL,
    target_weight NUMERIC(8,6) NOT NULL,
    thesis        TEXT NOT NULL,
    key_risks     JSONB NOT NULL,
    invalidation_triggers JSONB NOT NULL,
    horizon_days  INT NOT NULL,
    tool_trace    JSONB
);
CREATE INDEX IF NOT EXISTS ix_theses_ticker_ts ON theses (ticker, asof DESC);
-- Phase 5 (ORACLE) additive migration. agent_id has NO DEFAULT: a writer that forgets it fails
-- loud rather than being silently mislabelled 'oracle' (persistence.upsert_thesis sets it). The
-- unique index makes re-runs upsert (asof is the session key — orchestration.oracle._session_asof
-- stamps it to midnight UTC so a re-run collides). Idempotent; on a cold init the CREATE TABLE
-- above runs first, then this ALTER adds agent_id NOT NULL to the just-created empty table.
ALTER TABLE theses ADD COLUMN IF NOT EXISTS agent_id TEXT NOT NULL REFERENCES agents(id);
ALTER TABLE theses ADD COLUMN IF NOT EXISTS citations         JSONB NOT NULL DEFAULT '[]';
ALTER TABLE theses ADD COLUMN IF NOT EXISTS evidence_basis    JSONB NOT NULL DEFAULT '[]';
ALTER TABLE theses ADD COLUMN IF NOT EXISTS narrative_summary TEXT;  -- Phase 5.1 seam (nullable)
CREATE UNIQUE INDEX IF NOT EXISTS ux_theses_agent_ticker_asof ON theses (agent_id, ticker, asof);

CREATE TABLE IF NOT EXISTS model_registry (
    id         BIGSERIAL PRIMARY KEY,
    agent_id   TEXT NOT NULL REFERENCES agents(id),
    trained_at TIMESTAMPTZ NOT NULL,
    train_window TEXT NOT NULL,                    -- e.g. '2021-01..2024-12' (was `window`: reserved word)
    metrics    JSONB NOT NULL,                     -- auc, val_sharpe, feature importances
    artifact_path TEXT NOT NULL,
    active     BOOLEAN NOT NULL DEFAULT false
);

-- Human adjudication of the SENTINEL relevance filter (doc 04 weekly spot-check).
-- Deliberately SEPARATE from news_items.relevant: that column is the MODEL's verdict and is
-- history — it is never overwritten by a human label (same never-edit-history rule as the rest
-- of the schema). Ground truth accumulates here instead, so YES/NO precision can be tracked
-- with confidence intervals over time.
-- PK is (url_hash, reviewer): two reviewers may independently label the same item, which is
-- exactly how the 2026-07-13 sample was adjudicated.
CREATE TABLE IF NOT EXISTS relevance_labels (
    url_hash      TEXT NOT NULL REFERENCES news_items(url_hash),
    reviewer      TEXT NOT NULL,
    label         BOOLEAN NOT NULL,                -- the HUMAN verdict (ground truth)
    model_verdict BOOLEAN NOT NULL,                -- snapshot of news_items.relevant at review
                                                   -- time, so precision stays computable even if
                                                   -- the model is later re-run
    note          TEXT,
    sample_id     TEXT NOT NULL,                   -- which weekly batch this came from
    labeled_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (url_hash, reviewer)
);
CREATE INDEX IF NOT EXISTS ix_relevance_labels_sample ON relevance_labels (sample_id);

-- Cross-ticker URL collisions (observational; feeds the pre-registered 2026-07-22 attribution
-- check). news_items.url_hash keys on the URL ALONE, so a multi-company article attaches to
-- whichever ticker was ingested FIRST — later tickers hit ON CONFLICT DO NOTHING and their
-- (url, ticker) pairing is silently discarded. This table captures those discarded pairings
-- PROSPECTIVELY, so we can quantify the attribution bias instead of guessing it. It never
-- alters news_items; it only records "this URL, already stored under A, also arrived as B".
CREATE TABLE IF NOT EXISTS news_url_collisions (
    id               BIGSERIAL PRIMARY KEY,
    url_hash         TEXT NOT NULL,
    stored_ticker    TEXT,                          -- ticker the URL is stored under (won)
    attempted_ticker TEXT,                          -- ticker that arrived later (discarded)
    attempted_source TEXT,
    seen_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_news_collisions_hash ON news_url_collisions (url_hash);

CREATE TABLE IF NOT EXISTS health_events (
    id       BIGSERIAL PRIMARY KEY,
    ts       TIMESTAMPTZ NOT NULL DEFAULT now(),
    severity TEXT NOT NULL,                        -- info | warn | critical
    source   TEXT NOT NULL,
    detail   TEXT NOT NULL,
    cleared  BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS ix_health_open ON health_events (severity, cleared, ts DESC);
