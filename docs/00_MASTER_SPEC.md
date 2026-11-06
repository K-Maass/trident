# 00 — Original design brief

This design brief includes planned work. See the [README](../README.md) for the released surface.

## 1. Goal

Build three autonomous investing agents with fundamentally different information edges, run them in a **simulated ("paper") environment with fake money but real market data**, measure them honestly against each other and against SPY buy-and-hold, and design the system so that a validated agent can be **promoted to live trading by flipping configuration** — no rewrite.

Success looks like: three equity curves updating daily, every trade explainable from a logged rationale, a weekly report comparing risk-adjusted performance, and a documented, gated path to real money.

## 2. The three agents

| | QUANT | SENTINEL | ORACLE |
|---|---|---|---|
| Information edge | Price/volume statistics | Media context (news, Reddit, StockTwits, RSS) | Fundamentals (filings, financials, ratios) |
| Brain | Feature pipeline → LightGBM classifier ranking next-5-day returns. **Zero LLM.** | FinBERT sentiment scoring + LLM relevance filter → per-ticker sentiment index → trade on shifts | LangGraph agent with data tools → structured investment thesis → target weights |
| Horizon | 1–10 days | 1–15 days | 1–6 months |
| Cadence | Daily at 09:35 ET | Daily 08:30 ET + hourly monitor | Weekly (Mon 08:00 ET), event-triggered on earnings |
| Detail spec | `03_AGENT_QUANT.md` | `04_AGENT_SENTIMENT.md` | `05_AGENT_ANALYST.md` |

A fourth implicit "agent" is **BENCH**: 100% SPY, bought on day one, never touched. Every metric is reported relative to it.

## 3. System diagram

```
                    ┌──────────────── DATA LAYER ────────────────┐
                    │ Market bars (Alpaca/yfinance)  News (Finnhub│
                    │ /RSS/Reddit)  Fundamentals (EDGAR/yfinance) │
                    │ All cached → Postgres + parquet             │
                    └───────┬───────────────┬──────────────┬─────┘
                            ▼               ▼              ▼
                       ┌────────┐     ┌──────────┐   ┌─────────┐
                       │ QUANT  │     │ SENTINEL │   │ ORACLE  │
                       └───┬────┘     └────┬─────┘   └────┬────┘
                           │  Decision(ticker, target_weight, rationale)
                           ▼               ▼              ▼
                    ┌─────────────────────────────────────────────┐
                    │ RISK ENGINE  (position caps, exposure caps, │
                    │ liquidity filter, drawdown kill-switch)     │
                    └───────────────────┬─────────────────────────┘
                                        ▼ approved orders (tagged by agent)
                    ┌─────────────────────────────────────────────┐
                    │ EXECUTION SERVICE → Broker (Alpaca paper/live)│
                    └───────────────────┬─────────────────────────┘
                                        ▼ fills, dividends, marks
                    ┌─────────────────────────────────────────────┐
                    │ LEDGER: one virtual portfolio per agent      │
                    │ (cash, positions, equity) inside ONE broker  │
                    │ account. Source of truth for per-agent P&L.  │
                    └───────────────────┬─────────────────────────┘
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │ POSTGRES: decisions, orders, fills, signals, │
                    │ equity snapshots, health events              │
                    └───────┬─────────────────────────┬───────────┘
                            ▼                         ▼
                     EVALUATION (metrics,       DASHBOARD (Streamlit:
                     weekly reports, alerts)    3 curves vs SPY)
```

## 4. Core design decisions (and why)

1. **One Alpaca paper account, three virtual sub-portfolios.** Alpaca gives one paper account per login. Instead of juggling three accounts, each agent gets a virtual ledger (starting cash $30,000 each; $10,000 reserved for BENCH/SPY). Orders carry `client_order_id = "{agent}-{uuid}"` so every fill attributes exactly to one agent. Bonus: this ledger is *exactly* what's needed to later run multiple agents in one live account, so the work compounds.
2. **Broker abstraction (`broker/base.py`).** Agents and execution code only see the `Broker` interface. `AlpacaBroker(paper=True)` and `AlpacaBroker(paper=False)` are the same class — Alpaca's paper API is endpoint-identical to live. Promotion = keys + flag.
3. **Agents propose, the risk engine disposes.** Agents output `Decision`s (target weights + rationale), never orders. The risk engine converts to orders, clamps, or rejects. This is the safety boundary that makes eventual live trading sane.
4. **Everything logged with rationale.** `decisions.rationale` is a JSONB blob: for QUANT the top feature contributions, for SENTINEL the headlines and scores, for ORACLE the full thesis. This is what makes the comparison *interesting*, not just numbers.
5. **Backtest → paper → live pipeline.** Paper trading burns wall-clock time (a month of paper = a month of your life), so iterate in backtests first; only strategies that survive the backtest protocol get paper-deployed; only paper survivors reach the go-live gate (`10_GO_LIVE.md`).
6. **Daily/weekly cadence, long-only, no leverage (v1).** Avoids the US pattern-day-trader rule (<$25k accounts), fits free data tiers, keeps LLM costs ~$10–30/month, and keeps risk legible.

## 5. Universe

Start: ~40 liquid US large caps + SPY (see `config/universe.yaml`). Criteria: price > $5, avg daily dollar volume > $50M, listed > 2 years. Expand to S&P 100 in Phase 7 if data budgets allow. All agents share the universe so comparison is apples-to-apples.

## 6. Tools

Python modules separate signal generation, risk checks, execution, and accounting. PostgreSQL stores paper-mode state and the offline backtester uses an in-memory ledger. See the [architecture notes](01_ARCHITECTURE.md).

## 7. Evaluation (summary — full spec in `07_EVALUATION.md`)

Daily equity snapshots per agent → total return, CAGR, Sharpe, Sortino, max drawdown, Calmar, hit rate, turnover, alpha/beta vs SPY. Weekly Markdown+Telegram report. Honesty clause: with <1 year of paper data, Sharpe differences are rarely statistically significant — the report always prints the standard error.

## 8. Path to real money (summary — full spec in `10_GO_LIVE.md`)

Gates: ≥60 trading days paper, positive alpha vs SPY, drawdown within limits, 30 incident-free days, reconciliation clean, kill-switch fire drill passed, manual arming of `config/live.yaml`, live keys added by the operator only, starting capital ≤ CHF 1,000 per promoted agent. Swiss-residency broker notes included there.

## 9. Out of scope (v1)

Options/futures/crypto, shorting, intraday HFT, margin, auto-promotion to live, portfolio optimization across agents (each agent is siloed on purpose — we're comparing them).
