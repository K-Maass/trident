# 05 — ORACLE (LLM fundamental analyst)

## Thesis
An LLM with disciplined tool access to primary sources (filings, financials, ratios) can do competent junior-analyst work: form theses, compare valuation to quality, size positions with stated invalidation triggers. The edge (if any) is patience + breadth; the horizon is months, so it trades weekly.

## Architecture: LangGraph graph (agents/analyst/graph.py)

```
screen ──▶ research (per candidate, tool loop) ──▶ thesis ──▶ portfolio ──▶ decisions
```

- **screen** (cheap, weekly): rank universe by simple value+quality composite (earnings yield, ROIC proxy, revenue growth, FCF margin from cached fundamentals) → top ~10 candidates + all currently-held names (always re-underwrite holdings).
- **research** (per candidate): ReAct-style tool loop, `claude-sonnet-4-6`, temperature 0, max 12 tool calls/name. Tools (`tools.py`, all read from the data layer/cache — the LLM never hits raw internet):
  - `get_price_history(ticker, period)` — returns + drawdown context
  - `get_financials(ticker)` — 5y income/balance/cashflow from EDGAR companyfacts
  - `get_ratios(ticker)` — P/E, EV/EBITDA, FCF yield, margins, debt/EBITDA, vs 5y own history
  - `get_filing_sections(ticker, form, sections)` — latest 10-K/10-Q MD&A + Risk Factors (chunked; Haiku pre-summarizes to ~1.5k tokens per section, cached)
  - `get_recent_news(ticker, n)` — SENTINEL's ingested items (agents may share data, never signals)
  - `compare_peers(ticker)` — same ratios across universe sector peers
  - `get_insider_activity(ticker)` *(optional, Phase 7)* — Form 4 buys/sells last 90d via EDGAR (free; API Ninjas insider endpoint as pre-parsed alternative)
- **thesis** node: emit **structured `InvestmentThesis`** (models.py): `rating` (strong_buy/buy/hold/avoid), `conviction` 0–1, `target_weight` ≤ 12%, `thesis` ≤ 200 words, `key_risks` (3), `invalidation_triggers` (explicit, checkable — e.g. "gross margin < 38% next quarter"), `horizon_days`. Persist full text + tool trace to `theses`.
- **portfolio** node: deterministic Python, not LLM — combine theses: weights ∝ conviction·rating, cap 12%/name, ≤ 8 names, cash is the default. Diff vs current book → `Decision`s. Trade only if |Δweight| > 2% (fundamentals don't need 30bp fiddling).

## Prompting rules
System prompt pins: value+quality discipline, "no thesis without numbers from tools," cite which tool result supports each claim, explicit disconfirming-evidence section, and *"cash is an acceptable answer."* Ban: price targets without valuation math, momentum reasoning (that's QUANT's job — keeps the agents' edges distinct).

## Event triggers (Phase 7)
Between weekly runs: 8-K filing or earnings release on a held name → re-underwrite just that name next morning. Invalidation-trigger checker: nightly job evaluates each holding's stored triggers against fresh data; a fired trigger forces re-research and is flagged in the report (this is the mechanism that stops thesis drift).

## Cost control
Weekly full run ≈ 10 deep names × ~40k tokens ≈ manageable single-digit $ per run on Sonnet; screening and summarization on Haiku. Log tokens per run to `health_events(source='oracle_cost')`; monthly budget alarm at $25.

## Study list
- `virattt/ai-hedge-fund` — persona-based LLM investor agents (closest prior art; note it's backtest-oriented, ORACLE adds live paper loop)
- `TauricResearch/TradingAgents` — multi-agent debate structure (v2 idea: add a bear-case critic node)
- Damodaran's valuation course materials — for the ratio sanity the prompts should encode

## Failure modes
Hallucinated numbers (tools-only rule + require citations in thesis text), stale fundamentals (freshness check: companyfacts < 100 days old else WARN), overtrading on rewrites (2% trade threshold), verbosity drift (thesis capped 200 words — schema-enforced), position creep (portfolio node is code, so caps are hard).
