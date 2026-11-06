# 07 — EVALUATION

The point of the whole project: an honest, like-for-like comparison of three information edges against each other and against doing nothing (SPY).

## Data
`equity_snapshots(agent_id, ts, equity, cash)` daily at close for QUANT, SENTINEL, ORACLE, BENCH. Daily simple returns r_t = E_t/E_{t−1} − 1. Risk-free: FRED DGS3MO/252 (fallback 0, flagged in report).

## Metrics (evaluation/metrics.py — pure functions, unit-tested against hand-computed fixtures)

- Total return, **CAGR** = (E_T/E_0)^(252/T) − 1
- **Sharpe** = mean(r − rf)/std(r − rf) × √252  — plus its standard error (below)
- **Sortino** (downside std), **Max drawdown** = min(E_t/max_{s≤t}E_s − 1), **Calmar** = CAGR/|MaxDD|
- **Alpha/Beta**: OLS r_agent − rf = α + β(r_SPY − rf); report annualized α with t-stat
- **Hit rate** (positive days and per-round-trip wins), avg win/avg loss
- **Turnover** = Σ|traded value| / avg equity, annualized; **exposure** = avg invested fraction
- Per-agent extras: QUANT — live AUC vs training AUC (decay detector); SENTINEL — signal→return IC; ORACLE — thesis outcome vs rating, invalidation-trigger hit log

## Statistical honesty (print in every report)
SE(Sharpe) ≈ √((1 + SR²/2)/T_years). With 60 trading days (0.24y), SE ≈ 2.1 — i.e. **a measured Sharpe of 1.0 is indistinguishable from 0.** Conclusions before ~1 year are directional at best. The report must show SR ± SE and refuse the word "significant" unless |SR|/SE > 2. Also: paper fills are optimistic (no queue, no impact) — treat paper Sharpe as an upper bound.

## Reports
- **Daily digest** (Telegram, 16:25 ET): per agent — day P&L, equity, top mover, decisions made (one-line rationale), health events.
- **Weekly report** (Saturday, Markdown → `artifacts/reports/YYYY-WW.md` + Telegram summary): full metric table (agents + BENCH), equity & drawdown charts, best/worst decision of the week *with rationale*, per-agent extras, open risk items.
- **Promotion review** (on demand): the 10_GO_LIVE.md gate checklist auto-filled from DB.

## Comparison rules (pre-registered — no moving goalposts later)
1. Identical universe, costs, start date, and starting capital ($30k each).
2. Primary metric: **Sharpe vs BENCH** with SE; tie-breakers: MaxDD, then Calmar.
3. No parameter changes mid-evaluation window; any change resets that agent's window (log it in `model_registry`/git).
4. Minimum window before *any* ranking claim: 60 trading days. Before promotion talk: also see 10_GO_LIVE.md.

## Backtest ↔ paper consistency check
Once monthly: replay the paper period through `evaluation/backtest.py` with the same decisions log; fill prices should match within slippage assumptions. Divergence > 20 bps/month cumulative → investigate the simulator or the cost model. This closes the loop that keeps backtests trustworthy.
