# 03 — QUANT (statistical/ML agent, no LLM)

## Thesis
Short-horizon cross-sectional return differences contain weak but exploitable structure (momentum, mean-reversion, volatility regimes). A gradient-boosted tree over engineered features can rank tomorrow's relative winners slightly better than chance. Edge is tiny → discipline (validation + costs) is everything.

## v0 baselines (build FIRST, keep forever)
1. **MA-cross**: long ticker when 20d SMA > 100d SMA, equal-weight all longs, else cash.
2. **12-1 momentum**: monthly, long top-8 by trailing 12-month return skipping last month.
3. **RSI-2 mean reversion**: buy RSI(2)<10 in uptrend (close>200d SMA), exit RSI(2)>70.
4. **Markov regime rule** (long-only adaptation of the observable-Markov approach): label each ticker Bull/Bear/Sideways by 20d rolling return with a ±2% threshold — **fixed a priori; baselines don't get tuned** — estimate the 3×3 transition matrix on trailing data only (re-estimated walk-forward), hold when P(Bull|state) − P(Bear|state) > 0, else cash. Caveat: consecutive labels share 19 of 20 window days, so the impressive persistence diagonal is partly an overlapping-window artifact — this baseline is ~20d momentum in a Markov costume; don't be seduced by the matrix.

These are the bar the ML must beat *after costs* in identical backtests. If it can't, ship the best baseline as QUANT v1 — an honest result, not a failure.

## v1 ML pipeline

**Target**: cross-sectional — each day, label each ticker by whether its next-5-day return is in the top tercile of the universe (classification, `P(top tercile)`). Cross-sectional ranking is more robust than absolute up/down prediction and self-normalizes market regime.

**Features** (per ticker, per day — `agents/quant/features.py`, all lag-safe):
- Returns: 1d, 5d, 21d, 63d, 126d log returns; 21d return z-scored vs universe
- Mean-reversion: RSI(2), RSI(14), distance from 20d SMA in ATR units
- Trend: SMA20/SMA100 ratio, 63d linear-regression slope (normalized)
- Volatility: 21d realized vol, vol-of-vol, ATR(14)/price
- Volume: 5d/63d dollar-volume ratio (z), on-balance-volume slope
- Calendar: day-of-week, days-to-month-end
- Cross-asset: SPY 5d return, VIX proxy (SPY 21d realized vol) — regime context
- Regime: SPY Markov state one-hot (Bull/Bear/Sideways from the baseline-4 labeler) + days-in-current-regime — lag-safe by construction (trailing window only)

**Model**: LightGBM (`binary`, ~500 trees, lr 0.03, max_depth 4–6, feature_fraction 0.8, early stopping on validation). Log feature importances into `model_registry.metrics`.

**Training protocol — walk-forward with purge & embargo (non-negotiable)**:
- Train window 3y → validate 6m → test 6m; roll forward 6m; 2015→present.
- **Purge** overlapping-label rows at each boundary and **embargo** 5 further days (labels span 5 days — without this, leakage will fabricate a beautiful fake Sharpe). Concepts from López de Prado, *Advances in Financial Machine Learning* ch. 7 — worth reading.
- Retrain in production: monthly (1st Saturday), walk the window forward, register artifact in `model_registry`, hot-swap only if validation AUC ≥ current − 0.01.

**Signal → Decision**: daily, score universe → rank → target: equal-weight the top-8 names whose score clears the base-rate floor, else proportionally more cash. Max single-name weight from `risk_limits.yaml`. Turnover damping: only trade if |target − current| > 1% of book. Rationale logged = top-5 SHAP-style feature contributions per picked name.

> **Threshold correction (Phase 3, 2026-07-10).** This originally read "score > 0.55". Because the label is the **top tercile** (base rate ⅓), `P(top tercile)` centers on ~0.33 (observed 0.30–0.38), so 0.55 is unreachable and the agent would never trade. The operative floor is therefore the **⅓ base rate** ("above chance"), matching `agents/quant/agent.py:SCORE_THRESHOLD`. This was set from the score *distribution*, not from backtest returns. Sensitivity: the ML's OOS loss (Sharpe 0.43 vs momentum 1.23) is driven by weak ranking quality (pooled OOS AUC 0.533), not the threshold — no floor rescues a near-random ranking, so QUANT v1 ships as the momentum baseline (see roadmap build log).

## Backtesting
`vectorbt` for fast baseline sweeps; the event-driven `evaluation/backtest.py` (same risk+execution code as paper) for anything you intend to deploy. Costs model: 5 bps slippage + $0 commission. See the [evaluation design](07_EVALUATION.md).

## Study list (pipelines to borrow from, not fork)
- `microsoft/qlib` — full quant ML platform; steal their feature/label alignment patterns
- `AI4Finance-Foundation/FinRL` — RL framing (v2 idea, not v1)
- `polakowo/vectorbt` — vectorized backtests
- Kaggle "Jane Street Market Prediction" & "Optiver" solutions — feature-engineering gold

## Failure modes to watch
Leakage (purge/embargo!), overfit hyperparams (touch test set once), survivorship bias (universe is today's list — note this limitation in reports; acceptable for v1, fix with point-in-time universe in v2), regime breaks (2022-style) — report per-year metrics, never just the aggregate.
