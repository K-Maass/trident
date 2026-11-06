# 06 — RISK ENGINE & EXECUTION

The layer that makes live trading eventually sane. Agents *propose* target weights; this layer decides what actually happens. All parameters live in `config/risk_limits.yaml` — code contains no magic numbers.

## Risk engine checks (risk/engine.py — order matters)

Per-decision (reject or clamp, each with a machine-readable reason logged into `orders.risk_notes`):
1. **Instrument allowlist**: ticker ∈ universe.yaml; equities only.
2. **Side**: long-only v1 — no shorts, no leverage. Target weights ∈ [0, max_position_weight].
3. **Position cap**: single name ≤ 10% of that agent's book (ORACLE 12%).
4. **Gross exposure cap**: post-trade Σ|weights| ≤ 100%; scale all buys down pro-rata if breached.
5. **Liquidity**: 21d avg dollar volume ≥ $50M AND order ≤ 1% of ADV (irrelevant at paper size; **critical** discipline for live).
6. **Price sanity**: last trade within 10% of previous close, else reject (bad-print guard); price ≥ $5.
7. **Turnover budget**: agent's daily traded value ≤ 50% of its book.
8. **Market state**: exchange open (or opening within grace window), ticker not halted, not within 2 min of close.

Book-level (state machine, persisted in `agents.status`):
9. **Drawdown kill-switch**: agent equity < 85% of its high-water mark → status `halted`: flatten all its positions next session, refuse new decisions, alert CRITICAL. Reactivation is **manual only** (human sets status back after review).
10. **Global freeze**: `health_events` CRITICAL in last 24h (reconciliation break, repeated data failures) → no agent trades.

## Execution service (execution/service.py)

- Sizing: `qty = floor(target_Δweight × agent_equity / last_price)` (whole shares v1; Alpaca supports fractional — v1.1 option).
- Order type: **limit at last price + 10 bps buffer**, `time_in_force=DAY` (marketable limit ≈ market order with a bad-fill cap). Sells symmetric.
- Tag: `client_order_id = f"{agent}-{uuid4().hex[:12]}"` — the attribution key for the ledger.
- Lifecycle: submit → poll 30s → partial fills applied as they land → unfilled at EOD auto-expires; remainder is simply re-decided tomorrow (no chasing).
- Idempotency: before submitting, check no open order with same agent+ticker+side (crash-restart safety).
- Failure: broker reject → log, health WARN, skip. Never auto-retry a *reject* (only network errors retry).

## Reconciliation (nightly, 16:20 ET)

Broker positions == Σ ledger positions per ticker; broker equity == Σ agent equity (± unsettled). Any breach: CRITICAL event + global freeze until manually cleared. This job is the canary for every accounting bug — build it early (Phase 2), thank yourself later.

## Trading mode

See [trading mode](10_GO_LIVE.md) for the guard on live credentials. The demo uses the simulator and paper mode is the default.
