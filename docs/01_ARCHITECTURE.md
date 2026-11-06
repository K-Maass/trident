# Architecture

## Signal and execution path

1. **Inputs.** Market prices, a book snapshot, and agent-specific evidence enter a [market context](../src/trident/agents/base.py). QUANT uses price features, SENTINEL uses aggregated sentiment, and ORACLE uses verified theses.
2. **Decision.** Each strategy produces target weights with rationale. The [paper runner](../src/trident/orchestration/runner.py) logs the decisions before sending them for review.
3. **Risk.** The [risk engine](../src/trident/risk/engine.py) checks the universe, liquidity, exposure, and trading constraints. It returns a rejection or an approved target, which may be resized.
4. **Execution.** The [execution service](../src/trident/execution/service.py) sizes approved targets, records orders, calls the broker, and applies confirmed fills. The [broker interface](../src/trident/broker/base.py) separates the Alpaca adapter from simulated execution.
5. **Accounting.** The [ledger](../src/trident/ledger/book.py) updates cash and holdings from fills. The paper runner stores state and can compare combined virtual books with the broker account.

## Simulation

The [backtester](../src/trident/evaluation/backtest.py) uses the same risk engine, order sizing, and ledger math in memory. Historical price features stop at the previous session. Held positions are marked at the current open before building the strategy’s book snapshot. The simulator fills at the current open with explicit slippage and marks the book at the close. The [golden test](../tests/test_backtest.py) checks the resulting equity against a hand calculation.

## Storage and configuration

The [Postgres schema](../db/schema.sql) covers decisions, orders, fills, positions, equity snapshots, and health events, plus agent-specific inputs and research outputs. YAML files in [config](../config) define the trading universe, risk limits, sources, and planned cadence. Optional provider credentials are read through local environment settings.

The release focuses on offline evaluation and the paper-trading components. See [engineering notes](08_ROADMAP.md) for the later designs around provenance, recovery, and turnover authorization.
