# Engineering notes

Trident separates signal generation, order authorization, and accounting. That separation made it possible to test failure modes offline and reuse the same risk and sizing logic in simulation.

## What the release demonstrates

| Area | Design decision | Evidence in the repo |
| --- | --- | --- |
| Backtesting | Limit historical price features to pre-session data. Fill at the session open with explicit slippage and mark holdings at the close. | [Golden equity and look-ahead tests](../tests/test_backtest.py) |
| QUANT | Compare a walk-forward model with simpler rules at multiple slippage assumptions. Keep the decision to promote a model separate from training. | [Baseline runner](../scripts/backtest_baselines.py), [model evaluation runner](../scripts/backtest_quant.py) |
| SENTINEL | Distinguish direct coverage from incidental mentions, deduplicate titles by ticker, and treat missing sources as missing evidence. | [Relevance filtering](../src/trident/agents/sentiment/relevance.py), [dedupe tests](../tests/test_news.py) |
| ORACLE | Treat an LLM thesis as a proposal. Check numeric claims against retrieved facts or recomputed formulas before sizing. | [Citation verifier](../src/trident/agents/analyst/citations.py), [citation tests](../tests/test_oracle_citations.py) |

Backtests illustrate a method for comparing strategies. The included demo uses synthetic prices and supplies no evidence of investment performance.

## Later design work

The following designs came from a later prototype. The portfolio release includes the standalone turnover calculation and tests. The storage and execution integrations described here belong to that prototype.

- **Decision provenance.** Assign each run and portfolio a stable identity. Record the effective risk policy and input versions alongside the decision, then link the decision to order attempts and fills. Store cash and positions as observations with explicit quality markers so later corrections can be traced to the original inputs.
- **Order recovery.** Persist intent before sending a request. A timeout leaves the broker outcome unknown, so reconcile by the same client order ID and record the outcome before retrying. An unresolved submission still consumes exposure. Reconcile positions and cash before resuming after a discrepancy.
- **Structured evidence.** Give the research agent a schema for numeric claims with exact pointers to source fields. Render the narrative from verified claims, including units and precision. Use synthetic conformance cases to check that a plausible number attached to the wrong field fails.
- **Atomic turnover.** Count gross fills and conservatively priced outstanding orders against a limit based on a prior canonical equity observation. Buys and sells do not net. [The calculation](../src/trident/risk/turnover.py) is pure and covered by [tests](../tests/test_turnover.py). A production caller would read the evidence and record the new commitment under one portfolio lock. An actual fill is still ledger truth even if it takes turnover over budget.
