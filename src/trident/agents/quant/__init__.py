"""QUANT agent package.

**v1 (shipped 2026-07-10) = the 12-1 momentum baseline.** On the walk-forward OUT-OF-SAMPLE
go/no-go (2020–2026, backtest-auditor PASS), momentum beat the LightGBM ranker: Sharpe
1.23/1.23 vs 1.17/1.01 (5/10 bps) — the ML did not clear momentum by the +0.10 bar, and
momentum is far more cost-robust (584 vs 4528 trades). Per the binding decision rule in
docs/08_ROADMAP.md, the baseline ships. The ML pipeline (features/model/train/agent) is kept
for v2 re-evaluation on FUTURE data, not re-tuned against the same OOS window.
"""

from trident.agents.quant.baselines import MomentumBaseline

QUANT_V1 = "momentum_12_1"


def quant_agent(**kwargs) -> MomentumBaseline:
    """The standing QUANT v1 agent instance. `force_rebalance=True` only for the one-time
    initial entry at deploy (see MomentumBaseline)."""
    return MomentumBaseline(**kwargs)
