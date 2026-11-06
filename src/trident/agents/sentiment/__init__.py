"""SENTINEL agent package (doc 04) — media-context trader.

**v1 = `sentinel_rules_v1`** (agent.SentinelAgent): a rule agent over per-ticker sentiment
z-scores + attention surges (the ML/threshold-tuned version is a later revision, only on mature
accumulated data). The pipeline modules are: relevance (Haiku filter) → score (FinBERT) →
aggregate (sent_z / attention_z / disagreement) → agent (this package's decision rules).
"""

from trident.agents.sentiment.agent import SentinelAgent

SENTINEL_V1 = "sentinel_rules_v1"


def sentinel_agent(**kwargs) -> SentinelAgent:
    """The standing SENTINEL v1 agent instance."""
    return SentinelAgent(**kwargs)
