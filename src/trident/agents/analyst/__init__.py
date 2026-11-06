"""ORACLE agent package (doc 05) — the LLM fundamental analyst.

Division of labour (the two governing rules, CLAUDE.md):
  - Anything expressible as a formula is deterministic Python (screen.py ranks the table,
    portfolio.py sizes positions). The LLM is used only where interpretive judgment adds
    information over arithmetic — reading financials into a thesis.
  - The agent (agent.py) is a PURE function of (context + extras). All LLM/graph/network work
    lives in orchestration/oracle.py; theses reach the agent through ctx.extras so the runner's
    inputs_digest covers them.

Pipeline: screen (deterministic rank) -> research (ReAct tool loop) -> thesis (structured +
citation-checked) -> portfolio (pure sizing) = the agent.
"""

ORACLE_V1 = "oracle_llm_v1"


def oracle_agent(**kwargs):
    """The standing ORACLE v1 agent instance (lazy import so data-only modules can load the
    tools without pulling in the LLM stack)."""
    from trident.agents.analyst.agent import OracleAgent

    return OracleAgent(**kwargs)
