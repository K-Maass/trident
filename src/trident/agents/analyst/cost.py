"""ORACLE cost metering (doc 09). One accountant for every LLM call — so "from the very first
API call" is enforceable: there is no path to messages.create that skips this (see llm.py).

Dollar figures are ESTIMATES for the monthly budget governor and the per-run health event, not
billing. Rates are per-million-tokens keyed by model TIER (substring match) so a model-string
drift can't silently zero the meter. Cache reads bill ~0.1x input; cache writes ~1.25x (5-min
ephemeral). Sources: Claude API pricing (Sonnet 4.6 $3/$15, Haiku 4.5 $1/$5 per MTok).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# tier -> (input, output, cache_write, cache_read) USD per 1M tokens. First substring hit wins.
_PRICING: list[tuple[str, tuple[float, float, float, float]]] = [
    ("haiku", (1.00, 5.00, 1.25, 0.10)),
    ("sonnet", (3.00, 15.00, 3.75, 0.30)),
    ("opus", (5.00, 25.00, 6.25, 0.50)),
]
# Unknown tier -> price as the most expensive tier. Erring expensive makes the budget governor
# trip EARLY on an unrecognised model rather than under-count real spend.
_FALLBACK = (5.00, 25.00, 6.25, 0.50)


def _rates(model: str) -> tuple[float, float, float, float]:
    m = model.lower()
    for tier, rates in _PRICING:
        if tier in m:
            return rates
    return _FALLBACK


@dataclass
class _ModelTally:
    calls: int = 0
    input: int = 0
    output: int = 0
    cache_write: int = 0
    cache_read: int = 0


@dataclass
class CostMeter:
    """Accumulates token usage per model across a run and estimates the dollar cost."""

    by_model: dict[str, _ModelTally] = field(default_factory=dict)

    def record(self, model: str, usage) -> None:
        """Fold one response's `usage` into the tally. Tolerant of absent cache_* fields."""
        t = self.by_model.setdefault(model, _ModelTally())
        t.calls += 1
        t.input += int(getattr(usage, "input_tokens", 0) or 0)
        t.output += int(getattr(usage, "output_tokens", 0) or 0)
        t.cache_write += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        t.cache_read += int(getattr(usage, "cache_read_input_tokens", 0) or 0)

    @property
    def calls(self) -> int:
        return sum(t.calls for t in self.by_model.values())

    def dollars(self) -> float:
        total = 0.0
        for model, t in self.by_model.items():
            ci, co, cw, cr = _rates(model)
            total += (t.input * ci + t.output * co + t.cache_write * cw + t.cache_read * cr) / 1e6
        return round(total, 4)

    def summary(self) -> dict:
        """JSON-serializable payload for the `oracle_cost` health event detail."""
        return {
            "calls": self.calls,
            "estimated_usd": self.dollars(),
            "by_model": {
                model: {
                    "calls": t.calls,
                    "input_tokens": t.input,
                    "output_tokens": t.output,
                    "cache_write_tokens": t.cache_write,
                    "cache_read_tokens": t.cache_read,
                }
                for model, t in self.by_model.items()
            },
        }
