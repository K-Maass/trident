"""Accumulating validation of the SENTINEL relevance filter (doc 04 weekly spot-check).

The one-shot 20-verdict check told us the filter is roughly right; it cannot tell us whether it
STAYS right as the news mix drifts. So the weekly check accumulates instead of resetting: each
week we draw a stratified sample (N predicted-YES + N predicted-NO), a human labels it, the
labels are persisted, and precision is reported over ALL labels ever collected — with Wilson
confidence intervals, so a 19/20 week is never mistaken for proof.

Two precisions, because they fail differently and cost differently:
  YES-precision  = of the items the model called relevant, how many really are.
                   Failure = noise reaching FinBERT and moving sent_raw. This is the expensive one.
  NO-precision   = of the items the model rejected, how many really were irrelevant.
                   Failure = real signal silently dropped before it can ever be traded.

Wilson (not normal-approximation) because n is small and p is near 1, exactly where the naive
interval is worst — it can run past 1.0 and implies certainty we don't have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (lo, hi), both in [0, 1].

    Handles the degenerate cases the normal approximation gets wrong: n=0 → (0,1) (total
    ignorance, not a crash), and p=1 → an upper bound of exactly 1 with an honest lower bound
    strictly below 1 (20/20 is NOT proof of perfection).
    """
    if n <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} out of range for n={n}")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class PrecisionReport:
    yes_correct: int = 0
    yes_total: int = 0
    no_correct: int = 0
    no_total: int = 0

    @property
    def yes_precision(self) -> float | None:
        return self.yes_correct / self.yes_total if self.yes_total else None

    @property
    def no_precision(self) -> float | None:
        return self.no_correct / self.no_total if self.no_total else None

    @property
    def yes_ci(self) -> tuple[float, float]:
        return wilson_interval(self.yes_correct, self.yes_total)

    @property
    def no_ci(self) -> tuple[float, float]:
        return wilson_interval(self.no_correct, self.no_total)

    @property
    def n_labels(self) -> int:
        return self.yes_total + self.no_total

    def render(self) -> str:
        def line(name: str, correct: int, total: int, ci: tuple[float, float]) -> str:
            if not total:
                return f"  {name:14s} no labels yet"
            p = correct / total
            return (
                f"  {name:14s} {correct:3d}/{total:<3d} = {p:6.1%}   "
                f"95% CI [{ci[0]:.1%}, {ci[1]:.1%}]"
            )

        return "\n".join(
            [
                f"cumulative relevance precision ({self.n_labels} human labels)",
                line("YES-precision", self.yes_correct, self.yes_total, self.yes_ci),
                line("NO-precision", self.no_correct, self.no_total, self.no_ci),
            ]
        )


def precision_from_labels(labels: list[dict]) -> PrecisionReport:
    """Fold persisted human labels into cumulative precision.

    Stratifies on the MODEL's verdict (`model_verdict`), not the human's, because that is the
    population we sampled from — computing it the other way round would answer a question we did
    not sample for (recall, which a stratified sample cannot estimate).
    """
    rep = PrecisionReport()
    for row in labels:
        model_yes = bool(row["model_verdict"])
        agrees = bool(row["label"]) == model_yes
        if model_yes:
            rep.yes_total += 1
            rep.yes_correct += int(agrees)
        else:
            rep.no_total += 1
            rep.no_correct += int(agrees)
    return rep
