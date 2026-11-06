"""QUANT labels (doc 03) — cross-sectional, top-tercile classification target.

For each (date, ticker) the label is 1 iff the ticker's forward `horizon`-day return is in the
TOP TERCILE of the universe that day, else 0. Cross-sectional ranking self-normalizes market
regime (a flat-market winner and a bull-market winner both count) and is more robust than an
absolute up/down target.

The forward return LOOKS AHEAD by construction — that is what a label is. It is never a
feature. Leakage is prevented downstream by the walk-forward purge (= horizon) + embargo in
train.py; here we simply drop the last `horizon` dates that have no forward return yet.
"""

from __future__ import annotations

import pandas as pd


def make_labels(bars: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Tidy [date, ticker, label] top-tercile labels over the forward `horizon`.

    `bars`: tidy long [date, ticker, ..., close, ...] for the universe. Rows in the final
    `horizon` sessions (no forward return) and any ticker missing a forward price are dropped.
    """
    if bars.empty:
        return pd.DataFrame(columns=["date", "ticker", "label"])

    close = bars.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    close = close.sort_index()

    fwd = close.shift(-horizon) / close - 1.0  # forward horizon-day simple return

    # per-date cross-sectional rank; top third by forward return -> label 1
    pct = fwd.rank(axis=1, ascending=False, pct=True)  # smallest pct = best return
    label = (pct <= 1.0 / 3.0) & fwd.notna()
    label = label.where(fwd.notna())  # NaN where no forward return -> dropped on stack

    long = label.stack(future_stack=True).rename("label").reset_index()
    long = long.dropna(subset=["label"])
    long["label"] = long["label"].astype(int)
    long.columns = ["date", "ticker", "label"]
    return long.sort_values(["date", "ticker"]).reset_index(drop=True)


__all__ = ["make_labels"]
