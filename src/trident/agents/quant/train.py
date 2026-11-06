"""QUANT walk-forward training (doc 03, backtest-protocol steps 3-4).

Builds the cross-sectional feature matrix + top-tercile labels across the 40-name universe
(+ SPY for regime context) over all cached history, then trains LightGBM in a rolling
walk-forward: train 3y / validate 6m / test 6m, roll 6m. At EVERY internal boundary it
applies a PURGE = label horizon (5 sessions) plus an EMBARGO = 5 sessions (Lopez de Prado,
Advances in Financial ML ch.7): the last 10 sessions of each preceding segment are dropped so
no label window straddles a boundary and fabricates a fake edge.

Out-of-sample test predictions are pooled across folds; we report per-fold and pooled test
AUC. The final (most-recent-window) model is saved to artifacts/models/ and registered in the
`model_registry` table. Hyperparameters are fixed a priori in model.py and NOT tuned here —
the test folds are touched exactly once.

CLI:  uv run python -m trident.agents.quant.train
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from trident.agents.quant import model as qmodel
from trident.agents.quant.features import FEATURE_NAMES, compute_feature_panel
from trident.agents.quant.labels import make_labels
from trident.config import ARTIFACTS_DIR, load_universe
from trident.data import market

HORIZON = 5  # label span (sessions) -> also the purge length
EMBARGO = 5  # extra sessions dropped at each boundary (Lopez de Prado ch.7)
GAP = HORIZON + EMBARGO  # trailing sessions removed from each segment before a boundary

TRAIN_YEARS = 3
VAL_MONTHS = 6
TEST_MONTHS = 6
ROLL_MONTHS = 6

DATA_START = "2016-01-01"  # cache backfill floor (yfinance); doc says 2015 but bars begin 2016
MODEL_DIR = ARTIFACTS_DIR / "models"


# ----------------------------------------------------------------- dataset
def build_dataset(
    start: str = DATA_START, end: str | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    """Assemble the per-(date, ticker) feature+label frame across the universe.

    Returns (frame, sorted_unique_dates). `frame` has columns [date, ticker, *FEATURE_NAMES,
    label] with all-NaN warmup rows and label-less tail rows dropped.
    """
    end = end or datetime.now(UTC).date().isoformat()
    universe = load_universe()
    prices = market.get_history(universe, start, end)
    spy = market.get_history(["SPY"], start, end)

    panel = compute_feature_panel(prices, spy)
    labels = make_labels(prices, horizon=HORIZON)

    frame = panel.merge(labels, on=["date", "ticker"], how="inner")
    frame = frame.dropna(subset=[*FEATURE_NAMES, "label"]).reset_index(drop=True)
    frame["label"] = frame["label"].astype(int)
    dates = np.array(sorted(frame["date"].unique()))
    return frame, dates


# ------------------------------------------------------------- walk-forward
def make_folds(dates: np.ndarray) -> list[dict]:
    """Rolling train/val/test date boundaries (half-open on the right)."""
    first, last = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    folds: list[dict] = []
    k = 0
    while True:
        tr_start = first + pd.DateOffset(months=ROLL_MONTHS * k)
        tr_end = tr_start + pd.DateOffset(years=TRAIN_YEARS)
        va_end = tr_end + pd.DateOffset(months=VAL_MONTHS)
        te_end = va_end + pd.DateOffset(months=TEST_MONTHS)
        if va_end >= last:  # no room left for a test window
            break
        test_end = min(te_end, last + pd.Timedelta(days=1))
        n_test_dates = ((dates >= va_end) & (dates < test_end)).sum()
        if n_test_dates >= 20:  # skip a stub final test (< ~1 month)
            folds.append(
                {
                    "tr_start": tr_start,
                    "tr_end": tr_end,
                    "va_end": va_end,
                    "te_end": test_end,
                }
            )
        k += 1
    return folds


def _purge_cut(dates: np.ndarray, boundary: pd.Timestamp) -> pd.Timestamp:
    """The earliest date to DROP from the segment preceding `boundary`: GAP sessions before
    the first session >= boundary. Removing rows with date >= this value strips any label
    window that would straddle the boundary, plus the embargo."""
    idx_b = int(np.searchsorted(dates, np.datetime64(boundary), side="left"))
    cut_i = idx_b - GAP
    return pd.Timestamp(dates[cut_i]) if cut_i >= 0 else pd.Timestamp(dates[0])


def slice_fold(
    frame: pd.DataFrame, dates: np.ndarray, fold: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Purged+embargoed train, purged+embargoed val, and untouched OOS test slices."""
    d = frame["date"]
    train_cut = _purge_cut(dates, fold["tr_end"])  # drop train tail before val
    val_cut = _purge_cut(dates, fold["va_end"])  # drop val tail before test

    train = frame[(d >= fold["tr_start"]) & (d < fold["tr_end"]) & (d < train_cut)]
    val = frame[(d >= fold["tr_end"]) & (d < fold["va_end"]) & (d < val_cut)]
    test = frame[(d >= fold["va_end"]) & (d < fold["te_end"])]
    return train, val, test


def run_walk_forward(frame: pd.DataFrame, dates: np.ndarray) -> dict:
    """Train each fold, collect pooled OOS test predictions, report per-fold + pooled AUC.
    Returns a dict with the final booster and metrics for registration."""
    folds = make_folds(dates)
    print(
        f"\nWalk-forward: {len(folds)} folds (train {TRAIN_YEARS}y / val {VAL_MONTHS}m / "
        f"test {TEST_MONTHS}m, roll {ROLL_MONTHS}m; purge {HORIZON} + embargo {EMBARGO})\n"
    )

    pooled_y: list[np.ndarray] = []
    pooled_p: list[np.ndarray] = []
    pooled_pred: list[
        pd.DataFrame
    ] = []  # (date, ticker, score) OOS -> a valid walk-forward backtest
    per_fold: list[dict] = []
    final_booster = None
    final_fold = None

    header = f"{'fold':>4} {'train':>21} {'val_end':>10} {'test_end':>10} "
    header += f"{'ntr':>6} {'nval':>6} {'ntst':>6} {'val_auc':>8} {'test_auc':>8}"
    print(header)
    print("-" * len(header))

    for i, fold in enumerate(folds):
        train, val, test = slice_fold(frame, dates, fold)
        if train.empty or val.empty or test.empty:
            continue
        Xtr, ytr = train[FEATURE_NAMES], train["label"].to_numpy()
        Xva, yva = val[FEATURE_NAMES], val["label"].to_numpy()
        Xte, yte = test[FEATURE_NAMES], test["label"].to_numpy()

        booster = qmodel.train(Xtr, ytr, Xva, yva)
        p_va = qmodel.predict(booster, Xva)
        p_te = qmodel.predict(booster, Xte)

        val_auc = _safe_auc(yva, p_va)
        test_auc = _safe_auc(yte, p_te)
        pooled_y.append(yte)
        pooled_p.append(p_te)
        pooled_pred.append(test[["date", "ticker"]].assign(score=p_te))
        per_fold.append(
            {
                "fold": i,
                "train": f"{fold['tr_start']:%Y-%m}..{fold['tr_end']:%Y-%m}",
                "val_auc": val_auc,
                "test_auc": test_auc,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "best_iter": int(booster.best_iteration or 0),
            }
        )
        final_booster, final_fold = booster, fold

        print(
            f"{i:>4} {fold['tr_start']:%Y-%m}..{fold['tr_end']:%Y-%m}   "
            f"{fold['va_end']:%Y-%m-%d} {fold['te_end']:%Y-%m-%d} "
            f"{len(train):>6} {len(val):>6} {len(test):>6} "
            f"{val_auc:>8.4f} {test_auc:>8.4f}"
        )

    y_all = np.concatenate(pooled_y)
    p_all = np.concatenate(pooled_p)
    pooled_auc = _safe_auc(y_all, p_all)
    mean_test_auc = float(np.mean([f["test_auc"] for f in per_fold]))

    print("-" * len(header))
    print(f"\nPooled OOS test AUC: {pooled_auc:.4f}  (n={len(y_all)})")
    print(f"Mean per-fold test AUC: {mean_test_auc:.4f}")

    oos = (
        pd.concat(pooled_pred, ignore_index=True)
        if pooled_pred
        else pd.DataFrame(columns=["date", "ticker", "score"])
    )
    return {
        "final_booster": final_booster,
        "final_fold": final_fold,
        "pooled_auc": pooled_auc,
        "mean_test_auc": mean_test_auc,
        "per_fold": per_fold,
        "pooled_n": int(len(y_all)),
        "oos_predictions": oos,
    }


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


# ------------------------------------------------------------- registration
async def register_model(
    artifact_path: Path, train_window: str, metrics: dict, active: bool = True
) -> int | None:
    """INSERT the trained model into `model_registry` (deactivating prior quant rows so one
    stays active). Returns the new row id, or None if the DB is unreachable (then a JSON
    sidecar next to the artifact is the fallback record)."""
    from sqlalchemy import text

    from trident.db import dispose_engine, get_engine

    try:
        async with get_engine().begin() as conn:
            if active:
                await conn.execute(
                    text("UPDATE model_registry SET active=false WHERE agent_id='quant'")
                )
            row_id = (
                await conn.execute(
                    text(
                        "INSERT INTO model_registry "
                        "(agent_id, trained_at, train_window, metrics, artifact_path, active) "
                        "VALUES ('quant', :ts, :tw, CAST(:m AS JSONB), :ap, :active) "
                        "RETURNING id"
                    ),
                    {
                        "ts": datetime.now(UTC),
                        "tw": train_window,
                        "m": json.dumps(metrics),
                        "ap": str(artifact_path),
                        "active": active,
                    },
                )
            ).scalar_one()
        return int(row_id)
    except Exception as e:  # noqa: BLE001 — DB optional here; never lose the training result
        print(
            f"\n[warn] model_registry write failed ({type(e).__name__}: {e}). "
            "Writing JSON sidecar instead."
        )
        sidecar = artifact_path.with_suffix(".registry.json")
        sidecar.write_text(
            json.dumps(
                {
                    "agent_id": "quant",
                    "trained_at": datetime.now(UTC).isoformat(),
                    "train_window": train_window,
                    "metrics": metrics,
                    "artifact_path": str(artifact_path),
                    "active": active,
                },
                indent=2,
            )
        )
        print(f"[warn] sidecar: {sidecar}")
        return None
    finally:
        try:
            await dispose_engine()
        except Exception:  # noqa: BLE001
            pass


# -------------------------------------------------------------------- main
async def main() -> float:
    print("Building feature+label dataset across the universe ...")
    frame, dates = build_dataset()
    print(
        f"  rows={len(frame)}  tickers={frame['ticker'].nunique()}  "
        f"dates {pd.Timestamp(dates[0]):%Y-%m-%d} .. {pd.Timestamp(dates[-1]):%Y-%m-%d}  "
        f"base rate={frame['label'].mean():.3f}"
    )

    result = run_walk_forward(frame, dates)
    booster = result["final_booster"]
    fold = result["final_fold"]

    importances = qmodel.gain_importances(booster)
    top = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("\nTop feature importances (gain):")
    for name, g in top:
        print(f"  {name:<22} {g:>14.1f}")

    train_window = f"{fold['tr_start']:%Y-%m}..{fold['va_end']:%Y-%m}"
    final_val_auc = next((f["val_auc"] for f in reversed(result["per_fold"])), float("nan"))
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = MODEL_DIR / f"quant_lgbm_{ts}.txt"
    qmodel.save(booster, artifact_path)
    print(f"\nSaved model -> {artifact_path}")

    # Persist the pooled walk-forward OOS predictions (each score from a model trained only on
    # prior data) so M6 can backtest the ML validly — no in-sample look-ahead.
    oos_path = MODEL_DIR / "quant_oos_predictions.parquet"
    result["oos_predictions"].to_parquet(oos_path)
    print(f"Saved OOS predictions -> {oos_path}  (rows={len(result['oos_predictions'])})")

    metrics = {
        "auc": result["pooled_auc"],
        "val_auc": final_val_auc,
        "mean_test_auc": result["mean_test_auc"],
        "n_folds": len(result["per_fold"]),
        "pooled_test_n": result["pooled_n"],
        "purge": HORIZON,
        "embargo": EMBARGO,
        "per_fold": result["per_fold"],
        "feature_importances": importances,
    }
    row_id = await register_model(artifact_path, train_window, metrics, active=True)
    if row_id is not None:
        print(f"Registered model_registry id={row_id} (train_window={train_window}, active=true)")

    print(f"\nPOOLED OOS TEST AUC = {result['pooled_auc']:.4f}")
    return result["pooled_auc"]


if __name__ == "__main__":
    asyncio.run(main())
