"""Thin LightGBM wrapper for the QUANT cross-sectional ranker (doc 03).

Objective `binary` -> P(next-5d return in the universe's top tercile). Hyperparameters are
fixed per doc 03 (~500 trees, lr 0.03, shallow depth, feature_fraction 0.8, early stopping on
validation AUC) and are NOT tuned on the walk-forward test folds. Feature contributions use
LightGBM's built-in TreeSHAP (`pred_contrib=True`) so we log per-name attributions without a
separate `shap` dependency.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# Fixed a priori (doc 03). Shallow trees + column/row subsampling to fight overfit on a tiny,
# noisy cross-sectional edge. `is_unbalance` is off: the top tercile is ~1/3 by construction.
PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 5,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 50,
    "seed": 42,
    "bagging_seed": 42,
    "feature_fraction_seed": 42,
    "verbosity": -1,
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 50


def train(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    X_val: pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    params: dict | None = None,
    num_boost_round: int = NUM_BOOST_ROUND,
) -> lgb.Booster:
    """Train with early stopping on validation AUC. Column order is taken from X.columns and
    recorded in the booster so `predict` can realign an inference frame."""
    p = {**PARAMS, **(params or {})}
    feature_name = list(X.columns)
    dtrain = lgb.Dataset(X, label=np.asarray(y), feature_name=feature_name)
    dval = lgb.Dataset(X_val, label=np.asarray(y_val), reference=dtrain, feature_name=feature_name)
    return lgb.train(
        p,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )


def _aligned(booster: lgb.Booster, X: pd.DataFrame) -> pd.DataFrame:
    """Reorder/select columns to the booster's training feature order."""
    names = booster.feature_name()
    return X[names]


def predict(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """P(top tercile) at the best (early-stopped) iteration."""
    return booster.predict(_aligned(booster, X), num_iteration=booster.best_iteration)


def feature_contributions(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """TreeSHAP contributions, shape (n_rows, n_features + 1). The final column is the base
    (expected) value; contributions + base sum to the raw margin. Used to log the top-5
    feature attributions per picked name (doc 03 rationale)."""
    return booster.predict(
        _aligned(booster, X), pred_contrib=True, num_iteration=booster.best_iteration
    )


def gain_importances(booster: lgb.Booster) -> dict[str, float]:
    """Feature -> total split gain, for model_registry.metrics."""
    names = booster.feature_name()
    gains = booster.feature_importance(importance_type="gain")
    return {n: float(g) for n, g in zip(names, gains, strict=True)}


def save(booster: lgb.Booster, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path), num_iteration=booster.best_iteration)
    return path


def load(path: str | Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


__all__ = [
    "PARAMS",
    "train",
    "predict",
    "feature_contributions",
    "gain_importances",
    "save",
    "load",
]
