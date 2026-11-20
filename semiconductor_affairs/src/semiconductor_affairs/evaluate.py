"""Walk-forward evaluation and metrics, shared by all models.

Walk-forward = train on everything before a cut-off, test on the next
block, move the cut-off forward, repeat. Every prediction is made by a
model that has only seen the past, like it would be in real life.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from semiconductor_affairs.features import Dataset


def walk_forward_splits(
    timestamps: pd.DatetimeIndex,
    first_test: str = "2025-01-01",
    test_months: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_idx, test_idx) pairs with an expanding training window.

    The last training sample is dropped (embargo): its label is the
    return of the first test bar, so keeping it would leak test data.
    """
    splits = []
    start = pd.Timestamp(first_test, tz="UTC")
    while start < timestamps[-1]:
        end = start + pd.DateOffset(months=test_months)
        train = np.flatnonzero(timestamps < start)[:-1]
        test = np.flatnonzero((timestamps >= start) & (timestamps < end))
        if len(test):
            splits.append((train, test))
        start = end
    return splits


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """All in % (returns × 100) where it's a size."""
    err = y_pred - y_true
    mse_zero = np.mean(y_true**2)
    moved = y_true != 0  # a flat hour has no direction to get right
    pred = pd.Series(y_pred)
    out = {
        "n": len(y_true),
        "MAE %": np.mean(np.abs(err)) * 100,
        "RMSE %": np.sqrt(np.mean(err**2)) * 100,
        # 1 - MSE/MSE_zero: >0 beats always-predict-0, <0 is worse
        "skill vs zero %": (1 - np.mean(err**2) / mse_zero) * 100,
        "corr": np.nan,
        "rank corr": np.nan,
        "direction acc %": np.nan,
    }
    if pred.std() == 0:  # constant prediction (e.g. Zero): undefined
        return out
    out |= {
        "corr": pred.corr(pd.Series(y_true)),
        "rank corr": pred.corr(pd.Series(y_true), method="spearman"),
        "direction acc %": np.mean(
            np.sign(y_pred[moved]) == np.sign(y_true[moved])
        )
        * 100,
    }
    return out


def walk_forward(
    make_model: Callable[[], object],
    data: Dataset,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.Series:
    """Out-of-sample predictions for every test bar, refitting per fold.

    `make_model` returns a fresh, unfitted model with fit(X, y) and
    predict(X), so no state leaks from one fold to the next.
    """
    preds = []
    for train, test in splits:
        model = make_model()
        model.fit(data.X[train], data.y[train])
        preds.append(
            pd.Series(model.predict(data.X[test]), index=test)
        )
    return pd.concat(preds)


def report(data: Dataset, preds: pd.Series, by: str = "fold") -> pd.DataFrame:
    """Metrics per test period ('fold': quarter, or 'year') plus overall."""
    ts = data.timestamps[preds.index]
    if by == "year":
        groups = ts.year.astype(str)
    else:
        groups = ts.tz_localize(None).to_period("Q").astype(str)
    rows = {
        g: metrics(data.y[idx], preds[idx].to_numpy())
        for g, idx in pd.Series(preds.index, index=groups).groupby(level=0)
    }
    rows["all"] = metrics(data.y[preds.index], preds.to_numpy())
    return pd.DataFrame(rows).T
