"""Turn hourly returns into (window, target) samples.

Every model gets the same input: a window of the last `window` bars, each
bar holding a vector of features. Shape (n_samples, window, n_features).
Ridge flattens it; the LSTM reads it bar by bar. Same information, so the
comparison between them is fair.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from semiconductor_affairs.data import FIRST_BAR, bar_start
from semiconductor_affairs.ingest import TARGET

LEADERS = ["LRCX", "AMAT", "KLAC", "ASML", "TSM", "NVDA", "SOXX"]
BIG_MOVE_SIGMA = 2.0
VOL_LOOKBACK = 140  # bars (~4 weeks) to estimate a ticker's typical move


@dataclass
class Dataset:
    X: np.ndarray  # (n_samples, window, n_features)
    y: np.ndarray  # (n_samples,) target's log return in the next bar
    timestamps: pd.DatetimeIndex  # bar t: the last bar inside the window
    feature_names: list[str]


def big_moves(returns: pd.DataFrame, tickers=LEADERS) -> pd.DataFrame:
    """The return itself when it's a big move, else 0.

    "Big" = more than BIG_MOVE_SIGMA times the ticker's typical move over
    the previous VOL_LOOKBACK bars (only past data, so no look-ahead).
    In a linear model this gives big moves their own coefficient, which
    is the nonlinearity hinted at in notebook 02, section 5.
    """
    r = returns[tickers]
    typical = r.rolling(VOL_LOOKBACK).std().shift(1)
    is_big = r.abs() > BIG_MOVE_SIGMA * typical
    # NaN during the warm-up, so those bars are skipped, not silently 0
    return (r * is_big).where(typical.notna()).add_prefix("big_")


def build_panel(returns: pd.DataFrame, with_big_moves: bool) -> pd.DataFrame:
    """Per-bar feature vector: every ticker's return (+ big-move columns)."""
    parts = [returns]
    if with_big_moves:
        parts.append(big_moves(returns))
    return pd.concat(parts, axis=1)


def make_dataset(
    returns: pd.DataFrame,
    window: int = 7,
    with_big_moves: bool = False,
    include_first_bar: bool = False,
) -> Dataset:
    """Slide a window over the panel; the target is the next bar's return.

    Sample i uses bars (t-window+1 ... t) as input and the target's return
    in bar t+1 as label. By default, targets in the 09:30 bar (overnight
    gap) are skipped, as decided in notebook 02.
    """
    panel = build_panel(returns, with_big_moves)
    values = panel.to_numpy()
    target_next = returns[TARGET].shift(-1).to_numpy()
    next_bar = pd.Series(bar_start(returns.index)).shift(-1).to_numpy()

    # Skip the volatility warm-up for every variant, so datasets with and
    # without big-move features cover exactly the same bars.
    first = VOL_LOOKBACK + window
    X, y, ts = [], [], []
    for t in range(first, len(panel) - 1):
        if not include_first_bar and next_bar[t] == FIRST_BAR:
            continue
        x = values[t - window + 1 : t + 1]
        if np.isnan(x).any():  # start of the data / rolling warm-up
            continue
        X.append(x)
        y.append(target_next[t])
        ts.append(panel.index[t])

    return Dataset(
        X=np.stack(X),
        y=np.array(y),
        timestamps=pd.DatetimeIndex(ts),
        feature_names=list(panel.columns),
    )
