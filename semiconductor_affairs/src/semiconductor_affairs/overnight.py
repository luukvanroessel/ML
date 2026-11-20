"""Overnight-gap dataset: one sample per night, decided at the close.

Target: ICHR's gap = log(next day's open / today's close).
Inputs: a sequence of daily feature vectors, shape (n_nights, days, F),
using only what is known at today's close (plus the earnings calendar,
which is published weeks in advance).
"""

import numpy as np
import pandas as pd

from semiconductor_affairs.data import MARKET_TZ
from semiconductor_affairs.features import Dataset
from semiconductor_affairs.ingest import (
    DATA_DIR,
    EARNINGS_TICKERS,
    TARGET,
    TICKERS,
)

VOLUME_LOOKBACK = 20  # days, to judge whether today's volume is unusual


def daily_bars(ticker: str) -> pd.DataFrame:
    """Per trading day: open, close, last-bar start/return, volumes."""
    h = pd.read_parquet(DATA_DIR / f"{ticker}.parquet")
    day = h.index.tz_convert(MARKET_TZ).date
    g = h.groupby(day)
    d = pd.DataFrame({
        "open": g["open"].first(),
        "close": g["close"].last(),
        "first_bar": g.apply(lambda x: x.index[0]),
        "last_bar": g.apply(lambda x: x.index[-1]),
        "close_before_last": g["close"].nth(-2).set_axis(g.size().index),
        "last_volume": g["volume"].last(),
        "volume": g["volume"].sum(),
    })
    d.index = pd.to_datetime(d.index)
    return d


def ticker_features(d: pd.DataFrame) -> pd.DataFrame:
    """Returns known at today's close."""
    prev_close = d["close"].shift(1)
    return pd.DataFrame({
        "gap": np.log(d["open"] / prev_close),  # this morning's gap
        "intraday": np.log(d["close"] / d["open"]),  # open -> close
        "last": np.log(d["close"] / d["close_before_last"]),  # last bar
    })


def volume_features(d: pd.DataFrame) -> pd.DataFrame:
    """log(volume / median of the previous 20 days): 0 = a normal day."""
    def ratio(col):
        typical = d[col].rolling(VOLUME_LOOKBACK).median().shift(1)
        return np.log(d[col] / typical)

    return pd.DataFrame({
        "volume": ratio("volume"),
        "last_volume": ratio("last_volume"),
    })


def earnings_tonight(days: pd.DataFrame) -> pd.DataFrame:
    """1 if a ticker reports between today's last bar and tomorrow's open.

    Covers both after-close US reports (16:00 New York) and pre-market
    reports from ASML / TSMC (early morning New York).
    """
    e = pd.read_parquet(DATA_DIR / "earnings.parquet")
    start = days["last_bar"]
    end = days["first_bar"].shift(-1)
    out = {}
    for t in EARNINGS_TICKERS:
        times = e.loc[e["ticker"] == t, "timestamp"].to_numpy()
        out[f"earn_{t}"] = [
            float(((times >= s) & (times < n)).any()) if pd.notna(n) else 0.0
            for s, n in zip(start.to_numpy(), end.to_numpy())
        ]
    return pd.DataFrame(out, index=days.index)


def build_daily_panel() -> tuple[pd.DataFrame, pd.Series]:
    """(features per day, ICHR's gap the *next* morning) on common days."""
    bars = {t: daily_bars(t) for t in TICKERS}
    target = bars[TARGET]

    parts = [
        ticker_features(bars[t]).add_prefix(f"{t}_") for t in TICKERS
    ]
    parts.append(volume_features(target).add_prefix(f"{TARGET}_"))
    parts.append(earnings_tonight(target))
    # Calendar days until the next open: 1 on a weeknight, 3 over a
    # weekend, more over a holiday weekend. Known in advance.
    next_day = target.index.to_series().shift(-1)
    parts.append(
        (next_day - target.index.to_series()).dt.days.rename("night_days")
    )
    panel = pd.concat(parts, axis=1)

    next_gap = np.log(target["open"].shift(-1) / target["close"])
    return panel, next_gap.rename("next_gap")


def make_overnight_dataset(days: int = 1) -> Dataset:
    """Sliding window of `days` daily feature vectors -> next gap.

    Timestamps are the start of the last bar of the decision day (UTC),
    so evaluate.walk_forward_splits works unchanged.
    """
    panel, next_gap = build_daily_panel()
    last_bar = daily_bars(TARGET)["last_bar"]
    values = panel.to_numpy(dtype=float)
    X, y, ts = [], [], []
    for i in range(days - 1, len(panel)):
        x = values[i - days + 1 : i + 1]
        if np.isnan(x).any() or np.isnan(next_gap.iloc[i]):
            continue  # warm-up, or the last day (no next open yet)
        X.append(x)
        y.append(next_gap.iloc[i])
        ts.append(last_bar.iloc[i])
    return Dataset(
        X=np.stack(X),
        y=np.array(y),
        timestamps=pd.DatetimeIndex(ts),
        feature_names=list(panel.columns),
    )
