"""Load stored bars and turn them into model-ready series."""

import numpy as np
import pandas as pd

from semiconductor_affairs.ingest import DATA_DIR, TICKERS

MARKET_TZ = "America/New_York"
FIRST_BAR = "09:30"  # its return includes the overnight gap


def load_close(tickers: list[str] = TICKERS) -> pd.DataFrame:
    """Close prices, one column per ticker, on the union of timestamps."""
    return pd.DataFrame(
        {t: pd.read_parquet(DATA_DIR / f"{t}.parquet")["close"] for t in tickers}
    )


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Bar-to-bar log returns. A missing bar is forward-filled first,
    i.e. treated as "no trade, price unchanged" (return 0)."""
    return np.log(close.ffill()).diff().iloc[1:]


def bar_start(index: pd.DatetimeIndex) -> pd.Index:
    """Bar start time as 'HH:MM' in New York time."""
    return index.tz_convert(MARKET_TZ).strftime("%H:%M")
