"""Download hourly bars from Yahoo Finance and store them as Parquet.

Run with:  uv run python -m semiconductor_affairs.ingest

Each run merges new bars into the existing file per ticker, so running
this regularly builds a history longer than Yahoo's ~730-day hourly limit.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

TARGET = "ICHR"
INPUTS = [
    "LRCX",  # Lam Research: Ichor's biggest customer
    "AMAT",  # Applied Materials: second big customer
    "KLAC",  # equipment peer
    "ASML",  # equipment peer (US-listed ADR)
    "TSM",  # TSMC, the demand side (US-listed ADR)
    "NVDA",  # end demand for AI chips
    "UCTT",  # Ultra Clean, small-cap peer of Ichor
    "SOXX",  # semiconductor sector ETF
    "QQQ",  # broad tech market
]
TICKERS = [TARGET, *INPUTS]

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Yahoo rejects hourly requests that reach back exactly 730 days,
# so stay one day inside the limit.
MAX_HOURLY_PERIOD = "729d"


def fetch_hourly(ticker: str, period: str = MAX_HOURLY_PERIOD) -> pd.DataFrame:
    """Hourly OHLCV bars for one ticker, regular trading hours, UTC index."""
    df = yf.Ticker(ticker).history(
        period=period,
        interval="1h",
        auto_adjust=True,  # prices corrected for splits and dividends
        prepost=False,  # regular session only; pre/after-market is thin
    )
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = df.columns.str.lower()
    # Yahoo returns exchange-local time (New York); store everything in
    # UTC so tickers from other exchanges can be aligned later.
    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df


def merge_and_save(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Append new bars to the stored file; on overlap the newest wins.

    "Newest wins" matters because a bar fetched mid-hour is incomplete;
    the next run overwrites it with the finished bar.
    """
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, new])
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new
    combined = combined.sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path)
    return combined


def main() -> None:
    for ticker in TICKERS:
        bars = fetch_hourly(ticker)
        if bars.empty:
            print(f"{ticker:5s}  no data returned, skipped")
            continue
        stored = merge_and_save(bars, DATA_DIR / f"{ticker}.parquet")
        print(
            f"{ticker:5s}  fetched {len(bars):5d}  stored {len(stored):5d}  "
            f"{stored.index.min():%Y-%m-%d} → {stored.index.max():%Y-%m-%d}"
        )


if __name__ == "__main__":
    main()
