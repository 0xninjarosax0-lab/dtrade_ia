"""Data loading utilities for WIN 5-minute MVP.

This module supports:
1) Loading a local CSV with 5-minute bars.
2) Generating a realistic mock intraday dataset when real data is not available.

Expected CSV columns (case-insensitive):
- timestamp, open, high, low, close, volume
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DataConfig:
    csv_path: str | None = None
    start: str = "2023-01-02"
    end: str = "2024-12-31"
    seed: int = 42


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_csv_ohlcv(csv_path: str) -> pd.DataFrame:
    """Load OHLCV data from CSV and enforce required schema."""
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    return df


def generate_mock_win_5m(start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic-ish WIN intraday 5m OHLCV mock dataset.

    Trading session approximation:
    - Morning: 09:00 to 12:00
    - Afternoon: 13:00 to 17:30
    Weekdays only.
    """
    rng = np.random.default_rng(seed)

    days = pd.date_range(start=start, end=end, freq="B")
    all_ts: list[pd.Timestamp] = []

    for d in days:
        morning = pd.date_range(d.replace(hour=9, minute=0), d.replace(hour=12, minute=0), freq="5min")
        afternoon = pd.date_range(d.replace(hour=13, minute=0), d.replace(hour=17, minute=30), freq="5min")
        all_ts.extend(morning.tolist())
        all_ts.extend(afternoon.tolist())

    ts_index = pd.DatetimeIndex(all_ts, name="timestamp")
    n = len(ts_index)

    # Simulate price as drift + stochastic volatility regime.
    regime = rng.choice([0, 1, 2], size=n, p=[0.5, 0.35, 0.15])
    vol_map = np.array([0.0009, 0.0015, 0.0025])
    drift_map = np.array([0.0, 0.00005, -0.00003])

    vol = vol_map[regime]
    drift = drift_map[regime]
    returns = drift + rng.normal(0, vol, size=n)

    base_price = 110_000.0
    close = base_price * np.exp(np.cumsum(returns))

    # Build OHLC around close with realistic candle ranges.
    open_ = np.empty(n)
    high = np.empty(n)
    low = np.empty(n)

    open_[0] = close[0] * (1 + rng.normal(0, 0.0005))
    for i in range(1, n):
        open_[i] = close[i - 1] * (1 + rng.normal(0, 0.00025))

    spread = np.abs(rng.normal(0.0008, 0.00035, size=n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)

    # Intraday volume seasonality + noise.
    minute_of_day = ts_index.hour * 60 + ts_index.minute
    vol_shape = 1.0 + 0.6 * np.cos((minute_of_day - 9 * 60) / (8.5 * 60) * 2 * np.pi)
    volume = (rng.lognormal(mean=6.5, sigma=0.35, size=n) * vol_shape).astype(int)

    df = pd.DataFrame(
        {
            "timestamp": ts_index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": "WIN$",
        }
    )
    return df.reset_index(drop=True)


def load_data(config: DataConfig) -> pd.DataFrame:
    """Load data from CSV if available, otherwise generate mock data."""
    if config.csv_path:
        csv = Path(config.csv_path)
        if csv.exists():
            return load_csv_ohlcv(str(csv))
    return generate_mock_win_5m(config.start, config.end, seed=config.seed)
