"""Data loading and normalization for WIN 5m pipeline."""

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
    symbol: str = "WIN$"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    return out


def _build_session_id(ts: pd.Series) -> pd.Series:
    return ts.dt.strftime("%Y-%m-%d")


def load_csv_ohlcv(csv_path: str, symbol: str = "WIN$") -> pd.DataFrame:
    """Load OHLCV data from CSV and enforce a minimum bars_5m schema."""
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])

    if "symbol" not in df.columns:
        df["symbol"] = symbol
    if "is_roll_day" not in df.columns:
        df["is_roll_day"] = False

    df["session_id"] = _build_session_id(df["timestamp"])

    return df[
        [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "session_id",
            "is_roll_day",
        ]
    ].copy()


def generate_mock_win_5m(start: str, end: str, seed: int = 42, symbol: str = "WIN$") -> pd.DataFrame:
    """Generate realistic-ish mock WIN 5m bars with minimum bars_5m schema."""
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

    regime = rng.choice([0, 1, 2], size=n, p=[0.5, 0.35, 0.15])
    vol_map = np.array([0.0009, 0.0015, 0.0025])
    drift_map = np.array([0.0, 0.00005, -0.00003])

    returns = drift_map[regime] + rng.normal(0, vol_map[regime], size=n)
    base_price = 110_000.0
    close = base_price * np.exp(np.cumsum(returns))

    open_ = np.empty(n)
    open_[0] = close[0] * (1 + rng.normal(0, 0.0005))
    for i in range(1, n):
        open_[i] = close[i - 1] * (1 + rng.normal(0, 0.00025))

    spread = np.abs(rng.normal(0.0008, 0.00035, size=n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)

    minute_of_day = ts_index.hour * 60 + ts_index.minute
    vol_shape = 1.0 + 0.6 * np.cos((minute_of_day - 9 * 60) / (8.5 * 60) * 2 * np.pi)
    volume = (rng.lognormal(mean=6.5, sigma=0.35, size=n) * vol_shape).astype(int)

    df = pd.DataFrame(
        {
            "timestamp": ts_index,
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "session_id": ts_index.strftime("%Y-%m-%d"),
            "is_roll_day": False,
        }
    )
    return df.reset_index(drop=True)


def load_data(config: DataConfig) -> pd.DataFrame:
    """Load CSV data if path exists; otherwise return synthetic bars."""
    if config.csv_path:
        csv = Path(config.csv_path)
        if csv.exists():
            return load_csv_ohlcv(str(csv), symbol=config.symbol)
    return generate_mock_win_5m(config.start, config.end, seed=config.seed, symbol=config.symbol)