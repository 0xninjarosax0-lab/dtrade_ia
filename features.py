"""Feature engineering for WIN 5m MVP.

Indicators included:
- SMA (short/long)
- RSI
- ATR
- MACD (line, signal, histogram)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical features and binary target for next-bar direction."""
    data = df.copy().sort_values("timestamp").reset_index(drop=True)

    data["ret_1"] = data["close"].pct_change()

    data["sma_10"] = data["close"].rolling(10).mean()
    data["sma_30"] = data["close"].rolling(30).mean()
    data["sma_ratio"] = data["sma_10"] / data["sma_30"] - 1.0

    data["rsi_14"] = _rsi(data["close"], 14)
    data["atr_14"] = _atr(data, 14)
    data["atr_norm"] = data["atr_14"] / data["close"]

    ema_fast = data["close"].ewm(span=12, adjust=False).mean()
    ema_slow = data["close"].ewm(span=26, adjust=False).mean()
    data["macd"] = ema_fast - ema_slow
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    data["vol_chg"] = data["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    data["hl_range"] = (data["high"] - data["low"]) / data["close"]

    # Target: 1 = buy, 0 = sell based on next bar close return
    future_ret = data["close"].shift(-1) / data["close"] - 1.0
    data["target"] = (future_ret > 0).astype(int)

    # Optional human-readable signal label.
    data["signal_label"] = np.where(data["target"] == 1, "buy", "sell")

    return data


def get_feature_columns() -> list[str]:
    return [
        "ret_1",
        "sma_ratio",
        "rsi_14",
        "atr_norm",
        "macd",
        "macd_signal",
        "macd_hist",
        "vol_chg",
        "hl_range",
    ]
