"""Feature and label engineering for WIN 5m supervised pipeline."""

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
    """Build anti-leakage feature set using only information known up to t."""
    data = df.copy().sort_values("timestamp").reset_index(drop=True)

    for lag in [1, 2, 3, 6, 12]:
        data[f"ret_{lag}"] = data["close"].pct_change(lag)

    data["rsi_14"] = _rsi(data["close"], 14)
    data["atr_14"] = _atr(data, 14)
    data["atr_14_norm"] = data["atr_14"] / data["close"]

    data["range_norm"] = (data["high"] - data["low"]) / data["close"]

    typical_price = (data["high"] + data["low"] + data["close"]) / 3.0
    cum_tpv = (typical_price * data["volume"]).groupby(data["session_id"]).cumsum()
    cum_vol = data["volume"].groupby(data["session_id"]).cumsum().replace(0, np.nan)
    data["vwap"] = cum_tpv / cum_vol
    data["vwap_distance"] = (data["close"] / data["vwap"]) - 1.0

    data["hour"] = data["timestamp"].dt.hour
    data["minute"] = data["timestamp"].dt.minute
    minute_of_day = data["hour"] * 60 + data["minute"]
    data["minute_sin"] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
    data["minute_cos"] = np.cos(2 * np.pi * minute_of_day / (24 * 60))

    volume_mean = data.groupby(["hour", "minute"])["volume"].transform("mean")
    volume_std = data.groupby(["hour", "minute"])["volume"].transform("std").replace(0, np.nan)
    data["vol_zscore_hora"] = (data["volume"] - volume_mean) / volume_std

    data["trend_strength"] = (data["close"].rolling(10).mean() / data["close"].rolling(30).mean()) - 1.0

    rolling_vol = data["ret_1"].rolling(30).std()
    q1 = rolling_vol.rolling(200).quantile(0.33)
    q2 = rolling_vol.rolling(200).quantile(0.66)
    data["regime_vol"] = np.select(
        [rolling_vol <= q1, rolling_vol <= q2],
        [0, 1],
        default=2,
    )

    data["is_macro_window"] = False
    data.loc[(data["hour"] == 10) & (data["minute"].between(0, 15)), "is_macro_window"] = True
    data.loc[(data["hour"] == 14) & (data["minute"].between(0, 15)), "is_macro_window"] = True

    return data


def build_labels(
    df: pd.DataFrame,
    horizon_bars: int = 3,
    cost_buffer_bps: float = 2.0,
) -> pd.DataFrame:
    """Build binary and ternary labels for t+h horizon with cost buffer."""
    out = df[["timestamp", "symbol", "close"]].copy()
    out["horizon_bars"] = horizon_bars

    future_return = df["close"].shift(-horizon_bars) / df["close"] - 1.0
    out["future_return_pts"] = df["close"] * future_return

    threshold = cost_buffer_bps / 10_000.0
    out["label_bin"] = (future_return > threshold).astype(int)

    out["label_tri"] = np.select(
        [future_return > threshold, future_return < -threshold],
        [1, -1],
        default=0,
    )

    return out


def get_feature_columns() -> list[str]:
    return [
        "ret_1",
        "ret_2",
        "ret_3",
        "ret_6",
        "ret_12",
        "atr_14_norm",
        "range_norm",
        "vwap_distance",
        "vol_zscore_hora",
        "trend_strength",
        "regime_vol",
        "minute_sin",
        "minute_cos",
        "rsi_14",
        "is_macro_window",
    ]