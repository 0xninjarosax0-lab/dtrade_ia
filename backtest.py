"""Simple backtest engine for classification-based buy/sell signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    accuracy: float
    cumulative_return: float
    max_drawdown: float
    equity_curve: pd.Series
    report_df: pd.DataFrame


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def run_backtest(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    trade_cost_bps: float = 1.0,
) -> BacktestResult:
    """Backtest strategy from model predictions.

    - y_pred = 1 => long (+1)
    - y_pred = 0 => short (-1)
    """
    df = test_df.copy().reset_index(drop=True)
    df = df.iloc[: len(y_pred)].copy()

    df["pred"] = y_pred
    df["position"] = np.where(df["pred"] == 1, 1, -1)

    df["asset_ret"] = df["close"].pct_change().fillna(0.0)
    df["turnover"] = df["position"].diff().abs().fillna(0.0)

    # Trade cost in bps per position change unit.
    cost = (trade_cost_bps / 10_000.0) * df["turnover"]

    # Use previous bar position to avoid lookahead in bar return.
    df["strategy_ret"] = df["position"].shift(1).fillna(0) * df["asset_ret"] - cost

    df["equity"] = (1.0 + df["strategy_ret"]).cumprod()

    accuracy = float((df["pred"] == df["target"]).mean())
    cumulative_return = float(df["equity"].iloc[-1] - 1.0)
    mdd = max_drawdown(df["equity"])

    return BacktestResult(
        accuracy=accuracy,
        cumulative_return=cumulative_return,
        max_drawdown=mdd,
        equity_curve=df["equity"],
        report_df=df,
    )
