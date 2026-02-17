"""backtesting.py integration layer for signal-based WIN strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class BacktestPyResult:
    stats: pd.Series
    equity_curve: pd.DataFrame
    trades: pd.DataFrame


def _require_backtesting_py() -> tuple[type, type, type | None]:
    try:
        from backtesting import Backtest, Strategy  # type: ignore
        from backtesting.lib import FractionalBacktest  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "Dependency 'backtesting' not found. Install with: pip install backtesting"
        ) from exc
    return Backtest, Strategy, FractionalBacktest


def run_backtesting_py(
    df: pd.DataFrame,
    cash: float = 100_000,
    commission: float = 0.0002,
    exclusive_orders: bool = True,
    use_fractional: bool = True,
) -> BacktestPyResult:
    """Run signal-based backtest using backtesting.py.

    Required columns:
    - Open, High, Low, Close, Volume
    - signal (1 long, -1 short, 0 flat)
    """
    required_cols = {"Open", "High", "Low", "Close", "Volume", "signal"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for backtesting.py: {sorted(missing)}")

    bt_df = df.copy()

    Backtest, Strategy, FractionalBacktest = _require_backtesting_py()

    class SignalStrategy(Strategy):
        def next(self) -> None:
            sig = int(self.data.signal[-1])

            if sig == 1:
                if self.position.is_short:
                    self.position.close()
                if not self.position:
                    self.buy()
            elif sig == -1:
                if self.position.is_long:
                    self.position.close()
                if not self.position:
                    self.sell()
            else:
                if self.position:
                    self.position.close()

    backtest_cls: type[Any] = Backtest
    if use_fractional and FractionalBacktest is not None:
        backtest_cls = FractionalBacktest

    bt = backtest_cls(
        bt_df,
        SignalStrategy,
        cash=cash,
        commission=commission,
        exclusive_orders=exclusive_orders,
    )
    stats = bt.run()

    equity_curve = stats.get("_equity_curve", pd.DataFrame())
    trades = stats.get("_trades", pd.DataFrame())

    return BacktestPyResult(stats=stats, equity_curve=equity_curve, trades=trades)