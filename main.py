"""Professionalized MVP pipeline for WIN 5m: data -> features/labels -> model -> signal -> backtesting.py."""

from __future__ import annotations

import argparse

from backtest import run_backtesting_py
from data import DataConfig, load_data
from features import build_features, build_labels, get_feature_columns
from model import train_with_timeseries_split
from signal_engine import apply_signal_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WIN 5m supervised pipeline + backtesting.py")
    parser.add_argument("--csv", type=str, default=None, help="Path to local WIN 5m CSV")
    parser.add_argument("--start", type=str, default="2023-01-02", help="Start date for mock generation")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date for mock generation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for mock data")
    parser.add_argument("--symbol", type=str, default="WIN$", help="Symbol name")

    parser.add_argument("--splits", type=int, default=5, help="TimeSeriesSplit folds")
    parser.add_argument("--horizon-bars", type=int, default=3, help="Label horizon in bars")
    parser.add_argument("--cost-buffer-bps", type=float, default=2.0, help="Cost+buffer threshold for labels")
    parser.add_argument("--threshold-buy", type=float, default=0.55, help="Long decision threshold")
    parser.add_argument("--threshold-sell", type=float, default=0.45, help="Short decision threshold")
    parser.add_argument("--max-trades-day", type=int, default=8, help="Daily max trades")

    parser.add_argument("--cash", type=float, default=100_000, help="Initial cash for backtesting.py")
    parser.add_argument("--commission", type=float, default=0.0002, help="Commission ratio per trade")
    return parser.parse_args()


def _to_backtesting_schema(df):
    out = df.copy()
    out = out.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    action_to_signal = {"buy": 1, "sell": -1, "flat": 0}
    out["signal"] = out["action"].map(action_to_signal).fillna(0).astype(int)
    out = out.set_index("timestamp")
    return out[["Open", "High", "Low", "Close", "Volume", "signal"]]


def main() -> None:
    args = parse_args()

    cfg = DataConfig(
        csv_path=args.csv,
        start=args.start,
        end=args.end,
        seed=args.seed,
        symbol=args.symbol,
    )

    bars = load_data(cfg)
    features = build_features(bars)
    labels = build_labels(features, horizon_bars=args.horizon_bars, cost_buffer_bps=args.cost_buffer_bps)

    ds = features.merge(
        labels[["timestamp", "label_bin", "label_tri", "future_return_pts", "horizon_bars"]],
        on="timestamp",
        how="left",
    )

    feature_cols = get_feature_columns()
    model_result = train_with_timeseries_split(
        ds,
        feature_cols=feature_cols,
        target_col="label_bin",
        n_splits=args.splits,
        decision_threshold=args.threshold_buy,
    )

    test_slice = ds.loc[model_result.X_test.index].copy()
    test_slice["prob_long"] = model_result.y_proba

    signal_df = apply_signal_contract(
        test_slice,
        prob_col="prob_long",
        threshold_buy=args.threshold_buy,
        threshold_sell=args.threshold_sell,
        max_trades_per_day=args.max_trades_day,
    )

    bt_df = _to_backtesting_schema(signal_df)
    bt_result = run_backtesting_py(
        bt_df,
        cash=args.cash,
        commission=args.commission,
        exclusive_orders=True,
    )

    print("=== WIN 5m Pipeline (backtesting.py) ===")
    print(f"Rows loaded (bars_5m): {len(bars):,}")
    print(f"Rows modelable (gold): {len(ds.dropna(subset=feature_cols + ['label_bin'])):,}")
    print(f"Feature columns: {feature_cols}")
    print(f"CV accuracies: {[round(x, 4) for x in model_result.cv_accuracies]}")
    print(f"Test accuracy: {model_result.test_accuracy:.4f}")
    print(f"Backtesting.py Return [%]: {bt_result.stats.get('Return [%]')}")
    print(f"Backtesting.py Max. Drawdown [%]: {bt_result.stats.get('Max. Drawdown [%]')}")
    print(f"Backtesting.py # Trades: {bt_result.stats.get('# Trades')}")


if __name__ == "__main__":
    main()