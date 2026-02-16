"""

MVP pipeline for WIN 5m AI agent research.

How to run:
    python main.py --csv path/to/win_5m.csv
or (mock data):
    python main.py


"""

from __future__ import annotations

import argparse

from backtest import run_backtest
from data import DataConfig, load_data
from features import build_features, get_feature_columns
from model import train_with_timeseries_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WIN 5m AI MVP")
    parser.add_argument("--csv", type=str, default=None, help="Path to local WIN 5m CSV")
    parser.add_argument("--start", type=str, default="2023-01-02", help="Start date for mock generation")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date for mock generation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for mock data")
    parser.add_argument("--splits", type=int, default=5, help="TimeSeriesSplit folds")
    parser.add_argument("--cost-bps", type=float, default=1.0, help="Transaction cost in bps per turnover unit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = DataConfig(csv_path=args.csv, start=args.start, end=args.end, seed=args.seed)
    df = load_data(cfg)
    feat_df = build_features(df)
    feature_cols = get_feature_columns()

    model_result = train_with_timeseries_split(
        feat_df,
        feature_cols=feature_cols,
        target_col="target",
        n_splits=args.splits,
    )

    # Align test slice rows for backtest using X_test index.
    test_slice = feat_df.loc[model_result.X_test.index].copy()
    bt = run_backtest(test_slice, model_result.y_pred, trade_cost_bps=args.cost_bps)

    print("=== WIN 5m MVP Results ===")
    print(f"Rows loaded: {len(df):,}")
    print(f"Rows after feature prep: {len(feat_df):,}")
    print(f"Feature columns: {feature_cols}")
    print(f"TimeSeries CV accuracies: {[round(x, 4) for x in model_result.cv_accuracies]}")
    print(f"Test accuracy: {model_result.test_accuracy:.4f}")
    print(f"Backtest cumulative return: {bt.cumulative_return:.4%}")
    print(f"Backtest max drawdown: {bt.max_drawdown:.4%}")


if __name__ == "__main__":
    main()
