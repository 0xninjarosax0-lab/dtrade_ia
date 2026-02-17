"""Professionalized MVP pipeline for WIN 5m: data -> features/labels -> model -> signal -> backtesting.py."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument(
        "--use-fractional-backtest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FractionalBacktest to avoid whole-unit cash constraints",
    )
    parser.add_argument(
        "--show-last-trades",
        type=int,
        default=10,
        help="Quantidade de trades finais exibidos no resumo do terminal",
    )
    parser.add_argument(
        "--export-trades-csv",
        type=str,
        default=None,
        help="Caminho opcional para exportar trades executados em CSV",
    )
    parser.add_argument(
        "--export-equity-csv",
        type=str,
        default=None,
        help="Caminho opcional para exportar curva de equity em CSV",
    )
    parser.add_argument(
        "--export-signals-csv",
        type=str,
        default=None,
        help="Caminho opcional para exportar probabilidades e ações do modelo em CSV",
    )
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




def _print_trade_summary(trades, limit: int) -> None:
    if trades is None or trades.empty:
        print("Nenhum trade foi executado pelo backtesting.py nesta execução.")
        return

    display_cols = [
        col
        for col in [
            "EntryTime",
            "ExitTime",
            "Size",
            "EntryPrice",
            "ExitPrice",
            "PnL",
            "ReturnPct",
            "Duration",
        ]
        if col in trades.columns
    ]
    n_rows = max(1, int(limit))
    print(f"\nÚltimos {min(n_rows, len(trades))} trades:")
    print(trades[display_cols].tail(n_rows).to_string(index=False))


def _export_if_requested(df, output_path: str | None, label: str) -> None:
    if not output_path:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Arquivo de {label} salvo em: {out}")


def _print_signal_diagnostics(signal_df) -> None:
    if signal_df is None or signal_df.empty:
        print("Diagnóstico de sinais: dataframe vazio.")
        return

    action_counts = signal_df["action"].value_counts(dropna=False).to_dict() if "action" in signal_df else {}
    reason_counts = signal_df["reason_code"].value_counts(dropna=False).to_dict() if "reason_code" in signal_df else {}

    print("\nDiagnóstico de sinais:")
    print(f"- Contagem de ações: {action_counts}")
    print(f"- Contagem de motivos: {reason_counts}")

    if "prob_long" in signal_df:
        q = signal_df["prob_long"].quantile([0.01, 0.1, 0.5, 0.9, 0.99]).to_dict()
        q_fmt = {k: round(v, 4) for k, v in q.items()}
        print(f"- Quantis de prob_long: {q_fmt}")

    non_flat = int((signal_df.get("action") != "flat").sum()) if "action" in signal_df else 0
    if non_flat == 0:
        print(
            "- Nenhum sinal acionável foi gerado. Considere afrouxar os thresholds, por exemplo: "
            "--threshold-buy 0.52 --threshold-sell 0.48"
        )

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
        use_fractional=args.use_fractional_backtest,
    )

    print("=== Pipeline WIN 5m (backtesting.py) ===")
    print(f"Linhas carregadas (bars_5m): {len(bars):,}")
    print(f"Linhas modeláveis (gold): {len(ds.dropna(subset=feature_cols + ['label_bin'])):,}")
    print(f"Colunas de features: {feature_cols}")
    print(f"Acurácias de CV: {[round(x, 4) for x in model_result.cv_accuracies]}")
    print(f"Acurácia de teste: {model_result.test_accuracy:.4f}")
    print(f"Backtesting.py Retorno [%]: {bt_result.stats.get('Return [%]')}")
    print(f"Backtesting.py Drawdown Máx. [%]: {bt_result.stats.get('Max. Drawdown [%]')}")
    print(f"Backtesting.py Nº de Trades: {bt_result.stats.get('# Trades')}")

    _print_signal_diagnostics(signal_df)
    _print_trade_summary(bt_result.trades, limit=args.show_last_trades)

    _export_if_requested(signal_df, args.export_signals_csv, label="sinais")
    _export_if_requested(bt_result.trades, args.export_trades_csv, label="trades")
    equity_curve = bt_result.equity_curve.reset_index(drop=False)
    _export_if_requested(equity_curve, args.export_equity_csv, label="curva de equity")


if __name__ == "__main__":
    main()