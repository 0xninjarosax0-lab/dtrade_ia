"""Signal contract and risk-aware decision utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskState:
    daily_trade_count: int = 0
    max_trades_per_day: int = 8
    trading_enabled: bool = True


@dataclass
class SignalDecision:
    action: str
    confidence: float
    size_multiplier: float
    reason_code: str


def infer_action(prob_long: float, threshold_buy: float, threshold_sell: float) -> str:
    if prob_long >= threshold_buy:
        return "buy"
    if prob_long <= threshold_sell:
        return "sell"
    return "flat"


def decide_signal(
    prob_long: float,
    risk_state: RiskState,
    threshold_buy: float = 0.55,
    threshold_sell: float = 0.45,
) -> SignalDecision:
    """Return standardized signal contract output."""
    if not risk_state.trading_enabled:
        return SignalDecision("flat", float(prob_long), 0.0, "RISK_DISABLED")

    if risk_state.daily_trade_count >= risk_state.max_trades_per_day:
        return SignalDecision("flat", float(prob_long), 0.0, "MAX_TRADES_REACHED")

    action = infer_action(prob_long, threshold_buy=threshold_buy, threshold_sell=threshold_sell)
    confidence = float(max(prob_long, 1 - prob_long))

    if action == "flat":
        return SignalDecision("flat", confidence, 0.0, "LOW_EDGE")

    size_multiplier = 1.0 if confidence < 0.65 else 1.25
    return SignalDecision(action, confidence, size_multiplier, "MODEL_EDGE")


def apply_signal_contract(
    df: pd.DataFrame,
    prob_col: str,
    threshold_buy: float = 0.55,
    threshold_sell: float = 0.45,
    max_trades_per_day: int = 8,
) -> pd.DataFrame:
    """Generate action/confidence/size/reason_code columns from model probabilities."""
    out = df.copy()

    actions: list[str] = []
    confidences: list[float] = []
    sizes: list[float] = []
    reasons: list[str] = []

    current_session = None
    trades_today = 0

    for _, row in out.iterrows():
        session = row["session_id"]
        if session != current_session:
            current_session = session
            trades_today = 0

        risk_state = RiskState(
            daily_trade_count=trades_today,
            max_trades_per_day=max_trades_per_day,
            trading_enabled=True,
        )

        decision = decide_signal(
            prob_long=float(row[prob_col]),
            risk_state=risk_state,
            threshold_buy=threshold_buy,
            threshold_sell=threshold_sell,
        )

        actions.append(decision.action)
        confidences.append(decision.confidence)
        sizes.append(decision.size_multiplier)
        reasons.append(decision.reason_code)

        if decision.action in {"buy", "sell"}:
            trades_today += 1

    out["action"] = actions
    out["confidence"] = np.array(confidences)
    out["size_multiplier"] = np.array(sizes)
    out["reason_code"] = reasons
    return out
