"""
Risk management utilities for Agentic-Trader
- Position sizing by percent risk
- ATR-based stop-loss calculation
- Exposure checks
"""
import os
from typing import Dict, Any, Sequence

try:
    import numpy as np
except Exception:
    np = None

try:
    import talib
except Exception:
    talib = None

# Config defaults (can be overridden by environment variables)
RISK_PER_TRADE_PERCENT = float(os.getenv("RISK_PER_TRADE_PERCENT", "0.01"))  # 1% default
MAX_PORTFOLIO_EXPOSURE_PERCENT = float(os.getenv("MAX_PORTFOLIO_EXPOSURE_PERCENT", "0.2"))  # 20% of equity
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "3"))
MIN_POSITION_QTY = int(os.getenv("MIN_POSITION_QTY", "1"))


def compute_atr_stoploss(high: Sequence[float], low: Sequence[float], close: Sequence[float], ltp: float, action: str, multiplier: float = ATR_MULTIPLIER) -> float:
    """Compute an ATR-based stop-loss price.

    Args:
        high/low/close: price series (iterables of floats)
        ltp: last traded price
        action: 'BUY' or 'SELL'
        multiplier: ATR multiplier to set stop distance

    Returns:
        stop_price (float)
    """
    atr_val = None
    # Prefer talib + numpy when available
    if talib is not None and np is not None:
        try:
            arr_high = np.array(high)
            arr_low = np.array(low)
            arr_close = np.array(close)
            atr_series = talib.ATR(arr_high, arr_low, arr_close, timeperiod=14)
            atr_nonan = atr_series[~np.isnan(atr_series)]
            if len(atr_nonan) > 0:
                atr_val = float(atr_nonan[-1])
        except Exception:
            atr_val = None

    # Fallback to simple range-based estimate when talib/numpy are unavailable or fail
    if atr_val is None:
        if len(close) > 1:
            ranges = [abs(float(close[i]) - float(close[i - 1])) for i in range(1, len(close))]
            atr_val = float(sum(ranges) / len(ranges)) if len(ranges) > 0 else max(0.01, ltp * 0.005)
        else:
            atr_val = max(0.01, ltp * 0.005)

    if action.upper() == "BUY":
        stop = ltp - multiplier * atr_val
    else:
        stop = ltp + multiplier * atr_val

    # Ensure stop isn't on the wrong side of price; clamp minimally
    if action.upper() == "BUY":
        stop = min(stop, ltp - 0.01)
    else:
        stop = max(stop, ltp + 0.01)

    return float(round(stop, 2))


def compute_position_size_by_risk(ltp: float, stop_price: float, account_equity: float, risk_percent: float = RISK_PER_TRADE_PERCENT, max_investment: float = None) -> Dict[str, Any]:
    """Calculate quantity based on percent risk and stop-price.

    Args:
        ltp: Last trade price
        stop_price: stop price for the trade
        account_equity: total account equity/cash to compute risk from
        risk_percent: fraction of equity to risk per trade (e.g., 0.01 = 1%)
        max_investment: cap per trade (currency); optional

    Returns:
        Dictionary with quantity, risk_amount, per_share_risk, and final capped quantity
    """
    if ltp <= 0 or account_equity <= 0:
        return {"error": "Invalid ltp or account_equity", "quantity": 0}

    per_share_risk = abs(ltp - stop_price)
    if per_share_risk <= 0:
        return {"error": "Stop price must be different from LTP", "quantity": 0}

    risk_amount = account_equity * float(risk_percent)
    qty_by_risk = int(risk_amount / per_share_risk)

    qty_by_max_investment = None
    if max_investment is not None and max_investment > 0:
        qty_by_max_investment = int(max_investment / ltp)

    # Final quantity is min of risk-based and max-investment (if provided)
    if qty_by_max_investment is not None:
        quantity = min(qty_by_risk, qty_by_max_investment)
    else:
        quantity = qty_by_risk

    # Enforce minimal quantity
    if quantity < MIN_POSITION_QTY:
        return {
            "error": "Calculated quantity below minimum",
            "quantity": 0,
            "per_share_risk": per_share_risk,
            "risk_amount": risk_amount
        }

    return {
        "quantity": int(quantity),
        "qty_by_risk": int(qty_by_risk),
        "qty_by_max_investment": int(qty_by_max_investment) if qty_by_max_investment is not None else None,
        "per_share_risk": float(per_share_risk),
        "risk_amount": float(risk_amount)
    }


def enforce_exposure_limits(current_positions: Dict[str, Any], symbol: str, proposed_quantity: int, ltp: float, max_per_symbol_investment: float) -> Dict[str, Any]:
    """Ensure proposed trade does not exceed per-symbol exposure limits.

    Args:
        current_positions: mapping symbol -> position (must contain 'quantity' and 'ltp')
        symbol: symbol being traded
        proposed_quantity: qty to add (positive for buy)
        ltp: current price
        max_per_symbol_investment: cap in currency

    Returns:
        dict with allowed bool and reason
    """
    try:
        existing = current_positions.get(symbol, {})
        existing_qty = int(existing.get("quantity", 0))
        existing_val = existing_qty * existing.get("ltp", ltp)
        proposed_val = proposed_quantity * ltp
        total_val = existing_val + proposed_val

        if total_val > max_per_symbol_investment:
            return {"allowed": False, "reason": f"Exposure limit exceeded: Rs.{total_val:.2f} > Rs.{max_per_symbol_investment:.2f}"}

        return {"allowed": True, "reason": "OK"}
    except Exception as e:
        return {"allowed": False, "reason": str(e)}


# Trailing stop utilities
def compute_trailing_stop(entry_price: float, peak_price: float, trail_percent: float = 0.03, action: str = "BUY") -> float:
    """Compute a trailing stop based on the peak price since entry.

    - For LONG (BUY): trailing_stop = peak_price * (1 - trail_percent)
    - For SHORT (SELL): trailing_stop = peak_price * (1 + trail_percent)

    Args:
        entry_price: the trade entry price (for reference)
        peak_price: the highest (for long) or lowest (for short) price observed since entry
        trail_percent: fraction (e.g., 0.03 = 3%)
        action: 'BUY' for longs, 'SELL' for shorts

    Returns:
        trailing stop price (rounded to 2 decimals)
    """
    if trail_percent <= 0:
        raise ValueError("trail_percent must be positive")

    if action.upper() == "BUY":
        ts = float(peak_price) * (1 - float(trail_percent))
        # trailing cannot be above current price
        ts = min(ts, float(peak_price) - 0.01)
    else:
        ts = float(peak_price) * (1 + float(trail_percent))
        ts = max(ts, float(peak_price) + 0.01)

    return float(round(ts, 2))


def update_trailing_stop(current_trailing: float, candidate_trailing: float, action: str = "BUY") -> float:
    """Update existing trailing stop only in a favorable direction.

    - For LONG (BUY): trailing stop may only move UP (higher)
    - For SHORT (SELL): trailing stop may only move DOWN (lower)
    """
    if current_trailing is None:
        return candidate_trailing

    if action.upper() == "BUY":
        return float(round(max(current_trailing, candidate_trailing), 2))
    else:
        return float(round(min(current_trailing, candidate_trailing), 2))
