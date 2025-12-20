import os
import math

import pytest

from risk import compute_atr_stoploss, compute_position_size_by_risk, compute_trailing_stop, update_trailing_stop


def test_compute_position_size_by_risk_basic():
    ltp = 100.0
    stop = 95.0
    equity = 100000.0
    res = compute_position_size_by_risk(ltp=ltp, stop_price=stop, account_equity=equity, risk_percent=0.01, max_investment=10000)
    assert res["quantity"] > 0
    assert res["per_share_risk"] == 5.0
    assert math.isclose(res["risk_amount"], 1000.0)


def test_compute_position_size_by_risk_min_qty():
    ltp = 1000.0
    stop = 999.5
    equity = 100000.0
    res = compute_position_size_by_risk(ltp=ltp, stop_price=stop, account_equity=equity, risk_percent=0.01)
    # When per-share risk is tiny, we expect a non-zero quantity but limited by logic
    assert res.get("quantity", 0) >= 0
    assert "per_share_risk" in res


def test_compute_atr_stoploss_fallback():
    # Minimal arrays to trigger fallback without talib
    highs = [100, 101, 102, 103]
    lows = [99, 100, 101, 102]
    closes = [99.5, 100.5, 101.5, 102.5]
    stop = compute_atr_stoploss(highs, lows, closes, ltp=103.0, action="BUY", multiplier=1)
    assert stop < 103.0
    assert isinstance(stop, float)


def test_trailing_stop_basic():
    entry = 100.0
    peak = 110.0
    ts = compute_trailing_stop(entry_price=entry, peak_price=peak, trail_percent=0.05, action="BUY")
    assert ts == 104.5


def test_update_trailing_stop_moves_only_favorably():
    current = 100.0
    candidate = 102.0
    updated = update_trailing_stop(current_trailing=current, candidate_trailing=candidate, action="BUY")
    assert updated == 102.0
    # Candidate lower than current should not move it down
    candidate2 = 101.0
    updated2 = update_trailing_stop(current_trailing=updated, candidate_trailing=candidate2, action="BUY")
    assert updated2 == 102.0
