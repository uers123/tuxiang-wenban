"""Tests for chart axis calibration logic (pure functions)."""

from __future__ import annotations

from doc_textify.chart import _is_nice_log_tick, _tick_to_data


def test_nice_log_tick_detection() -> None:
    """1–9 × 10^k mantissa values are nice log ticks; others are not."""
    assert _is_nice_log_tick(1.0)
    assert _is_nice_log_tick(2.0)
    assert _is_nice_log_tick(3.0)      # 3×10^0, integer mantissa
    assert _is_nice_log_tick(50.0)     # 5×10^1
    assert _is_nice_log_tick(100.0)    # 1×10^2
    # Non-integer mantissas are NOT nice ticks (e.g. OCR misread 10 → 16)
    assert not _is_nice_log_tick(16.0)  # 1.6×10^1
    assert not _is_nice_log_tick(7.5)   # 7.5×10^0
    assert not _is_nice_log_tick(0.0)
    assert not _is_nice_log_tick(-5.0)


def test_tick_to_data_log_axis() -> None:
    """Log-axis calibration: value = 10^(a*px + b)."""
    # a=0.01, b=0 → at px=100: 10^1=10; at px=150: 10^1.5 ≈ 31.6
    calib = {"a": 0.01, "b": 0.0}
    v = _tick_to_data(calib, 100.0, log_axis=True)
    assert v is not None
    assert abs(v - 10.0) < 0.01
    v2 = _tick_to_data(calib, 150.0, log_axis=True)
    assert v2 is not None
    assert 30.0 < v2 < 33.0


def test_tick_to_data_linear_axis() -> None:
    """Linear calibration: value = a*px + b."""
    calib = {"a": 0.04, "b": -4.0}  # px=100 → 0, px=200 → 4
    v = _tick_to_data(calib, 100.0, log_axis=False)
    assert v is not None
    assert abs(v - 0.0) < 0.01
    v2 = _tick_to_data(calib, 200.0, log_axis=False)
    assert v2 is not None
    assert abs(v2 - 4.0) < 0.01


def test_tick_to_data_bad_calibration() -> None:
    """Missing keys or exceptions should degrade to None, not crash."""
    assert _tick_to_data({}, 150.0, log_axis=True) is None
    assert _tick_to_data({"a": 0.01}, 150.0, log_axis=False) is None
