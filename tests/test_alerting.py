"""Unit tests for the consecutive-window alert rule (theta = 0.82, K = 3)."""

from __future__ import annotations

import pytest

from src.alerting import find_alert_segments


def run(scores, k=3, theta=0.82):
    starts = [0.5 * i for i in range(len(scores))]
    ends = [s + 2.0 for s in starts]
    return find_alert_segments(scores, starts, ends, theta, k)


def test_no_alert_when_all_below_threshold():
    assert run([0.1, 0.5, 0.81, 0.82, 0.3]) == []


def test_threshold_is_strict_greater_than():
    assert run([0.82, 0.82, 0.82]) == []
    assert len(run([0.8201, 0.83, 0.9])) == 1


def test_isolated_spikes_do_not_alert():
    assert run([0.99, 0.1, 0.99, 0.99, 0.1, 0.95]) == []


def test_exactly_k_consecutive_windows_alert():
    (seg,) = run([0.1, 0.9, 0.95, 0.91, 0.2])
    assert (seg.start_window, seg.end_window) == (1, 3)
    assert seg.start_s == 0.5 and seg.end_s == pytest.approx(1.5 + 2.0)
    assert seg.max_score == 0.95


def test_trailing_run_is_closed():
    (seg,) = run([0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
    assert (seg.start_window, seg.end_window) == (2, 5)


def test_multiple_segments():
    segs = run([0.9] * 3 + [0.1] + [0.95] * 4)
    assert [(s.start_window, s.end_window) for s in segs] == [(0, 2), (4, 7)]


def test_k_equals_one_flags_every_crossing():
    assert len(run([0.9, 0.1, 0.9], k=1)) == 2


def test_invalid_inputs():
    with pytest.raises(ValueError):
        find_alert_segments([0.9], [0.0, 0.5], [2.0], 0.82, 3)
    with pytest.raises(ValueError):
        find_alert_segments([0.9], [0.0], [2.0], 0.82, 0)
