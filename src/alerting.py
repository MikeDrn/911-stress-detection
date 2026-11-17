"""Dispatcher alert rule.

A single loud window (a door slam, a cough) must not page a dispatcher. An
alert is raised only when at least ``consecutive_windows`` *consecutive*
windows score strictly above ``threshold`` (theta). Each maximal run of such
windows becomes one alert segment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AlertSegment:
    start_window: int
    end_window: int  # inclusive
    start_s: float
    end_s: float
    max_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def find_alert_segments(
    scores: Sequence[float],
    starts: Sequence[float],
    ends: Sequence[float],
    threshold: float = 0.82,
    consecutive_windows: int = 3,
) -> list[AlertSegment]:
    if not (len(scores) == len(starts) == len(ends)):
        raise ValueError("scores, starts and ends must have equal length")
    if consecutive_windows < 1:
        raise ValueError("consecutive_windows must be >= 1")

    segments: list[AlertSegment] = []
    run_start: int | None = None
    for i, s in enumerate([*scores, float("-inf")]):  # sentinel closes a trailing run
        if s > threshold:
            if run_start is None:
                run_start = i
            continue
        if run_start is not None:
            run_end = i - 1
            if run_end - run_start + 1 >= consecutive_windows:
                segments.append(
                    AlertSegment(
                        start_window=run_start,
                        end_window=run_end,
                        start_s=float(starts[run_start]),
                        end_s=float(ends[run_end]),
                        max_score=float(max(scores[run_start : run_end + 1])),
                    )
                )
            run_start = None
    return segments
