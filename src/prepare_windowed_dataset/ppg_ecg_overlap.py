"""
PPG / ECG continuous segments and overlap regions (same rules as run_windowing_aligned).

- PPG segments: gap between consecutive samples > ppg_gap_ms is a break (default 1 h).
- ECG segments: gap > ecg_gap_ms is a break (default 100 ms).
- Overlap: intersection of segments; returns (start_ms, end_ms, ppg_idx, ecg_idx) per region.
"""
from __future__ import annotations

import numpy as np


def _sanitize_timestamps(t_ms: np.ndarray) -> np.ndarray:
    """Drop NaN/Inf, sort ascending (avoids bad samples breaking segment detection)."""
    x = np.asarray(t_ms, dtype=np.float64)
    if x.size == 0:
        return x
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    return np.sort(x)


def get_continuous_segments(t_ms: np.ndarray, gap_threshold_ms: float) -> list[tuple[float, float]]:
    """Split into contiguous segments by gap; returns [(t_start, t_end), ...] at sample endpoints."""
    ts = _sanitize_timestamps(t_ms)
    if len(ts) < 2:
        return [(float(ts[0]), float(ts[0]))] if len(ts) == 1 else []
    dt = np.diff(ts)
    gaps = np.where(dt > gap_threshold_ms)[0]
    bounds = np.concatenate([[0], gaps + 1, [len(ts)]])
    return [
        (float(ts[bounds[i]]), float(ts[bounds[i + 1] - 1]))
        for i in range(len(bounds) - 1)
    ]


def segments_overlap(seg_a: tuple[float, float], seg_b: tuple[float, float]) -> bool:
    return seg_a[0] <= seg_b[1] and seg_b[0] <= seg_a[1]


def find_ppg_ecg_overlap_regions(
    ppg_t_ms: np.ndarray,
    ecg_t_ms: np.ndarray,
    *,
    ppg_gap_ms: float = 3600000.0,
    ecg_gap_ms: float = 100.0,
) -> list[tuple[float, float, int, int]]:
    """
    Same overlap logic as run_windowing_aligned.
    Each item: (overlap_start_ms, overlap_end_ms, ppg_segment_index, ecg_segment_index).
    """
    ppg_segs = get_continuous_segments(ppg_t_ms, ppg_gap_ms)
    ecg_segs = get_continuous_segments(ecg_t_ms, ecg_gap_ms)
    overlap_regions: list[tuple[float, float, int, int]] = []
    for pi, ps in enumerate(ppg_segs):
        for ei, es in enumerate(ecg_segs):
            if segments_overlap(ps, es):
                s = max(ps[0], es[0])
                e = min(ps[1], es[1])
                if e > s:
                    overlap_regions.append((s, e, pi, ei))
    return overlap_regions
