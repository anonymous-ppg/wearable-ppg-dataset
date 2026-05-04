import numpy as np
from typing import Iterator, Tuple, List


def _slice_by_time(
    t_ms: np.ndarray,
    x: np.ndarray,
    t0_ms: float,
    t1_ms: float,
) -> np.ndarray:
    """
    Slice a signal x by time interval [t0_ms, t1_ms] using searchsorted.
    Assumes t_ms is sorted ascending and aligned with x.
    """
    i0 = int(np.searchsorted(t_ms, t0_ms, side="left"))
    i1 = int(np.searchsorted(t_ms, t1_ms, side="right"))
    return x[i0:i1]


def _get_contiguous_segments(
    t_ms: np.ndarray,
    gap_threshold_ms: float,
    region_start: float,
    region_end: float,
) -> List[Tuple[float, float]]:
    """
    Find contiguous segments within [region_start, region_end] where
    consecutive timestamps have dt <= gap_threshold_ms.
    Returns list of (t_start, t_end) per segment.
    """
    mask = (t_ms >= region_start) & (t_ms <= region_end)
    t = t_ms[mask]
    if len(t) < 2:
        return []
    dt = np.diff(t)
    gap_idx = np.where(dt > gap_threshold_ms)[0]
    seg_ends = np.concatenate([[-1], gap_idx, [len(t) - 1]])
    segments = []
    for i in range(len(seg_ends) - 1):
        s, e = seg_ends[i] + 1, seg_ends[i + 1]
        if e >= s:
            segments.append((float(t[s]), float(t[e])))
    return segments


def _get_valid_regions(
    ppg_t_ms: np.ndarray,
    ecg_t_ms: np.ndarray,
    overlap_start: float,
    overlap_end: float,
    gap_threshold_ms: float = 12.0,
    segment_gap_ms: float | None = None,
) -> List[Tuple[float, float]]:
    """
    Find time regions where BOTH PPG and ECG are contiguous.
    For each ECG contiguous segment, find PPG contiguous segments within it.
    segment_gap_ms: if set, use for segment definition (larger = longer segments).
    Returns list of (t_start, t_end).
    """
    seg_gap = segment_gap_ms if segment_gap_ms is not None else gap_threshold_ms
    ecg_segs = _get_contiguous_segments(
        ecg_t_ms, seg_gap, overlap_start, overlap_end
    )
    valid = []
    for (ecg_s, ecg_e) in ecg_segs:
        ppg_segs = _get_contiguous_segments(
            ppg_t_ms, seg_gap, ecg_s, ecg_e
        )
        valid.extend(ppg_segs)
    return valid


def iter_time_windows_ppg_ecg(
    ppg_t_ms: np.ndarray,
    ppg: np.ndarray,
    ppg_fs: float,
    ecg_t_ms: np.ndarray,
    ecg_uv: np.ndarray,
    window_sec: float = 10.0,
    stride_sec: float = 1.0,
    require_fixed_ppg_len: bool = True,
    min_ecg_samples: int = 50,
    gap_threshold_ms: float = 11.0,
    dt_tolerance_ms: float = 1000.0,
) -> Iterator[Tuple[float, float, np.ndarray, np.ndarray]]:
    """
    Yield aligned windows using the same [t0, t1] boundaries:
      (t0_ms, t1_ms, ppg_seg, ecg_seg)

    PPG continuity: only yield windows where consecutive PPG samples have
    dt <= gap_threshold_ms (otherwise considered disconnection, window discarded).
    Time span check: t1-t0 must be within [expected - dt_tolerance, expected + dt_tolerance].

    All timestamps are in epoch milliseconds.
    """
    win_len_ppg = int(round(window_sec * ppg_fs))
    stride_ppg = int(round(stride_sec * ppg_fs))
    expected_dt_ms = window_sec * 1000.0
    dt_min_ms = expected_dt_ms - dt_tolerance_ms
    dt_max_ms = expected_dt_ms + dt_tolerance_ms

    n = len(ppg)
    if n < win_len_ppg:
        return

    start = 0
    while start + win_len_ppg <= n:
        end = start + win_len_ppg

        # Check PPG continuity: no gap > gap_threshold_ms within window
        gap_found = False
        for j in range(start, end - 1):
            dt = ppg_t_ms[j + 1] - ppg_t_ms[j]
            if dt > gap_threshold_ms:
                start = j + 1
                gap_found = True
                break
        if gap_found:
            continue

        t0 = float(ppg_t_ms[start])
        t1 = float(ppg_t_ms[end - 1])
        dt_ms = t1 - t0

        # Time span check
        if dt_ms < dt_min_ms or dt_ms > dt_max_ms:
            start += stride_ppg
            continue

        ppg_seg = ppg[start:end]
        if require_fixed_ppg_len and len(ppg_seg) != win_len_ppg:
            start += stride_ppg
            continue

        ecg_seg = _slice_by_time(ecg_t_ms, ecg_uv, t0, t1)
        if ecg_seg.size < min_ecg_samples:
            start += stride_ppg
            continue

        yield t0, t1, ppg_seg, ecg_seg
        start += stride_ppg


def iter_time_windows_ppg_ecg_both_continuous(
    ppg_t_ms: np.ndarray,
    ppg: np.ndarray,
    ppg_fs: float,
    ecg_t_ms: np.ndarray,
    ecg_uv: np.ndarray,
    window_sec: float = 10.0,
    stride_sec: float = 1.0,
    require_fixed_ppg_len: bool = True,
    min_ecg_samples: int = 50,
    gap_threshold_ms: float = 11.0,
    dt_tolerance_ms: float = 1000.0,
) -> Iterator[Tuple[float, float, np.ndarray, np.ndarray]]:
    """
    Same as iter_time_windows_ppg_ecg, but only windows within regions where
    BOTH PPG and ECG are contiguous. Yields (t0_ms, t1_ms, ppg_seg, ecg_seg).
    """
    overlap_start = max(float(ppg_t_ms[0]), float(ecg_t_ms[0]))
    overlap_end = min(float(ppg_t_ms[-1]), float(ecg_t_ms[-1]))
    if overlap_end <= overlap_start:
        return

    # Use larger gap for segment definition to get longer valid regions
    valid_regions = _get_valid_regions(
        ppg_t_ms, ecg_t_ms, overlap_start, overlap_end,
        gap_threshold_ms=gap_threshold_ms,
        segment_gap_ms=100.0,
    )

    for v_start, v_end in valid_regions:
        mask = (ppg_t_ms >= v_start) & (ppg_t_ms <= v_end)
        ppg_t_region = ppg_t_ms[mask]
        ppg_region = ppg[mask]
        if len(ppg_t_region) < int(round(window_sec * ppg_fs)):
            continue
        for t0, t1, ppg_seg, ecg_seg in iter_time_windows_ppg_ecg(
            ppg_t_ms=ppg_t_region,
            ppg=ppg_region,
            ppg_fs=ppg_fs,
            ecg_t_ms=ecg_t_ms,
            ecg_uv=ecg_uv,
            window_sec=window_sec,
            stride_sec=stride_sec,
            require_fixed_ppg_len=require_fixed_ppg_len,
            min_ecg_samples=min_ecg_samples,
            gap_threshold_ms=gap_threshold_ms,
            dt_tolerance_ms=dt_tolerance_ms,
        ):
            yield t0, t1, ppg_seg, ecg_seg