"""
MSPTD: multi-scale peak/trough detection for HR from PPG.

Reference: Bishop & Ercole, Acta Neurochirurgica Supplement, 2018.
Ported from ppg-beats MATLAB (MIT).
"""
import numpy as np

WIN_LEN_SEC = 6.0
OVERLAP = 0.2
HR_MIN_BPM = 30
HR_MAX_BPM = 200


def _detect_peaks_and_onsets_msptd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-scale LMS peak and onset detection.
    x: 1D signal, 0-based samples.
    Returns (peaks, onsets) as 0-based index arrays.
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < 10:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # PPG already detrended + bandpassed in pipeline; no extra detrend here

    L = int(np.ceil(N / 2)) - 1  # max scale
    if L < 1:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    m_max = np.zeros((L, N), dtype=bool)
    m_min = np.zeros((L, N), dtype=bool)

    for k in range(1, L + 1):
        centers = np.arange(k, N - k)
        lefts = centers - k
        rights = centers + k
        m_max[k - 1, centers] = (x[centers] > x[lefts]) & (x[centers] > x[rights])
        m_min[k - 1, centers] = (x[centers] < x[lefts]) & (x[centers] < x[rights])

    gamma_max = np.sum(m_max, axis=1)
    gamma_min = np.sum(m_min, axis=1)
    lambda_max = int(np.argmax(gamma_max)) + 1
    lambda_min = int(np.argmax(gamma_min)) + 1

    m_max = m_max[:lambda_max, :]
    m_min = m_min[:lambda_min, :]

    # peaks: columns where all scales say local max (i.e. ~m_max has zeros in that col)
    m_max_sum = np.sum(~m_max, axis=0)
    p = np.where(m_max_sum == 0)[0]
    m_min_sum = np.sum(~m_min, axis=0)
    t = np.where(m_min_sum == 0)[0]

    return p.astype(np.int64), t.astype(np.int64)


def _msptd_beat_detector(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    MSPTD entry: detect peaks and onsets.
    Returns (peaks, onsets) as 0-based sample indices.
    """
    sig = np.asarray(sig, dtype=np.float64)
    n_samp = len(sig)
    no_samps_in_win = int(WIN_LEN_SEC * fs)

    if n_samp <= no_samps_in_win:
        win_starts = [0]
        win_ends = [n_samp]
    else:
        win_offset = int(no_samps_in_win * (1 - OVERLAP))
        win_starts = list(range(0, n_samp - no_samps_in_win, win_offset))
        win_ends = [s + no_samps_in_win for s in win_starts]
        if win_ends[-1] < n_samp:
            win_starts.append(n_samp - no_samps_in_win)
            win_ends.append(n_samp)

    all_peaks = []
    all_onsets = []
    tol = max(1, int(np.ceil(fs * 0.05)))

    for ws, we in zip(win_starts, win_ends):
        win_sig = sig[ws:we]
        p, t = _detect_peaks_and_onsets_msptd(win_sig)

        for i in range(len(p)):
            curr = p[i]
            lo = max(0, curr - tol)
            hi = min(len(win_sig), curr + tol + 1)
            if lo < hi:
                local_idx = np.argmax(win_sig[lo:hi])
                p[i] = lo + local_idx

        for i in range(len(t)):
            curr = t[i]
            lo = max(0, curr - tol)
            hi = min(len(win_sig), curr + tol + 1)
            if lo < hi:
                local_idx = np.argmin(win_sig[lo:hi])
                t[i] = lo + local_idx

        all_peaks.extend((p + ws).tolist())
        all_onsets.extend((t + ws).tolist())

    peaks = np.unique(np.array(all_peaks, dtype=np.int64))
    onsets = np.unique(np.array(all_onsets, dtype=np.int64))
    return peaks, onsets


def hr_from_msptd(ppg: np.ndarray, ppg_fs: float) -> float:
    """
    HR (bpm) from one PPG window via MSPTD.
    Returns np.nan on failure.
    """
    if len(ppg) < 50:
        return np.nan
    try:
        peaks, _ = _msptd_beat_detector(ppg, ppg_fs)
        if len(peaks) < 2:
            return np.nan
        ibi_s = np.diff(peaks) / ppg_fs
        ibi_s = ibi_s[(ibi_s >= 60.0 / HR_MAX_BPM) & (ibi_s <= 60.0 / HR_MIN_BPM)]
        if len(ibi_s) == 0:
            return np.nan
        hr = 60.0 / np.mean(ibi_s)
        if hr < HR_MIN_BPM or hr > HR_MAX_BPM:
            return np.nan
        return float(hr)
    except Exception:
        return np.nan
