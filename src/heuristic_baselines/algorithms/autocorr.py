"""
Autocorrelation peak for HR from PPG (use bandpassed PPG for best results).
"""
import numpy as np

# HR valid range 30-180 bpm -> period 0.33-2 s
HR_MIN_BPM = 30
HR_MAX_BPM = 180
PERIOD_MIN_MS = 60000.0 / HR_MAX_BPM  # 333 ms
PERIOD_MAX_MS = 60000.0 / HR_MIN_BPM  # 2000 ms

# Zero-pad length (same idea as FFT) for lag resolution and parabolic interpolation
PAD_LEN = 8192


def hr_from_autocorr(ppg: np.ndarray, ppg_fs: float) -> float:
    """
    First autocorrelation peak in valid lag -> HR (bpm).
    Input should be bandpassed PPG for best results.
    Returns np.nan on failure.
    """
    if len(ppg) < 50:
        return np.nan
    try:
        n = len(ppg)
        windowed = ppg.astype(np.float64) * np.hanning(n)
        padded = np.pad(windowed, (0, PAD_LEN - n), mode="constant", constant_values=0)
        # Autocorr mode='full' length 2*PAD_LEN-1; zero lag at center index
        corr = np.correlate(padded, padded, mode="full")
        # Positive lags: indices PAD_LEN-1 .. 2*PAD_LEN-2 -> lag 0 .. PAD_LEN-1
        corr_pos = corr[PAD_LEN - 1 :]
        lags = np.arange(len(corr_pos), dtype=np.float64)
        # Valid lag: period 0.33-2 s -> e.g. lag 33-200 at 100 Hz
        lag_min = int(np.ceil(ppg_fs * PERIOD_MIN_MS / 1000))
        lag_max = min(int(np.floor(ppg_fs * PERIOD_MAX_MS / 1000)), len(corr_pos) - 1)
        if lag_min >= lag_max:
            return np.nan
        corr_region = corr_pos[lag_min : lag_max + 1]
        peak_offset = np.argmax(corr_region)
        peak_idx = lag_min + peak_offset
        # Parabolic interpolation around peak
        alpha = corr_pos[peak_idx - 1] if peak_idx > 0 else corr_pos[peak_idx]
        beta = corr_pos[peak_idx]
        gamma = corr_pos[peak_idx + 1] if peak_idx < len(corr_pos) - 1 else corr_pos[peak_idx]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-10:
            delta = 0.5 * (alpha - gamma) / denom
            delta = np.clip(delta, -0.5, 0.5)
        else:
            delta = 0.0
        lag_samp = peak_idx + delta
        period_ms = lag_samp / ppg_fs * 1000
        hr = 60000.0 / period_ms
        if hr < HR_MIN_BPM or hr > HR_MAX_BPM:
            return np.nan
        return float(hr)
    except Exception:
        return np.nan
