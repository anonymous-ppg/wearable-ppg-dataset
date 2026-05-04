"""
FFT peak frequency for HR from PPG (use bandpassed PPG for best results).
"""
import numpy as np

# HR valid range 30-180 bpm -> 0.5-3 Hz
HR_MIN_BPM = 30
HR_MAX_BPM = 180
FREQ_MIN_HZ = HR_MIN_BPM / 60.0
FREQ_MAX_HZ = HR_MAX_BPM / 60.0

# Zero-pad to 8192: df ~ 0.012 Hz ~ 0.7 bpm
FFT_PAD_LEN = 8192


def hr_from_fft(ppg: np.ndarray, ppg_fs: float) -> float:
    """
    Dominant FFT bin in 0.5-3 Hz -> HR (bpm).
    Input should be bandpassed PPG for best results.
    Returns np.nan on failure.
    """
    if len(ppg) < 50:
        return np.nan
    try:
        n = len(ppg)
        windowed = ppg * np.hanning(n)
        padded = np.pad(windowed, (0, FFT_PAD_LEN - n), mode="constant", constant_values=0)
        fft_vals = np.abs(np.fft.rfft(padded))
        freqs = np.fft.rfftfreq(FFT_PAD_LEN, 1.0 / ppg_fs)
        # Peak search only inside 0.5-3 Hz
        mask = (freqs >= FREQ_MIN_HZ) & (freqs <= FREQ_MAX_HZ)
        if not np.any(mask):
            return np.nan
        fft_masked = np.where(mask, fft_vals, 0)
        peak_idx = np.argmax(fft_masked)
        # Parabolic interp: sub-bin peak frequency from peak and neighbors
        alpha = fft_vals[peak_idx - 1] if peak_idx > 0 else fft_vals[peak_idx]
        beta = fft_vals[peak_idx]
        gamma = fft_vals[peak_idx + 1] if peak_idx < len(fft_vals) - 1 else fft_vals[peak_idx]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-10:
            delta = 0.5 * (alpha - gamma) / denom
            delta = np.clip(delta, -0.5, 0.5)  # keep delta in [-0.5, 0.5]
        else:
            delta = 0.0
        bin_width = ppg_fs / FFT_PAD_LEN
        peak_freq = (peak_idx + delta) * bin_width
        hr = peak_freq * 60.0
        if hr < HR_MIN_BPM or hr > HR_MAX_BPM:
            return np.nan
        return float(hr)
    except Exception:
        return np.nan
