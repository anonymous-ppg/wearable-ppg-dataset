"""
NeuroKit2 PPG HR (optional: pip install neurokit2).
"""
import numpy as np

try:
    import neurokit2 as nk
except ImportError:
    nk = None  # type: ignore[misc, assignment]


def hr_from_neurokit_ppg(ppg: np.ndarray, ppg_fs: float) -> float:
    """
    ppg_clean (elgendi) + ppg_peaks (elgendi); HR (bpm) from inter-peak intervals.
    """
    if nk is None:
        return np.nan
    x = np.asarray(ppg, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) < 50:
        return np.nan
    try:
        x = nk.ppg_clean(x, sampling_rate=ppg_fs, method="elgendi")
        _, info = nk.ppg_peaks(x, sampling_rate=ppg_fs, method="elgendi", correct_artifacts=False)
        peaks = np.asarray(info.get("PPG_Peaks", []), dtype=np.int64).ravel()
        if peaks.size < 2:
            return np.nan
        ibi_ms = np.diff(peaks) / float(ppg_fs) * 1000.0
        ibi_ok = ibi_ms[(ibi_ms >= 300.0) & (ibi_ms <= 2000.0)]
        if ibi_ok.size < 2:
            return np.nan
        hr = 60000.0 / float(np.median(ibi_ok))
        if hr < 30.0 or hr > 220.0:
            return np.nan
        return float(hr)
    except Exception:
        return np.nan
