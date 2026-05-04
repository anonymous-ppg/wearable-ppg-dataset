"""
HeartPy HR from PPG (optional dependency: pip install heartpy).
"""
import numpy as np

try:
    import heartpy as hp
except ImportError:
    hp = None  # type: ignore[misc, assignment]


def hr_from_heartpy(ppg: np.ndarray, ppg_fs: float) -> float:
    """
    Run HeartPy on PPG; returns HR (bpm).
    """
    if hp is None:
        return np.nan
    if len(ppg) < 50:
        return np.nan
    try:
        scaled = hp.scale_data(ppg.astype(np.float64), lower=0, upper=1024)
        # Match MATLAB heartpy_beat_detector: enhance_peaks before peak detection
        enhanced = hp.enhance_peaks(scaled, iterations=2)
        wd, m = hp.process(enhanced, ppg_fs, bpmmin=40, bpmmax=180, windowsize=0.75)
        hr = m.get("bpm")
        if hr is None or np.isnan(hr):
            return np.nan
        hr = float(hr)
        if hr < 30 or hr > 200:
            return np.nan
        return hr
    except Exception:
        return np.nan
