"""PPG detrend + bandpass (0.7–3.5 Hz) for windowed NPZ."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, filtfilt

BAND_LOW_HZ = 0.7
BAND_HIGH_HZ = 3.5


def bandpass_filter(x: np.ndarray, low_hz: float, high_hz: float, fs: float, order: int = 2) -> np.ndarray:
    nyq = fs / 2
    low = max(0.01, low_hz / nyq)
    high = min(0.99, high_hz / nyq)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x.astype(np.float64))


def preprocess_ppg(ppg: np.ndarray, ppg_fs: float) -> np.ndarray:
    """Linear detrend + bandpass per window row."""
    out = np.zeros_like(ppg, dtype=np.float64)
    for i in range(ppg.shape[0]):
        x = detrend(ppg[i].astype(np.float64), type="linear")
        out[i] = bandpass_filter(x, BAND_LOW_HZ, BAND_HIGH_HZ, ppg_fs)
    return out


def write_preprocess_npz(src_npz: Path, out_npz: Path) -> None:
    """Read merged alignment_windows npz, filter ppg_ir/green, set ppg_preprocessed, save."""
    data = dict(np.load(src_npz, allow_pickle=True))
    ppg_fs = float(np.asarray(data["ppg_fs"]).item()) if "ppg_fs" in data else 100.0
    if "ppg_ir" in data:
        data["ppg_ir"] = preprocess_ppg(np.asarray(data["ppg_ir"]), ppg_fs)
    if "ppg_green" in data:
        data["ppg_green"] = preprocess_ppg(np.asarray(data["ppg_green"]), ppg_fs)
    if "ppg" in data and "ppg_ir" not in data and "ppg_green" not in data:
        data["ppg"] = preprocess_ppg(np.asarray(data["ppg"]), ppg_fs)
    data["ppg_preprocessed"] = np.array(True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".npz", prefix=out_npz.stem + "_", dir=str(out_npz.parent))
    os.close(fd)
    tmp_p = Path(tmp)
    try:
        np.savez_compressed(tmp_p, **data)
        os.replace(tmp_p, out_npz)
    except BaseException:
        tmp_p.unlink(missing_ok=True)
        raise
