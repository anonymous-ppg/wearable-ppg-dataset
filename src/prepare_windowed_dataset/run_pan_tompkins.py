"""
R-peaks and ECG quality on alignment_windows.npz segments; compute hr_gt and append to NPZ.

Requires: pip install neurokit2

Run with this package directory on ``sys.path`` (see ``run_pipeline.py``).
"""
import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import welch

_p = Path(__file__).resolve()
_pkg_root = str(_p.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
from config import ALIGNMENT_WINDOWS_BASE, ECG_FS, OUTPUTS_DIR

try:
    import neurokit2 as nk
except ImportError:
    print("Please install neurokit2: pip install neurokit2")
    sys.exit(1)

# Common thresholds (tune here)
MIN_HR = 40.0
MAX_HR = 180.0
MIN_BEATS = 4
RR_LOW_SEC = 0.25
RR_HIGH_SEC = 2.0


def _safe_quality_label(value: object) -> str:
    if isinstance(value, str):
        return value
    if np.isscalar(value):
        return str(value)
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        return "unknown"
    return str(arr[0])


# def ecg_window_check(ecg_uv: np.ndarray, sampling_rate: float) -> tuple[float, int, str, int, str]:
#     """
#     Returns:
#     - hr_gt: value only when valid, otherwise NaN
#     - n_peaks
#     - quality_label
#     - valid_flag (0/1)
#     - invalid_reason
#     """
#
#     # Extra thresholds (tune as needed)
#     MAX_RR_CV = 0.35          # irregularity threshold
#     MIN_QRS_PROM = 100.0      # in uV, adjust to your ECG amplitude scale
#     MAX_HF_RATIO = 0.40       # fraction of power in 20-40 Hz band
#
#     ecg_valid = ecg_uv[~np.isnan(ecg_uv)]
#     if len(ecg_valid) < 50:
#         return np.nan, 0, "unknown", 0, "too_short"
#
#     try:
#         ecg_cleaned = nk.ecg_clean(
#             ecg_valid,
#             sampling_rate=sampling_rate,
#             method="pantompkins1985"
#         )
#
#         quality_raw = nk.ecg_quality(
#             ecg_cleaned,
#             sampling_rate=sampling_rate,
#             method="zhao2018",
#             approach="fuzzy"
#         )
#         quality_label = _safe_quality_label(quality_raw)
#
#         _, info = nk.ecg_peaks(
#             ecg_cleaned,
#             sampling_rate=sampling_rate,
#             method="pantompkins1985"
#         )
#         peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int32)
#         n_peaks = int(peaks.size)
#
#         if n_peaks < MIN_BEATS:
#             return np.nan, n_peaks, quality_label, 0, "too_few_beats"
#
#         rr_sec = np.diff(peaks) / sampling_rate
#         if rr_sec.size == 0:
#             return np.nan, n_peaks, quality_label, 0, "no_rr"
#
#         hr = 60.0 / float(np.median(rr_sec))
#         if hr < MIN_HR or hr > MAX_HR:
#             return np.nan, n_peaks, quality_label, 0, "implausible_hr"
#
#         if np.any(rr_sec < RR_LOW_SEC) or np.any(rr_sec > RR_HIGH_SEC):
#             return np.nan, n_peaks, quality_label, 0, "implausible_rr"
#
#         # --- 1) Additional irregularity check ---
#         rr_mean = float(np.mean(rr_sec))
#         rr_std = float(np.std(rr_sec))
#         rr_cv = rr_std / rr_mean if rr_mean > 0 else np.inf
#         if rr_cv > MAX_RR_CV:
#             return np.nan, n_peaks, quality_label, 0, "rr_too_irregular"
#
#         # --- 2) QRS prominence check ---
#         # R peaks should stand out relative to the local / global baseline
#         peak_vals = ecg_cleaned[peaks]
#         baseline = np.median(ecg_cleaned)
#         qrs_prom = float(np.median(np.abs(peak_vals - baseline)))
#         if qrs_prom < MIN_QRS_PROM:
#             return np.nan, n_peaks, quality_label, 0, "weak_qrs"
#
#         # --- 3) Frequency-domain high-frequency noise check ---
#         nperseg = min(len(ecg_cleaned), 256)
#         freqs, pxx = welch(ecg_cleaned, fs=sampling_rate, nperseg=nperseg)
#
#         band_mask = (freqs >= 0.5) & (freqs <= 40.0)
#         hf_mask = (freqs >= 20.0) & (freqs <= 40.0)
#
#         total_power = np.trapz(pxx[band_mask], freqs[band_mask]) if np.any(band_mask) else 0.0
#         hf_power = np.trapz(pxx[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0.0
#         hf_ratio = hf_power / (total_power + 1e-12)
#
#         if hf_ratio > MAX_HF_RATIO:
#             return np.nan, n_peaks, quality_label, 0, "too_much_high_freq_noise"
#
#         if quality_label.lower() == "unacceptable":
#             return np.nan, n_peaks, quality_label, 0, "poor_quality"
#
#         return float(hr), n_peaks, quality_label, 1, "ok"
#
#     except Exception:
#         return np.nan, 0, "unknown", 0, "exception"


def ecg_window_check(ecg_uv: np.ndarray, sampling_rate: float) -> tuple[float, int, str, int, str]:
    """
    Returns:
    - hr_gt: value only when valid, otherwise NaN
    - n_peaks
    - quality_label
    - valid_flag (0/1)
    - invalid_reason
    """
    MAX_RR_CV = 0.2
    if np.isnan(ecg_uv).any():
        return np.nan, 0, "unknown", 0, "contains_nan"
    ecg_valid = ecg_uv

    if len(ecg_valid) < 50:
        return np.nan, 0, "unknown", 0, "too_short"

    try:
        ecg_cleaned = nk.ecg_clean(ecg_valid, sampling_rate=sampling_rate, method="pantompkins1985")
        quality_raw = nk.ecg_quality(ecg_cleaned, sampling_rate=sampling_rate, method="zhao2018", approach="fuzzy")
        quality_label = _safe_quality_label(quality_raw)

        _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method="pantompkins1985")
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int32)
        n_peaks = int(peaks.size)

        if n_peaks < MIN_BEATS:
            return np.nan, n_peaks, quality_label, 0, "too_few_beats"

        rr_sec = np.diff(peaks) / sampling_rate
        if rr_sec.size == 0:
            return np.nan, n_peaks, quality_label, 0, "no_rr"

        hr = 60.0 / float(np.median(rr_sec))
        if hr < MIN_HR or hr > MAX_HR:
            return np.nan, n_peaks, quality_label, 0, "implausible_hr"

        if np.any(rr_sec < RR_LOW_SEC) or np.any(rr_sec > RR_HIGH_SEC):
            return np.nan, n_peaks, quality_label, 0, "implausible_rr"

        rr_mean = float(np.mean(rr_sec))
        rr_std = float(np.std(rr_sec))
        rr_cv = rr_std / rr_mean if rr_mean > 0 else np.inf
        if rr_cv > MAX_RR_CV:
            return np.nan, n_peaks, quality_label, 0, "rr_too_irregular"

        if quality_label.lower() == "unacceptable":
            return np.nan, n_peaks, quality_label, 0, "poor_quality"

        return float(hr), n_peaks, quality_label, 1, "ok"
    except Exception:
        return np.nan, 0, "unknown", 0, "exception"


def main():
    data_dir = OUTPUTS_DIR
    npz_path = data_dir / f"{ALIGNMENT_WINDOWS_BASE}.npz"

    if not npz_path.exists():
        print(f"Please run run_windowing_aligned.py first to generate {npz_path}")
        sys.exit(1)

    core_keys = frozenset({"t0_ms", "t1_ms", "ppg", "ecg", "ecg_valid_len", "ppg_fs"})
    try:
        with np.load(npz_path) as z:
            t0_ms = np.asarray(z["t0_ms"])
            t1_ms = np.asarray(z["t1_ms"])
            ppg = np.asarray(z["ppg"])
            ecg = np.asarray(z["ecg"])
            ecg_valid_len = np.asarray(z["ecg_valid_len"])
            ppg_fs = float(np.asarray(z["ppg_fs"])) if "ppg_fs" in z.files else 100.0
            extra_raw: dict[str, np.ndarray] = {}
            for k in z.files:
                if k in core_keys:
                    continue
                extra_raw[k] = np.asarray(z[k])
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        print(
            "[ERROR] alignment_windows NPZ is corrupt or incompletely written (ZIP read failed).\n"
            f"  File: {npz_path}\n"
            f"  Detail: {exc!r}\n"
            "  Delete this file and re-run windowing (or full pipeline: python run_pipeline.py from src/prepare_windowed_dataset)."
        )
        sys.exit(1)

    n = len(t0_ms)

    print(f"Windows: {n}, ecg_fs: {ECG_FS} Hz")

    keep_indices = []
    hr_gt_keep = []
    n_peaks_keep = []
    for i in range(n):
        valid_len = int(ecg_valid_len[i])
        ecg_seg = ecg[i, :valid_len]
        hr, n_p, _quality_label, valid_flag, _invalid_reason = ecg_window_check(ecg_seg, ECG_FS)
        if valid_flag == 1:
            keep_indices.append(i)
            hr_gt_keep.append(hr)
            n_peaks_keep.append(n_p)

    keep_indices_arr = np.array(keep_indices, dtype=np.int32)
    kept_n = int(keep_indices_arr.size)
    dropped_n = n - kept_n

    if kept_n == 0:
        print("No valid ECG windows found. Nothing was written. Please check thresholds or input quality.")
        sys.exit(1)

    t0_ms_keep = t0_ms[keep_indices_arr]
    t1_ms_keep = t1_ms[keep_indices_arr]
    ppg_keep = ppg[keep_indices_arr]
    ecg_keep = ecg[keep_indices_arr]
    ecg_valid_len_keep = ecg_valid_len[keep_indices_arr]
    hr_gt = np.array(hr_gt_keep, dtype=np.float64)
    n_peaks = np.array(n_peaks_keep, dtype=np.int32)

    print(f"Kept windows: {kept_n}/{n} ({100*kept_n/n:.1f}%)")
    print(f"Dropped invalid windows: {dropped_n}/{n} ({100*dropped_n/n:.1f}%)")

    out_path = npz_path
    save_kw: dict[str, np.ndarray] = {
        "t0_ms": t0_ms_keep,
        "t1_ms": t1_ms_keep,
        "ppg": ppg_keep,
        "ecg": ecg_keep,
        "ecg_valid_len": ecg_valid_len_keep,
        "ppg_fs": ppg_fs,
        "hr_gt": hr_gt,
        "n_peaks": n_peaks,
    }
    skip_copy = frozenset(save_kw.keys())
    for k, arr in extra_raw.items():
        if k in skip_copy:
            continue
        if arr.ndim >= 1 and arr.shape[0] == n:
            save_kw[k] = arr[keep_indices_arr]
    fd, tmp_str = tempfile.mkstemp(suffix=".npz", prefix=f"{out_path.stem}_", dir=str(out_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_str)
    try:
        np.savez_compressed(tmp_path, **save_kw)
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    print(f"\n[SAVED] Valid-only windows with hr_gt and n_peaks -> {out_path}")

    # Save filtering summary for batch-run quality tracking
    dropped_pct = 100.0 * dropped_n / n if n else 0.0
    kept_pct = 100.0 * kept_n / n if n else 0.0
    summary_path = data_dir / f"pan_tompkins_window_filter_summary_{ALIGNMENT_WINDOWS_BASE}.txt"
    lines = [
        "Pan-Tompkins window filtering summary",
        f"npz_path: {out_path}",
        "",
        f"total_windows_before: {n}",
        f"kept_valid_windows: {kept_n} ({kept_pct:.1f}%)",
        f"dropped_invalid_windows: {dropped_n} ({dropped_pct:.1f}%)",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SAVED] Window filtering summary -> {summary_path}")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(
        description="R-peaks and hr_gt on alignment_windows NPZ (config OUTPUTS_DIR + ALIGNMENT_WINDOWS_BASE; set RUN_* when spawned from run_pipeline).",
    )
    _ap.parse_args()
    main()
