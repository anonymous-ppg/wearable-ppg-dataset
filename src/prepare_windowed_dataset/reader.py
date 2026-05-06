import shutil

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


def load_ppg_raw_with_datetime(
    ppg_csv: str | Path,
    subject: str = "unknown",
    ppg_fs: float = 100.0,
) -> Dict[str, Any]:
    """
    Load PPG raw file with datetime for coarse/fine alignment.
    Returns: subject, site, ppg_fs, ppg_t_ms, ppg, ppg_datetime_str, ppg_phone_epoch_ms
    ppg_phone_epoch_ms: epoch ms in phone time (first sample datetime as reference).
    """
    df = pd.read_csv(ppg_csv, header=None, sep=r"\s*,\s*", engine="python")
    ppg_t_ms = df.iloc[:, 1].to_numpy(dtype=np.float64)
    ppg = df.iloc[:, 4].to_numpy(dtype=np.float64)
    site = str(df.iloc[0, 3]) if len(df) > 0 else "unknown"
    order = np.argsort(ppg_t_ms)
    ppg_t_ms = ppg_t_ms[order]
    ppg = ppg[order]

    dt_str = df.iloc[order, 2].astype(str).to_numpy()
    if len(df) > 0:
        first_dt = pd.to_datetime(dt_str[0], utc=False)
        first_epoch = first_dt.value / 1e6
        first_device_ms = ppg_t_ms[0]
        ppg_phone_epoch_ms = first_epoch + (ppg_t_ms - first_device_ms)
    else:
        ppg_phone_epoch_ms = np.array([], dtype=np.float64)

    return {
        "subject": subject,
        "site": site,
        "ppg_fs": float(ppg_fs),
        "ppg_t_ms": ppg_t_ms,
        "ppg": ppg,
        "ppg_datetime_str": dt_str,
        "ppg_phone_epoch_ms": ppg_phone_epoch_ms,
    }


def load_device_ppg_csv_noheader(
    ppg_csv: str | Path,
    subject: str = "unknown",
    ppg_fs: float = 100.0,
) -> Dict[str, Any]:
    """
    Read your device PPG file with NO header.
    Example row (comma separated, may contain spaces after commas):
        uptime_ms, epoch_ms, datetime_str, site, ppg_raw

    Returns:
        dict with keys: subject, site, ppg_fs, ppg_t_ms, ppg
    """
    # Use regex separator to tolerate spaces after commas
    df = pd.read_csv(ppg_csv, header=None, sep=r"\s*,\s*", engine="python")

    # Expected columns:
    # 0: uptime_ms
    # 1: epoch_ms
    # 2: datetime_str
    # 3: site (e.g., Ring_1)
    # 4: ppg_raw
    ppg_t_ms = df.iloc[:, 1].to_numpy(dtype=np.float64)
    ppg = df.iloc[:, 4].to_numpy(dtype=np.float64)
    site = str(df.iloc[0, 3]) if len(df) > 0 else "unknown"

    # Ensure time is monotonic
    order = np.argsort(ppg_t_ms)
    ppg_t_ms = ppg_t_ms[order]
    ppg = ppg[order]

    return {
        "subject": subject,
        "site": site,
        "ppg_fs": float(ppg_fs),
        "ppg_t_ms": ppg_t_ms,
        "ppg": ppg,
    }


def load_merged_npz_with_phone_time(
    npz_path: str | Path,
    ppg_raw_path: str | Path,
    ppg_channel: str = "ppg_ir",
    subject: str = "unknown",
    ppg_fs: float = 100.0,
) -> Dict[str, Any]:
    """
    Load merged npz and convert timestamps to phone time using PPG raw datetime.
    For old data where PPG uses device clock: ppg_phone_epoch_ms = epoch(first datetime)
    + (device_ms - first device_ms).

    Returns: subject, site, ppg_fs, ppg_t_ms (phone time), ppg
    """
    raw = load_ppg_raw_with_datetime(ppg_raw_path, subject=subject, ppg_fs=ppg_fs)
    first_epoch = float(pd.to_datetime(raw["ppg_datetime_str"][0], utc=False).value / 1e6)
    first_device_ms = float(raw["ppg_t_ms"][0])

    data = load_merged_npz(npz_path, ppg_channel=ppg_channel, subject=subject, ppg_fs=ppg_fs)
    device_ts = data["ppg_t_ms"]
    data["ppg_t_ms"] = first_epoch + (device_ts - first_device_ms)
    return data


def load_merged_npz(
    npz_path: str | Path,
    ppg_channel: str = "ppg_ir",
    subject: str = "unknown",
    ppg_fs: float = 100.0,
) -> Dict[str, Any]:
    """
    Load merged device npz (from merge_ppg.py).
    Returns same interface as load_device_ppg_csv_noheader for pipeline compatibility.

    Args:
        npz_path: Path to *_merged.npz
        ppg_channel: "ppg_ir" or "ppg_green"
        subject: Subject ID
        ppg_fs: PPG sampling rate (default 100 Hz)

    Returns:
        dict with keys: subject, site, ppg_fs, ppg_t_ms, ppg
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path)

    if ppg_channel not in ("ppg_ir", "ppg_green"):
        raise ValueError("ppg_channel must be 'ppg_ir' or 'ppg_green'")

    ppg_t_ms = data["timestamp"].astype(np.float64)
    ppg = data[ppg_channel].astype(np.float64)

    # Infer site from filename, e.g. Ring_1_merged.npz -> Ring_1; P1_Earring_raw.npz -> P1_Earring
    stem = npz_path.stem
    site = stem.replace("_merged", "")
    if site.endswith("_raw"):
        site = site[: -len("_raw")]

    return {
        "subject": subject,
        "site": site,
        "ppg_fs": float(ppg_fs),
        "ppg_t_ms": ppg_t_ms,
        "ppg": ppg,
    }


def load_polar_ecg_csv_semicolon(
    ecg_csv: str | Path,
    subject: str = "unknown",
) -> Dict[str, Any]:
    """
    Read Polar raw ECG CSV with semicolon delimiter and header, like:
        Phone timestamp;sensor timestamp [ns];timestamp [ms];ecg [uV]

    We construct an absolute ECG time axis in epoch milliseconds:
        ecg_epoch_ms = epoch_ms(first_phone_timestamp) + timestamp_ms

    Returns:
        dict with keys:
          subject, ecg_epoch_ms, ecg_uv, ecg_rel_ms, phone_start_epoch_ms, ecg_fs_est
    """
    df = pd.read_csv(ecg_csv, sep=";")

    # Column names from your sample
    phone_ts_str = df["Phone timestamp"]
    rel_ms = df["timestamp [ms]"].to_numpy(dtype=np.float64)
    ecg_uv = df["ecg [uV]"].to_numpy(dtype=np.float64)

    # Parse first phone timestamp as epoch ms
    # pandas will interpret ISO strings (e.g., 2026-02-21T15:32:34.402)
    phone_dt0 = pd.to_datetime(phone_ts_str.iloc[0], utc=False)
    phone_start_epoch_ms = phone_dt0.value / 1e6  # ns -> ms

    # Build absolute time axis for ECG samples
    ecg_epoch_ms = phone_start_epoch_ms + rel_ms

    # Estimate ECG sampling rate from rel_ms (optional but helpful for peak detector later)
    if len(rel_ms) >= 2:
        dt_ms = np.diff(rel_ms)
        med_dt_ms = np.nanmedian(dt_ms)
        ecg_fs_est = float(1000.0 / med_dt_ms) if med_dt_ms > 0 else float("nan")
    else:
        ecg_fs_est = float("nan")

    return {
        "subject": subject,
        "ecg_epoch_ms": ecg_epoch_ms.astype(np.float64),
        "ecg_uv": ecg_uv.astype(np.float64),
        "ecg_rel_ms": rel_ms.astype(np.float64),
        "phone_start_epoch_ms": float(phone_start_epoch_ms),
        "ecg_fs_est": ecg_fs_est,
    }


def _finalize_ecg_sorted(
    ecg_epoch_ms: np.ndarray,
    ecg_uv: np.ndarray,
    rel_ms: np.ndarray,
    subject: str,
) -> Dict[str, Any]:
    """Sort by ecg_epoch_ms and build the standard ECG dict (same as merged CSV recomputed)."""
    order = np.argsort(ecg_epoch_ms)
    ecg_epoch_ms = ecg_epoch_ms[order]
    ecg_uv = ecg_uv[order]
    rel_ms = rel_ms[order]

    phone_start_epoch_ms = float(ecg_epoch_ms[0]) if len(ecg_epoch_ms) > 0 else float("nan")

    if len(rel_ms) >= 2:
        dt_ms = np.diff(rel_ms)
        med_dt_ms = np.nanmedian(dt_ms)
        ecg_fs_est = float(1000.0 / med_dt_ms) if med_dt_ms > 0 else float("nan")
    else:
        ecg_fs_est = float("nan")

    return {
        "subject": subject,
        "ecg_epoch_ms": ecg_epoch_ms.astype(np.float64),
        "ecg_uv": ecg_uv.astype(np.float64),
        "ecg_rel_ms": rel_ms.astype(np.float64),
        "phone_start_epoch_ms": phone_start_epoch_ms,
        "ecg_fs_est": ecg_fs_est,
    }


def load_ecg_merged_csv_recomputed(
    ecg_csv: str | Path,
    subject: str = "unknown",
) -> Dict[str, Any]:
    """
    Read merged ECG CSV and recompute epoch_ms from phone_time + rel_time_ms
    (per source_file), instead of using merge's time_ms.

    Header: phone_time, time_ms, sensor_time_ns, rel_time_ms, ecg_uv, source_file

    For each source_file: ecg_epoch_ms = epoch(first_phone_time) + rel_time_ms.

    Returns:
        dict with keys:
          subject, ecg_epoch_ms, ecg_uv, ecg_rel_ms, phone_start_epoch_ms, ecg_fs_est
    """
    df = pd.read_csv(ecg_csv, sep=",")

    ecg_epoch_ms = np.zeros(len(df), dtype=np.float64)
    for src, grp in df.groupby("source_file"):
        phone_dt0 = pd.to_datetime(grp["phone_time"].iloc[0], utc=False)
        first_epoch_ms = phone_dt0.value / 1e6
        rel = grp["rel_time_ms"].to_numpy(dtype=np.float64)
        ecg_epoch_ms[grp.index] = first_epoch_ms + rel

    ecg_uv = df["ecg_uv"].to_numpy(dtype=np.float64)
    rel_ms = df["rel_time_ms"].to_numpy(dtype=np.float64)

    return _finalize_ecg_sorted(ecg_epoch_ms, ecg_uv, rel_ms, subject)


def load_ecg_merged_npz_recomputed(
    ecg_npz: str | Path,
    subject: str = "unknown",
) -> Dict[str, Any]:
    """
    Read merged ECG NPZ with the same logical columns as ``merge_ecg`` CSV output:
      phone_time, rel_time_ms, ecg_uv, source_file
    (optional extra keys like time_ms, sensor_time_ns are ignored).

    Recomputes ``ecg_epoch_ms`` per source_file from phone_time + rel_time_ms, same as
    :func:`load_ecg_merged_csv_recomputed`.
    """
    path = Path(ecg_npz)
    with np.load(path, allow_pickle=True) as z:
        required = ("phone_time", "rel_time_ms", "ecg_uv", "source_file")
        missing = [k for k in required if k not in z.files]
        if missing:
            raise KeyError(f"{path}: missing keys {missing}; expected at least {required}")
        phone_time = z["phone_time"]
        rel_ms = np.asarray(z["rel_time_ms"], dtype=np.float64)
        ecg_uv = np.asarray(z["ecg_uv"], dtype=np.float64)
        source_file = z["source_file"]

    df = pd.DataFrame({"phone_time": phone_time, "rel_time_ms": rel_ms, "source_file": source_file})
    ecg_epoch_ms = np.zeros(len(rel_ms), dtype=np.float64)
    for _src, grp in df.groupby("source_file"):
        phone_dt0 = pd.to_datetime(grp["phone_time"].iloc[0], utc=False)
        first_epoch_ms = phone_dt0.value / 1e6
        rel = grp["rel_time_ms"].to_numpy(dtype=np.float64)
        ecg_epoch_ms[grp.index] = first_epoch_ms + rel

    return _finalize_ecg_sorted(ecg_epoch_ms, ecg_uv, rel_ms, subject)


def load_ecg_merged_recomputed(
    ecg_path: str | Path,
    subject: str = "unknown",
) -> Dict[str, Any]:
    """
    Load merged ECG from ``.npz`` (recomputed epoch axis) or ``.csv`` (recomputed).
    """
    path = Path(ecg_path)
    suf = path.suffix.lower()
    if suf == ".npz":
        return load_ecg_merged_npz_recomputed(path, subject=subject)
    if suf == ".csv":
        return load_ecg_merged_csv_recomputed(path, subject=subject)
    raise ValueError(f"Unsupported ECG file type (expected .npz or .csv): {path}")


def resolve_ecg_merged_path(data_dir: Path) -> Optional[Path]:
    """
    Locate merged ECG for runs using ``data_dir`` as the dataset folder (e.g. ``outputs/P1``).

    Resolution order:

    1. ``{data_dir}/ecg_merged.npz``
    2. ``{data_dir}/ecg_merged.csv``
    3. ``{data_dir}/{participant}_polar_ecg_raw.npz`` with ``participant = data_dir.name``.
    """
    for name in ("ecg_merged.npz", "ecg_merged.csv"):
        p = data_dir / name
        if p.is_file():
            return p
    participant_id = data_dir.name
    if participant_id:
        p = data_dir / f"{participant_id}_polar_ecg_raw.npz"
        if p.is_file():
            return p
    return None


def copy_ecg_into_dataset_output(
    participant_id: str,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """
    Copy ECG into ``output_dir`` so :func:`resolve_ecg_merged_path` finds it:
    ``input_dir/ecg_merged.npz`` or ``input_dir/ecg_merged.csv`` -> same name under ``output_dir``.
    """
    for name in ("ecg_merged.npz", "ecg_merged.csv"):
        src = input_dir / name
        if src.is_file():
            dst = output_dir / name
            shutil.copy2(src, dst)
            return dst
    raw = input_dir / f"{participant_id}_polar_ecg_raw.npz"
    if raw.is_file():
        dst = output_dir / "ecg_merged.npz"
        shutil.copy2(raw, dst)
        return dst
    raise FileNotFoundError(
        f"No ECG for participant {participant_id}: expected {input_dir / 'ecg_merged.npz'} or "
        f"{input_dir / 'ecg_merged.csv'} or {input_dir / f'{participant_id}_polar_ecg_raw.npz'}"
    )


def load_ecg_merged_csv(
    ecg_csv: str | Path,
    subject: str = "unknown",
) -> Dict[str, Any]:
    """
    Read merged ECG CSV (comma-delimited) with header:
        phone_time, time_ms, sensor_time_ns, rel_time_ms, ecg_uv, source_file

    Uses time_ms directly as epoch milliseconds for the time axis.

    Returns:
        dict with keys:
          subject, ecg_epoch_ms, ecg_uv, ecg_rel_ms, phone_start_epoch_ms, ecg_fs_est
    """
    df = pd.read_csv(ecg_csv, sep=",")

    ecg_epoch_ms = df["time_ms"].to_numpy(dtype=np.float64)
    ecg_uv = df["ecg_uv"].to_numpy(dtype=np.float64)
    rel_ms = df["rel_time_ms"].to_numpy(dtype=np.float64)

    # Ensure monotonic
    order = np.argsort(ecg_epoch_ms)
    ecg_epoch_ms = ecg_epoch_ms[order]
    ecg_uv = ecg_uv[order]
    rel_ms = rel_ms[order]

    phone_start_epoch_ms = float(ecg_epoch_ms[0]) if len(ecg_epoch_ms) > 0 else float("nan")

    # Estimate ECG sampling rate from rel_ms
    if len(rel_ms) >= 2:
        dt_ms = np.diff(rel_ms)
        med_dt_ms = np.nanmedian(dt_ms)
        ecg_fs_est = float(1000.0 / med_dt_ms) if med_dt_ms > 0 else float("nan")
    else:
        ecg_fs_est = float("nan")

    return {
        "subject": subject,
        "ecg_epoch_ms": ecg_epoch_ms.astype(np.float64),
        "ecg_uv": ecg_uv.astype(np.float64),
        "ecg_rel_ms": rel_ms.astype(np.float64),
        "phone_start_epoch_ms": phone_start_epoch_ms,
        "ecg_fs_est": ecg_fs_est,
    }