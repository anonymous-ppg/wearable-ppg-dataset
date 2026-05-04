"""
Re-window using coarse/fine overlap timing.

- PPG: merged npz (phone-time timestamps after merge).
- ECG: ``load_ecg_merged_recomputed`` (.npz or .csv; epoch from phone_time + rel_time_ms per source_file).
- Both-continuous windowing inside overlap regions.
- Writes alignment_windows.npz (ppg, ecg, ecg_valid_len, t0_ms, t1_ms, ppg_fs).

Optional ``--ppg-dir``: load merged PPG npz from that dir (e.g. raw timestamp alignment tests).

Helpers are sibling modules (``reader``, ``windowing``, ``ppg_ecg_overlap``); ``config`` is in this package directory.

PPG path: env ``RUN_PPG_NPZ_PATH`` or ``--ppg-npz`` (e.g. ``.../P1_Earring_raw.npz``), else ``{ppg_dir}/{DEVICE}_merged.npz``.
ECG: ``RUN_ECG_PATH``, else ``--ecg`` / resolver.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve()
_pkg_root = str(_p.parent)
for _dir in (_pkg_root,):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
from config import (
    ALIGNMENT_WINDOWS_BASE,
    ALIGNMENT_WINDOW_SEC,
    ALIGNMENT_WINDOW_STRIDE_SEC,
    DEVICE,
    ECG_MIN_SAMPLES_FRAC,
    OUTPUTS_DIR,
    PPG_WINDOW_GAP_THRESHOLD_MS,
    PPG_CHANNEL,
    SUBJECT,
)
from ppg_ecg_overlap import find_ppg_ecg_overlap_regions
from reader import (
    load_ecg_merged_recomputed,
    load_merged_npz,
    resolve_ecg_merged_path,
)
from windowing import (
    iter_time_windows_ppg_ecg,
    iter_time_windows_ppg_ecg_both_continuous,
)

# Match ECG sampling rate assumption used in run_pan_tompkins.py (min/padded window length)
_ECG_FS_EST = 130.0
# Fraction of theoretical ECG samples required inside window (from config)
_MIN_ECG_FRAC = ECG_MIN_SAMPLES_FRAC
ECG_PAD_LEN = max(1500, int(np.ceil(ALIGNMENT_WINDOW_SEC * _ECG_FS_EST * 1.15)))
MIN_ECG_LEN = max(50, int(np.floor(ALIGNMENT_WINDOW_SEC * _ECG_FS_EST * _MIN_ECG_FRAC)))

PPG_FS = 100.0
_MIN_PPG_SAMPLES = int(round(ALIGNMENT_WINDOW_SEC * PPG_FS))


def main():
    outs = OUTPUTS_DIR
    parser = argparse.ArgumentParser(description="PPG-ECG aligned windowing")
    parser.add_argument(
        "--ppg-dir",
        type=Path,
        default=None,
        help="Directory of merged PPG npz (default: config.OUTPUTS_DIR)",
    )
    parser.add_argument("--ecg-offset-ms", type=float, default=0, help="ECG time offset in milliseconds (e.g. -28800000 means ECG -8h)")
    parser.add_argument(
        "--ecg",
        type=Path,
        default=None,
        help="Merged ECG (.npz or .csv). Default: ecg_merged.npz / ecg_merged.csv or <participant>_ecg_raw.npz under OUTPUTS_DIR; else set RUN_ECG_PATH.",
    )
    parser.add_argument(
        "--ppg-npz",
        type=Path,
        default=None,
        help="Explicit PPG multi-sensor npz (e.g. P1_Earring_raw.npz); overrides RUN_PPG_NPZ_PATH and {DEVICE}_merged.npz",
    )
    args = parser.parse_args()
    ppg_dir = args.ppg_dir or outs
    env_ppg = os.environ.get("RUN_PPG_NPZ_PATH")
    if args.ppg_npz is not None:
        ppg_npz = args.ppg_npz.resolve()
    elif env_ppg:
        ppg_npz = Path(env_ppg).resolve()
    else:
        ppg_npz = (ppg_dir / f"{DEVICE}_merged.npz").resolve()

    env_ecg = os.environ.get("RUN_ECG_PATH")
    if env_ecg:
        ecg_path = Path(env_ecg).resolve()
    else:
        ecg_path = args.ecg.resolve() if args.ecg else resolve_ecg_merged_path(outs)
    if ecg_path is None:
        print(
            "[ERROR] No ECG file found. Under OUTPUTS_DIR, add ecg_merged.npz / ecg_merged.csv / "
            "<participant>_ecg_raw.npz, or set env RUN_ECG_PATH to your ECG file (batch run sets this)."
        )
        sys.exit(1)

    # ECG: recompute timestamp axis (same logic for npz as former csv merge)
    print(f"ECG file: {ecg_path}")
    ecg = load_ecg_merged_recomputed(ecg_path, subject=SUBJECT)
    ecg_t = ecg["ecg_epoch_ms"] + args.ecg_offset_ms
    ecg["ecg_epoch_ms"] = ecg_t

    # PPG: merged / raw npz (accel aligned with PPG if present)
    print(f"PPG npz: {ppg_npz}")
    ppg = load_merged_npz(ppg_npz, ppg_channel=PPG_CHANNEL, subject=SUBJECT)
    ppg_t = ppg["ppg_t_ms"]
    ppg_signal = ppg["ppg"]

    ax_all = ay_all = az_all = None
    with np.load(ppg_npz, allow_pickle=True) as zraw:
        if all(k in zraw.files for k in ("accel_x", "accel_y", "accel_z")):
            ax_all = np.asarray(zraw["accel_x"], dtype=np.float64).copy()
            ay_all = np.asarray(zraw["accel_y"], dtype=np.float64).copy()
            az_all = np.asarray(zraw["accel_z"], dtype=np.float64).copy()
            if not (len(ax_all) == len(ppg_t) == len(ay_all) == len(az_all)):
                print("[WARN] accel length mismatch vs PPG; omitting accel_* this run")
                ax_all = ay_all = az_all = None
            else:
                print("[INFO] PPG npz has accel_x/y/z; writing into windowed npz")

    overlap_regions = find_ppg_ecg_overlap_regions(ppg_t, ecg_t)

    print("=== Overlap regions ===")
    for i, (s, e, pi, ei) in enumerate(overlap_regions):
        print(f"  Region {i}: PPG{pi}&ECG{ei}, {s:.0f} - {e:.0f} ({(e-s)/1000/60:.1f} min)")

    t0_list = []
    t1_list = []
    ppg_list = []
    ppg_t_list = []
    ecg_list = []
    ecg_valid_len_list = []
    ax_list: list[np.ndarray] = []
    ay_list: list[np.ndarray] = []
    az_list: list[np.ndarray] = []

    def _collect(
        t0: float,
        t1: float,
        ppg_seg: np.ndarray,
        ppg_t_seg: np.ndarray,
        ecg_seg: np.ndarray,
        ax_seg: np.ndarray | None,
        ay_seg: np.ndarray | None,
        az_seg: np.ndarray | None,
    ) -> None:
        if len(ecg_seg) < MIN_ECG_LEN:
            return
        t0_list.append(t0)
        t1_list.append(t1)
        ppg_list.append(ppg_seg.astype(np.float64))
        ppg_t_list.append(ppg_t_seg.astype(np.float64))
        ecg_valid_len_list.append(len(ecg_seg))
        ecg_padded = np.zeros(ECG_PAD_LEN, dtype=np.float64)
        ecg_padded[: len(ecg_seg)] = ecg_seg.astype(np.float64)
        ecg_list.append(ecg_padded)
        if ax_seg is not None and ay_seg is not None and az_seg is not None:
            ax_list.append(ax_seg.astype(np.float64))
            ay_list.append(ay_seg.astype(np.float64))
            az_list.append(az_seg.astype(np.float64))

    for s, e, pi, ei in overlap_regions:
        mask = (ppg_t >= s) & (ppg_t <= e)
        ppg_t_region = ppg_t[mask]
        ppg_region = ppg_signal[mask]
        ax_r = ay_r = az_r = None
        if ax_all is not None:
            ax_r = ax_all[mask]
            ay_r = ay_all[mask]
            az_r = az_all[mask]
        if len(ppg_t_region) < _MIN_PPG_SAMPLES:
            continue
        for t0, t1, ppg_seg, ecg_seg in iter_time_windows_ppg_ecg_both_continuous(
            ppg_t_ms=ppg_t_region,
            ppg=ppg_region,
            ppg_fs=PPG_FS,
            ecg_t_ms=ecg["ecg_epoch_ms"],
            ecg_uv=ecg["ecg_uv"],
            window_sec=ALIGNMENT_WINDOW_SEC,
            stride_sec=ALIGNMENT_WINDOW_STRIDE_SEC,
            require_fixed_ppg_len=True,
            min_ecg_samples=50,
            gap_threshold_ms=PPG_WINDOW_GAP_THRESHOLD_MS,
        ):
            i0 = int(np.searchsorted(ppg_t_region, t0, side="left"))
            ppg_t_seg = ppg_t_region[i0 : i0 + len(ppg_seg)]
            if len(ppg_t_seg) != len(ppg_seg):
                ppg_t_seg = t0 + np.arange(len(ppg_seg), dtype=np.float64) * (1000.0 / PPG_FS)
            n_w = len(ppg_seg)
            ax_s = ax_r[i0 : i0 + n_w] if ax_r is not None else None
            ay_s = ay_r[i0 : i0 + n_w] if ay_r is not None else None
            az_s = az_r[i0 : i0 + n_w] if az_r is not None else None
            _collect(t0, t1, ppg_seg, ppg_t_seg, ecg_seg, ax_s, ay_s, az_s)

    # Fallback: if both-continuous yields no windows, try overlap-only
    if len(t0_list) == 0:
        print("  both-continuous produced no windows, trying overlap-only")
        for s, e, pi, ei in overlap_regions:
            mask = (ppg_t >= s) & (ppg_t <= e)
            ppg_t_region = ppg_t[mask]
            ppg_region = ppg_signal[mask]
            ax_r = ay_r = az_r = None
            if ax_all is not None:
                ax_r = ax_all[mask]
                ay_r = ay_all[mask]
                az_r = az_all[mask]
            if len(ppg_t_region) < _MIN_PPG_SAMPLES:
                continue
            for t0, t1, ppg_seg, ecg_seg in iter_time_windows_ppg_ecg(
                ppg_t_ms=ppg_t_region,
                ppg=ppg_region,
                ppg_fs=PPG_FS,
                ecg_t_ms=ecg["ecg_epoch_ms"],
                ecg_uv=ecg["ecg_uv"],
                window_sec=ALIGNMENT_WINDOW_SEC,
                stride_sec=ALIGNMENT_WINDOW_STRIDE_SEC,
                require_fixed_ppg_len=True,
                min_ecg_samples=50,
                gap_threshold_ms=PPG_WINDOW_GAP_THRESHOLD_MS,
            ):
                i0 = int(np.searchsorted(ppg_t_region, t0, side="left"))
                ppg_t_seg = ppg_t_region[i0 : i0 + len(ppg_seg)]
                if len(ppg_t_seg) != len(ppg_seg):
                    ppg_t_seg = t0 + np.arange(len(ppg_seg), dtype=np.float64) * (1000.0 / PPG_FS)
                n_w = len(ppg_seg)
                ax_s = ax_r[i0 : i0 + n_w] if ax_r is not None else None
                ay_s = ay_r[i0 : i0 + n_w] if ay_r is not None else None
                az_s = az_r[i0 : i0 + n_w] if az_r is not None else None
                _collect(t0, t1, ppg_seg, ppg_t_seg, ecg_seg, ax_s, ay_s, az_s)

    if t0_list:
        t0_ms = np.array(t0_list, dtype=np.float64)
        t1_ms = np.array(t1_list, dtype=np.float64)
        ppg = np.stack(ppg_list, axis=0)
        ppg_t_ms = np.stack(ppg_t_list, axis=0)
        ecg = np.stack(ecg_list, axis=0)
        ecg_valid_len = np.array(ecg_valid_len_list, dtype=np.int32)

        out_npz = outs / f"{ALIGNMENT_WINDOWS_BASE}.npz"
        ppg_fs_val = float(PPG_FS)
        save_kw: dict[str, object] = {
            "t0_ms": t0_ms,
            "t1_ms": t1_ms,
            "ppg": ppg,
            "ppg_t_ms": ppg_t_ms,
            "ecg": ecg,
            "ecg_valid_len": ecg_valid_len,
            "ppg_fs": ppg_fs_val,
        }
        if len(ax_list) == len(ppg_list) and ax_list:
            save_kw["accel_x"] = np.stack(ax_list, axis=0)
            save_kw["accel_y"] = np.stack(ay_list, axis=0)
            save_kw["accel_z"] = np.stack(az_list, axis=0)
        fd, tmp_path = tempfile.mkstemp(suffix=".npz", prefix=f"{out_npz.stem}_", dir=str(out_npz.parent))
        os.close(fd)
        tmp_npz = Path(tmp_path)
        try:
            np.savez_compressed(tmp_npz, **save_kw)
            os.replace(tmp_npz, out_npz)
        except BaseException:
            tmp_npz.unlink(missing_ok=True)
            raise
        print(f"\n[SAVED] {len(t0_list)} windows -> {out_npz}")
        print(f"  ppg shape: {ppg.shape}, ecg shape: {ecg.shape}")
        if "accel_x" in save_kw:
            print(f"  accel shape: {save_kw['accel_x'].shape}")
    else:
        print("\n[WARNING] No windows were produced")


if __name__ == "__main__":
    main()
