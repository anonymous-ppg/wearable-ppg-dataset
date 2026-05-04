"""
Pipeline: windowing → Pan-Tompkins; optional batch over participants/devices via ``config.py``,
then merge IR+GREEN. If the PPG npz contains accel_x/y/z, those are written through windowing so
final alignment_windows_<device>.npz includes accel (compressed). After merge, per-channel
``*_ppg_ir.npz`` / ``*_ppg_green.npz`` intermediates are deleted; only the merged file remains.

Usage (from ``src/prepare_windowed_dataset``, i.e. this package directory)::

    python run_pipeline.py

Participants and wearables come only from ``config.py``. Child steps are spawned with
``_DATALOADER_CHILD=1`` (set by batch logic; do not set by hand).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parent
for _root in (_PACKAGE_ROOT,):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
from config import (  # noqa: E402
    ALIGNMENT_WINDOWS_BASE,
    DEVICE,
    DATALOADER_OUTPUT_ROOT,
    DATALOADER_PIPELINE_PARTICIPANTS,
    ECG_FS,
    OUTPUTS_DIR,
    PPG_CHANNEL,
    WEARABLE_DEVICE_ROLES,
    dataloader_ecg_raw_npz,
    dataloader_ppg_raw_npz,
    normalize_participant_id,
    participant_output_device,
)


def default_run_label() -> str:
    """Example: Earring_2 + ppg_ir -> earring2_ir."""
    dev = DEVICE.replace("_", "").lower()
    ch = PPG_CHANNEL.removeprefix("ppg_")
    return f"{dev}_{ch}"


def run(cmd: list[str], desc: str, *, env: dict[str, str] | None = None) -> bool:
    """Run command and return False on failure."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    print("=" * 60)
    full_env = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, cwd=_PACKAGE_ROOT, env=full_env)
    if r.returncode != 0:
        print(f"\n[FAILED] {desc} (exit {r.returncode})")
        return False
    return True


def effective_window_coverage_sec(npz_path: Path) -> float:
    """Union of [t0_ms, t1_ms] intervals; returns deduplicated coverage duration in seconds."""
    if not npz_path.exists():
        return 0.0
    data = np.load(npz_path, allow_pickle=True)
    if "t0_ms" not in data or "t1_ms" not in data:
        return 0.0
    t0 = np.asarray(data["t0_ms"], dtype=np.float64)
    t1 = np.asarray(data["t1_ms"], dtype=np.float64)
    if len(t0) == 0 or len(t1) == 0:
        return 0.0

    intervals = sorted((float(a), float(b)) for a, b in zip(t0, t1) if np.isfinite(a) and np.isfinite(b) and b > a)
    if not intervals:
        return 0.0

    merged_iv: list[list[float]] = [[intervals[0][0], intervals[0][1]]]
    for s, e in intervals[1:]:
        if s <= merged_iv[-1][1]:
            merged_iv[-1][1] = max(merged_iv[-1][1], e)
        else:
            merged_iv.append([s, e])
    total_ms = sum(e - s for s, e in merged_iv)
    return total_ms / 1000.0


def npz_pipeline_stats(npz_path: Path) -> tuple[int, int | None, str | None]:
    if not npz_path.exists():
        return 0, None, None
    data = np.load(npz_path, allow_pickle=True)
    if "t0_ms" not in data:
        return 0, None, None
    n = int(len(np.asarray(data["t0_ms"])))
    if "hr_gt" not in data:
        return n, None, None
    hr_gt = np.asarray(data["hr_gt"], dtype=np.float64)
    valid_count = int(np.sum(~np.isnan(hr_gt)))
    pct = 100.0 * valid_count / n if n else 0.0
    ratio_line = f"{valid_count}/{n} ({pct:.1f}%)"
    return n, valid_count, ratio_line


def _window_key_pairs(t0_ms: np.ndarray, t1_ms: np.ndarray) -> list[tuple[int, int]]:
    t0 = np.asarray(t0_ms, dtype=np.float64).ravel()
    t1 = np.asarray(t1_ms, dtype=np.float64).ravel()
    if t0.shape != t1.shape:
        raise ValueError("t0_ms and t1_ms have different shapes")
    return list(zip(np.rint(t0).astype(np.int64), np.rint(t1).astype(np.int64)))


def _align_npz_to_reference_windows(npz_path: Path, ref_npz_path: Path) -> tuple[bool, str]:
    if not npz_path.exists():
        return False, f"target npz not found: {npz_path.name}"
    if not ref_npz_path.exists():
        return False, f"reference npz not found: {ref_npz_path.name}"

    src = np.load(npz_path, allow_pickle=True)
    ref = np.load(ref_npz_path, allow_pickle=True)
    try:
        if "t0_ms" not in src or "t1_ms" not in src:
            return False, f"target missing window keys: {npz_path.name}"
        if "t0_ms" not in ref or "t1_ms" not in ref:
            return False, f"reference missing window keys: {ref_npz_path.name}"
        if "hr_gt" not in ref:
            return False, f"reference has no hr_gt: {ref_npz_path.name}"

        src_t0 = np.asarray(src["t0_ms"], dtype=np.float64)
        src_t1 = np.asarray(src["t1_ms"], dtype=np.float64)
        ref_t0 = np.asarray(ref["t0_ms"], dtype=np.float64)
        ref_t1 = np.asarray(ref["t1_ms"], dtype=np.float64)

        if src_t0.shape == ref_t0.shape and np.allclose(src_t0, ref_t0, equal_nan=True) and np.allclose(src_t1, ref_t1, equal_nan=True):
            return False, f"already aligned with reference: {ref_npz_path.name}"

        src_keys = _window_key_pairs(src_t0, src_t1)
        ref_keys = _window_key_pairs(ref_t0, ref_t1)
        src_pos = {k: i for i, k in enumerate(src_keys)}
        idx = []
        for k in ref_keys:
            if k not in src_pos:
                return False, f"reference window not found in target: {k}"
            idx.append(src_pos[k])
        idx_arr = np.asarray(idx, dtype=np.int64)

        n_src = int(src_t0.shape[0])
        out_data: dict[str, np.ndarray] = {}
        for k in src.files:
            arr = np.asarray(src[k])
            if arr.ndim > 0 and arr.shape[0] == n_src:
                out_data[k] = arr[idx_arr]
            else:
                out_data[k] = arr

        out_data["hr_gt"] = np.asarray(ref["hr_gt"])
        if "n_peaks" in ref:
            out_data["n_peaks"] = np.asarray(ref["n_peaks"])

        np.savez_compressed(npz_path, **out_data)
        return True, f"aligned to {ref_npz_path.name}: {len(idx_arr)} windows"
    finally:
        src.close()
        ref.close()


def write_pipeline_run_doc(
    out_path: Path,
    *,
    run_label: str,
    npz_rel: str,
    eff_sec: float,
    n_windows: int,
    hr_ratio_line: str | None,
) -> None:
    h = int(eff_sec // 3600)
    m = int((eff_sec % 3600) // 60)
    sec = int(eff_sec % 60)
    lines = [
        "Pipeline run summary",
        f"Written at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Dataset",
        f"  run_label: {run_label}",
        f"  DEVICE (config): {DEVICE}",
        f"  PPG_CHANNEL (config): {PPG_CHANNEL}",
        f"  alignment windows NPZ basename: {ALIGNMENT_WINDOWS_BASE}",
        f"  NPZ path (relative to package root): {npz_rel}",
        "",
        "Statistics",
        f"  Deduplicated effective duration (union of [t0_ms, t1_ms] windows): {eff_sec:.1f} s ({h:02d}:{m:02d}:{sec:02d})",
        f"  Windows: {n_windows}, ecg_fs: {ECG_FS} Hz",
    ]
    if hr_ratio_line is not None:
        lines.append(f"  Valid hr_gt: {hr_ratio_line}")
    else:
        lines.append(
            "  Valid hr_gt: N/A (no hr_gt key in NPZ; e.g. Pan-Tompkins skipped with --skip-pan-tompkins)"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_single_pipeline_inner(
    *,
    skip_windowing: bool,
    skip_pan_tompkins: bool,
    run_label: str | None,
    child_env: dict[str, str] | None,
) -> None:
    """Single device × single channel: windowing + Pan-Tompkins (same behavior as before)."""
    data_dir = OUTPUTS_DIR

    if not skip_windowing:
        if not run(
            [sys.executable, str(_PACKAGE_ROOT / "run_windowing_aligned.py")],
            "1. Windowing (alignment + segments)",
            env=child_env,
        ):
            sys.exit(1)

    if not skip_pan_tompkins:
        if not run(
            [sys.executable, str(_PACKAGE_ROOT / "run_pan_tompkins.py")],
            "2. Pan-Tompkins (hr_gt)",
            env=child_env,
        ):
            sys.exit(1)
    else:
        if PPG_CHANNEL == "ppg_green":
            out_npz = data_dir / f"{ALIGNMENT_WINDOWS_BASE}.npz"
            ref_base = ALIGNMENT_WINDOWS_BASE.replace("_ppg_green", "_ppg_ir")
            ref_npz = data_dir / f"{ref_base}.npz"
            changed, msg = _align_npz_to_reference_windows(out_npz, ref_npz)
            tag = "ALIGNED" if changed else "ALIGN-SKIP"
            print(f"  [{tag}] {msg}")

    print("\n" + "=" * 60)
    print("  Pipeline finished")
    print("=" * 60)
    out_npz = data_dir / f"{ALIGNMENT_WINDOWS_BASE}.npz"
    print(f"  Output: {out_npz}")
    eff_sec = effective_window_coverage_sec(out_npz)
    h = int(eff_sec // 3600)
    m = int((eff_sec % 3600) // 60)
    s = int(eff_sec % 60)
    print(f"  Deduplicated effective duration: {eff_sec:.1f} s ({h:02d}:{m:02d}:{s:02d})")

    n_win, _valid_n, hr_ratio = npz_pipeline_stats(out_npz)
    print(f"  Windows: {n_win}, ecg_fs: {ECG_FS} Hz")
    if hr_ratio is not None:
        print(f"  Valid hr_gt: {hr_ratio}")
    else:
        print("  Valid hr_gt: (no hr_gt)")

    rl = run_label if run_label else default_run_label()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = data_dir / f"pipeline_run_{rl}_{ts}.txt"
    write_pipeline_run_doc(
        summary_path,
        run_label=rl,
        npz_rel=str(out_npz.relative_to(_PACKAGE_ROOT)),
        eff_sec=eff_sec,
        n_windows=n_win,
        hr_ratio_line=hr_ratio,
    )
    print(f"  Run summary written to: {summary_path.relative_to(_PACKAGE_ROOT)}")


def run_dataloader_batch() -> None:
    """
    Per ``config``: for each participant × wearable, read inputs/{P}/{P}_{Role}_raw.npz and {P}_ecg_raw.npz,
    write alignment_windows_* under outputs/{P}/ (windowing includes accel), Pan-Tompkins, merged dual-channel npz
    (compressed; IR/GREEN/accel).
    """
    channels: tuple[str, ...] = ("ppg_ir", "ppg_green")
    py = sys.executable
    pipeline_py = _PACKAGE_ROOT / "run_pipeline.py"
    merge_py = _PACKAGE_ROOT / "merge_alignment_windows_dual_channel.py"

    participants = [normalize_participant_id(x) for x in DATALOADER_PIPELINE_PARTICIPANTS]
    roles = list(WEARABLE_DEVICE_ROLES)
    if not roles:
        print("[ERROR] WEARABLE_DEVICE_ROLES is empty: keep at least one wearable in config.py.")
        sys.exit(1)

    print(f"[batch] participants={participants}")
    print(f"[batch] devices(roles)={roles}")

    for pid in participants:
        ecg_npz = dataloader_ecg_raw_npz(pid)
        out_dir = (DATALOADER_OUTPUT_ROOT / pid).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if not ecg_npz.is_file():
            print(f"[SKIP] {pid}: missing ECG file {ecg_npz}")
            continue

        for role in roles:
            ppg_npz = dataloader_ppg_raw_npz(pid, role)
            if not ppg_npz.is_file():
                print(f"[SKIP] {pid}/{role}: missing PPG file {ppg_npz}")
                continue

            device_id = participant_output_device(pid, role)
            print(f"\n{'#'*60}\n# {pid} / {role} -> RUN_DEVICE={device_id}\n{'#'*60}")

            for i, ch in enumerate(channels):
                skip_pt = ch != channels[0]
                extra: list[str] = []
                if skip_pt:
                    extra.append("--skip-pan-tompkins")

                env = {
                    **os.environ,
                    "_DATALOADER_CHILD": "1",
                    "RUN_SUBJECT": pid,
                    "RUN_DEVICE": device_id,
                    "RUN_PPG_CHANNEL": ch,
                    "RUN_OUTPUTS_DIR": str(out_dir),
                    "RUN_PPG_NPZ_PATH": str(ppg_npz.resolve()),
                    "RUN_ECG_PATH": str(ecg_npz.resolve()),
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                }
                cmd = [py, str(pipeline_py), *extra]
                print(f"\n[batch step] {device_id} / {ch}")
                r = subprocess.run(cmd, cwd=_PACKAGE_ROOT, env=env)
                if r.returncode != 0:
                    print(f"[FAIL] {device_id} {ch} exit {r.returncode}")
                    sys.exit(r.returncode)

            merged_windows = out_dir / f"alignment_windows_{device_id}.npz"
            if not run(
                [py, str(merge_py), "--data-dir", str(out_dir), "--devices", device_id],
                f"Merge IR+GREEN -> {merged_windows.name}",
            ):
                sys.exit(1)

            print(f"[OK] {device_id} final output: {merged_windows}")

            for ch_suffix in ("ppg_ir", "ppg_green"):
                intermediate = out_dir / f"alignment_windows_{device_id}_{ch_suffix}.npz"
                if intermediate.is_file():
                    intermediate.unlink()
                    print(f"[REMOVED] intermediate {intermediate.name}")

    print("\n[batch] All done.")


def main() -> None:
    if os.environ.get("_DATALOADER_CHILD") == "1":
        parser = argparse.ArgumentParser(
            description="Internal: single-device pipeline step (spawned by batch; uses RUN_* env from parent)",
        )
        parser.add_argument("--skip-windowing", action="store_true", help="Skip windowing")
        parser.add_argument("--skip-pan-tompkins", action="store_true", help="Skip Pan-Tompkins")
        parser.add_argument(
            "--run-label",
            default=None,
            metavar="NAME",
            help="Run label written into summary (default: derived from DEVICE+PPG_CHANNEL, e.g. earring2_ir)",
        )
        args = parser.parse_args()
        _run_single_pipeline_inner(
            skip_windowing=args.skip_windowing,
            skip_pan_tompkins=args.skip_pan_tompkins,
            run_label=args.run_label,
            child_env=None,
        )
        return

    batch_parser = argparse.ArgumentParser(
        description="Batch PPG-ECG windowing + Pan-Tompkins + merge (paths from config.py). "
        "Run from src/prepare_windowed_dataset: python run_pipeline.py",
    )
    batch_parser.parse_args()
    run_dataloader_batch()


if __name__ == "__main__":
    main()
