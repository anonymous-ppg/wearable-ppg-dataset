"""
Batch: optional preprocess + selected heuristics on merged alignment_windows npz.

Run from ``src/heuristic_baselines`` (this directory; ``inputs/<Px>/`` and ``outputs/<Px>/``)::

    python runner.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
import config  # noqa: E402  # pylint: disable=wrong-import-position

from algorithms import resolve_algorithm
from io_utils import merged_windows_npz, normalize_participant_id, participant_device_id, ppg_from_npz
from preprocess import preprocess_ppg, write_preprocess_npz


def _metrics_lines(hr_est: np.ndarray, hr_gt: np.ndarray, label: str) -> list[str]:
    """Same text as printed per algorithm run; also written to *_runlog.txt."""
    lines: list[str] = []
    valid = int(np.sum(~np.isnan(hr_est)))
    n = len(hr_est)
    lines.append(f"  [{label}] valid windows: {valid}/{n} ({100.0 * valid / n:.1f}%)")
    if valid == 0 or hr_gt.size == 0:
        return lines
    mask = ~np.isnan(hr_est) & ~np.isnan(hr_gt)
    if int(np.sum(mask)) == 0:
        lines.append("  [metrics] no overlapping valid hr_gt / estimate")
        return lines
    err = hr_gt[mask] - hr_est[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    rho = float(np.corrcoef(hr_gt[mask], hr_est[mask])[0, 1])
    lines.append(f"  [metrics] MAE={mae:.2f} RMSE={rmse:.2f} rho={rho:.4f} (n={int(np.sum(mask))})")
    return lines


def run_one_device_channel(
    *,
    raw_npz: Path,
    result_dir: Path,
    device_id: str,
    ppg_channel: str,
    algorithms: tuple[str, ...],
    run_preprocess: bool,
) -> None:
    pre_npz = raw_npz.with_name(raw_npz.stem + "_preprocess.npz")
    if run_preprocess:
        if not pre_npz.is_file():
            print(f"  [preprocess] writing {pre_npz.name}")
            write_preprocess_npz(raw_npz, pre_npz)
        load_path = pre_npz
    else:
        load_path = raw_npz

    with np.load(load_path, allow_pickle=True) as z:
        data = {k: np.asarray(z[k]) for k in z.files}

    t0_ms = np.asarray(data["t0_ms"], dtype=np.float64)
    hr_gt = np.asarray(data["hr_gt"], dtype=np.float64)
    ppg_fs = float(np.asarray(data["ppg_fs"]).item()) if "ppg_fs" in data else 100.0

    ppg = np.asarray(ppg_from_npz(data, ppg_channel), dtype=np.float64)

    n = int(ppg.shape[0])
    result_dir.mkdir(parents=True, exist_ok=True)

    for alg_name in algorithms:
        try:
            hr_fn, hr_col = resolve_algorithm(alg_name)
        except KeyError:
            print(f"  [SKIP] unknown algorithm: {alg_name}")
            continue
        hrs: list[float] = []
        for i in range(n):
            hrs.append(float(hr_fn(ppg[i], ppg_fs)))
        hr_arr = np.array(hrs, dtype=np.float64)
        csv_path = result_dir / f"model_results_{device_id}_{ppg_channel}_{alg_name}.csv"
        df = pd.DataFrame({"t0_ms": t0_ms, "hr_gt": hr_gt, hr_col: hr_arr})
        df.to_csv(csv_path, index=False)
        log_lines = [f"  [SAVED] {csv_path.name}", *_metrics_lines(hr_arr, hr_gt, alg_name)]
        for line in log_lines:
            print(line)
        log_path = result_dir / f"model_results_{device_id}_{ppg_channel}_{alg_name}_runlog.txt"
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"  [SAVED] {log_path.name}")


def main() -> None:
    root = config.HEURISTIC_WINDOWS_ROOT
    participants = [normalize_participant_id(p) for p in config.HEURISTIC_PIPELINE_PARTICIPANTS]
    roles = list(config.HEURISTIC_DEVICE_ROLES)
    _ch_cfg = config.HEURISTIC_PPG_CHANNELS
    channels = [_ch_cfg] if isinstance(_ch_cfg, str) else list(_ch_cfg)
    want_algs = [a.strip().lower() for a in config.HEURISTIC_ALGORITHMS]
    bad = [a for a in want_algs if a not in config.KNOWN_HEURISTIC_ALGORITHMS]
    if bad:
        print(f"[ERROR] Unknown algorithm(s): {bad}. Known: {sorted(config.KNOWN_HEURISTIC_ALGORITHMS)}")
        sys.exit(1)
    if not want_algs:
        print("[ERROR] HEURISTIC_ALGORITHMS is empty")
        sys.exit(1)

    result_root = config.HEURISTIC_RESULT_ROOT
    print(f"[heuristic] windows_npz_root={root}")
    print(f"[heuristic] result_root={result_root}")
    print(f"[heuristic] participants={participants} roles={roles} channels={channels}")
    print(f"[heuristic] preprocess={config.HEURISTIC_RUN_PREPROCESS} algorithms={want_algs}")

    for pid in participants:
        result_dir = (result_root / pid).resolve()
        result_dir.mkdir(parents=True, exist_ok=True)
        for role in roles:
            raw = merged_windows_npz(root, pid, role)
            dev_id = participant_device_id(pid, role)
            if not raw.is_file():
                print(f"\n[SKIP] missing {raw}")
                continue
            print(f"\n{'='*60}\n  {dev_id}\n  npz={raw.name}\n  out -> {result_dir}\n{'='*60}")
            for ch in channels:
                if ch not in ("ppg_ir", "ppg_green"):
                    print(f"  [SKIP] unknown channel {ch}")
                    continue
                print(f"\n  --- channel {ch} ---")
                run_one_device_channel(
                    raw_npz=raw,
                    result_dir=result_dir,
                    device_id=dev_id,
                    ppg_channel=ch,
                    algorithms=tuple(want_algs),
                    run_preprocess=config.HEURISTIC_RUN_PREPROCESS,
                )

    print("\n[heuristic] Done.")


if __name__ == "__main__":
    main()
