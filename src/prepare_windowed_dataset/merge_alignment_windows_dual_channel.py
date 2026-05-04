"""
Merge per-device dual-channel alignment window files into one NPZ.

Input (per device):
  alignment_windows_{device}_ppg_ir.npz
  alignment_windows_{device}_ppg_green.npz

Output:
  alignment_windows_{device}.npz

``config`` lives in this package directory (same folder as this script).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve()
_pkg_root = str(_p.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
from config import OUTPUTS_DIR


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.floating):
        return np.allclose(a, b, equal_nan=True)
    return np.array_equal(a, b)


def _window_key_pairs(t0_ms: np.ndarray, t1_ms: np.ndarray) -> list[tuple[int, int]]:
    t0 = np.asarray(t0_ms, dtype=np.float64).ravel()
    t1 = np.asarray(t1_ms, dtype=np.float64).ravel()
    if t0.shape != t1.shape:
        raise ValueError("t0_ms and t1_ms have different shapes")
    # Window boundaries are in ms; rounding guards tiny float noise.
    return list(zip(np.rint(t0).astype(np.int64), np.rint(t1).astype(np.int64)))


def _index_map_by_window(ref_t0: np.ndarray, ref_t1: np.ndarray, src_t0: np.ndarray, src_t1: np.ndarray) -> np.ndarray:
    ref_keys = _window_key_pairs(ref_t0, ref_t1)
    src_keys = _window_key_pairs(src_t0, src_t1)
    src_pos = {k: i for i, k in enumerate(src_keys)}
    idx: list[int] = []
    for k in ref_keys:
        if k not in src_pos:
            raise KeyError(f"missing window key {k}")
        idx.append(src_pos[k])
    return np.asarray(idx, dtype=np.int64)


def _maybe_take_rows(arr: np.ndarray, idx: np.ndarray, n_rows: int) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim > 0 and x.shape[0] == n_rows:
        return x[idx]
    return x


def merge_one_device(data_dir: Path, device: str) -> Path:
    ir_path = data_dir / f"alignment_windows_{device}_ppg_ir.npz"
    green_path = data_dir / f"alignment_windows_{device}_ppg_green.npz"
    out_path = data_dir / f"alignment_windows_{device}.npz"

    if not ir_path.is_file() or not green_path.is_file():
        raise FileNotFoundError(
            f"Missing channel files for {device}: {ir_path.name}, {green_path.name}"
        )

    with np.load(ir_path, allow_pickle=True) as ir, np.load(green_path, allow_pickle=True) as gr:
        n_ir = int(np.asarray(ir["t0_ms"]).shape[0]) if "t0_ms" in ir else 0
        n_gr = int(np.asarray(gr["t0_ms"]).shape[0]) if "t0_ms" in gr else 0
        ir_idx = np.arange(n_ir, dtype=np.int64)
        gr_idx = np.arange(n_gr, dtype=np.int64)

        # If only one channel ran Pan-Tompkins, that side may be "valid-only" windows.
        # Align the other side to the filtered window list via (t0_ms, t1_ms) keys.
        t0_same = "t0_ms" in ir and "t0_ms" in gr and _same(np.asarray(ir["t0_ms"]), np.asarray(gr["t0_ms"]))
        t1_same = "t1_ms" in ir and "t1_ms" in gr and _same(np.asarray(ir["t1_ms"]), np.asarray(gr["t1_ms"]))
        if not (t0_same and t1_same):
            ir_has_hr = "hr_gt" in ir
            gr_has_hr = "hr_gt" in gr
            try:
                if ir_has_hr and not gr_has_hr:
                    gr_idx = _index_map_by_window(ir["t0_ms"], ir["t1_ms"], gr["t0_ms"], gr["t1_ms"])
                    print(f"[INFO] {device}: align GREEN windows to IR valid-only set ({len(gr_idx)} windows)")
                elif gr_has_hr and not ir_has_hr:
                    ir_idx = _index_map_by_window(gr["t0_ms"], gr["t1_ms"], ir["t0_ms"], ir["t1_ms"])
                    print(f"[INFO] {device}: align IR windows to GREEN valid-only set ({len(ir_idx)} windows)")
                else:
                    raise ValueError(
                        f"{device}: IR/GREEN windows differ but cannot infer reference side "
                        "(both have hr_gt or both missing hr_gt)"
                    )
            except KeyError as ex:
                raise ValueError(f"{device}: unable to align windows between IR/GREEN ({ex})") from ex

        required_shared = ("t0_ms", "t1_ms", "ecg", "ecg_valid_len", "ppg_fs")
        for k in required_shared:
            if k not in ir or k not in gr:
                raise KeyError(f"{device}: missing shared key '{k}' in channel npz files")
            ir_k = _maybe_take_rows(ir[k], ir_idx, n_ir)
            gr_k = _maybe_take_rows(gr[k], gr_idx, n_gr)
            if not _same(ir_k, gr_k):
                raise ValueError(f"{device}: shared key mismatch for '{k}' between IR and GREEN")

        if "ppg_t_ms" in ir and "ppg_t_ms" in gr:
            ir_ppg_t = _maybe_take_rows(ir["ppg_t_ms"], ir_idx, n_ir)
            gr_ppg_t = _maybe_take_rows(gr["ppg_t_ms"], gr_idx, n_gr)
            if not _same(ir_ppg_t, gr_ppg_t):
                raise ValueError(f"{device}: ppg_t_ms mismatch between IR and GREEN")

        ir_t0 = _maybe_take_rows(ir["t0_ms"], ir_idx, n_ir)
        ir_t1 = _maybe_take_rows(ir["t1_ms"], ir_idx, n_ir)
        ir_ecg = _maybe_take_rows(ir["ecg"], ir_idx, n_ir)
        ir_ecg_valid_len = _maybe_take_rows(ir["ecg_valid_len"], ir_idx, n_ir)
        ir_ppg = _maybe_take_rows(ir["ppg"], ir_idx, n_ir)
        gr_ppg = _maybe_take_rows(gr["ppg"], gr_idx, n_gr)

        out_data: dict[str, np.ndarray] = {
            "t0_ms": ir_t0,
            "t1_ms": ir_t1,
            "ecg": ir_ecg,
            "ecg_valid_len": ir_ecg_valid_len,
            "ppg_fs": ir["ppg_fs"],
            "ppg_ir": ir_ppg,
            "ppg_green": gr_ppg,
        }

        if "ppg_t_ms" in ir:
            out_data["ppg_t_ms"] = _maybe_take_rows(ir["ppg_t_ms"], ir_idx, n_ir)

        for ak in ("accel_x", "accel_y", "accel_z"):
            if ak in ir:
                out_data[ak] = _maybe_take_rows(ir[ak], ir_idx, n_ir)

        # hr_gt / n_peaks come from ECG; IR and GREEN channel npz should agree when both ran Pan-Tompkins.
        # If GREEN skipped Pan-Tompkins, only IR may carry these keys — copy from IR (or GREEN when inverted).
        for k in ("hr_gt", "n_peaks"):
            if k in ir and k in gr:
                ir_k = _maybe_take_rows(ir[k], ir_idx, n_ir)
                gr_k = _maybe_take_rows(gr[k], gr_idx, n_gr)
                if not _same(ir_k, gr_k):
                    raise ValueError(f"{device}: key mismatch for '{k}' between IR and GREEN")
                out_data[k] = ir_k
            elif k in ir:
                out_data[k] = _maybe_take_rows(ir[k], ir_idx, n_ir)
            elif k in gr:
                out_data[k] = _maybe_take_rows(gr[k], gr_idx, n_gr)

        np.savez_compressed(out_path, **out_data)

    return out_path


def discover_devices(data_dir: Path) -> list[str]:
    devices: set[str] = set()
    for p in data_dir.glob("alignment_windows_*_ppg_ir.npz"):
        stem = p.stem
        prefix = "alignment_windows_"
        suffix = "_ppg_ir"
        if not (stem.startswith(prefix) and stem.endswith(suffix)):
            continue
        devices.add(stem[len(prefix) : -len(suffix)])
    return sorted(devices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge alignment_windows per device into dual-channel NPZ")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Directory containing alignment_windows files (default: {OUTPUTS_DIR})",
    )
    parser.add_argument(
        "--devices",
        nargs="*",
        default=[],
        help="Optional device list (default: auto-discover from *_ppg_ir.npz)",
    )
    args = parser.parse_args()

    data_dir = (args.data_dir or OUTPUTS_DIR).resolve()
    devices = args.devices or discover_devices(data_dir)
    if not devices:
        print(f"No devices found in {data_dir}")
        sys.exit(1)

    ok = 0
    for d in devices:
        out = merge_one_device(data_dir, d)
        ok += 1
        print(f"[SAVED] {out.name}")
    print(f"[OK] Merged dual-channel NPZ for {ok} device(s).")


if __name__ == "__main__":
    main()
