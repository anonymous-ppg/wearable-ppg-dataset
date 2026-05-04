"""
run_preprocess_ppg.py
---------------------
Applies detrend + bandpass (0.7-3.5 Hz) to each alignment_windows_*_ppg_green.npz
and writes the result to alignment_windows_*_ppg_green_preprocess.npz (original file
is not overwritten).

0.7-3.5 Hz corresponds to ~42-210 bpm.

Usage:
    PYTHONPATH=. python data_preprocess/run_preprocess_ppg.py \
        --data_dir ../../../anonymous-ppg-dataset/multisite-ppg-submission/ppg_windowed_data
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt, detrend

BAND_LOW_HZ  = 0.7
BAND_HIGH_HZ = 3.5
DEFAULT_FS   = 25.0   # your dataset sampling rate


def bandpass_filter(x: np.ndarray, low_hz: float, high_hz: float,
                    fs: float, order: int = 2) -> np.ndarray:
    nyq  = fs / 2
    low  = max(0.01, low_hz / nyq)
    high = min(0.99, high_hz / nyq)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x.astype(np.float64))


def preprocess_ppg(ppg: np.ndarray, ppg_fs: float) -> np.ndarray:
    """detrend (linear) + bandpass 0.7-3.5 Hz"""
    out = np.zeros_like(ppg, dtype=np.float64)
    for i in range(ppg.shape[0]):
        x      = detrend(ppg[i].astype(np.float64), type="linear")
        out[i] = bandpass_filter(x, BAND_LOW_HZ, BAND_HIGH_HZ, ppg_fs)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="PPG detrend + bandpass on alignment_windows npz"
    )
    parser.add_argument('--data_dir', default='../../../anonymous-ppg-dataset/multisite-ppg-submission/ppg_windowed_data', type=str,
                        help='Root directory containing participant subfolders (P1/, P2/, ...)')
    parser.add_argument('--fs', default=DEFAULT_FS, type=float,
                        help='Sampling rate in Hz (default: 25.0)')
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    npz_files = sorted(data_dir.rglob('*_ppg_green.npz'))

    # exclude already preprocessed files
    npz_files = [p for p in npz_files if '_preprocess' not in p.name]

    if not npz_files:
        print(f'No *_ppg_green.npz files found in {data_dir}')
        sys.exit(1)

    print(f'Found {len(npz_files)} files to preprocess...\n')

    for src_path in npz_files:
        out_path = src_path.with_name(src_path.stem + '_preprocess.npz')

        if out_path.exists():
            print(f'  [SKIP] already exists: {out_path.name}')
            continue

        print(f'  Processing {src_path.name} ...')
        data   = dict(np.load(src_path, allow_pickle=True))
        ppg    = data['ppg']
        ppg_fs = float(data['ppg_fs']) if 'ppg_fs' in data else args.fs

        print(f'    detrend + bandpass {BAND_LOW_HZ}-{BAND_HIGH_HZ} Hz ...')
        data['ppg']              = preprocess_ppg(ppg, ppg_fs)
        data['ppg_preprocessed'] = np.array(True)

        np.savez(out_path, **data)
        print(f'    → saved {out_path.name}')

    print('\nDone.')


if __name__ == '__main__':
    main()