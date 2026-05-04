# encoding=utf-8
"""
align_4device.py
--------
Generate per-participant aligned multi-device PPG files.

Aligns time-stamped PPG windows from all 4 devices (Earring, Ring, Watch, Necklace)
using a tolerance-based timestamp match. Supports three modalities:

  --modality green   → green PPG only (1 channel per device, 4 channels total)
                        Output: aligned_4device.npz
  --modality accel   → green + accel_z (2 channels per device, 8 channels total)
                        Output: aligned_8channel_accel.npz
  --modality ir      → green + IR PPG (2 channels per device, 8 channels total)
                        Output: aligned_8channel.npz

Run once before multisite experiments.

Input (single shared file pattern for all modalities):
    alignment_windows_<PID>_<Device>.npz
    Available keys: ppg_green, accel_z, ppg_ir, hr_gt, t0_ms
    The script picks which keys to extract per --modality.
"""

import numpy as np
import os
import glob
import argparse

from data_preprocess.participants_config import POSITION_TO_DEVICE


DEVICES      = list(POSITION_TO_DEVICE.values())
TOLERANCE_MS = 500


# ── Modality configuration ───────────────────────────────────────────────────
# Each modality specifies which keys to extract from the shared input file
# and where to save the output.

MODALITY_CONFIG = {
    'green': {
        'output_name':  'aligned_4device.npz',
        # (npz_key_in_input, output_suffix). e.g. earring → ppg_earring (green only)
        'channels': [('ppg_green', '')],
    },
    'accel': {
        'output_name':  'aligned_8channel_accel.npz',
        'channels': [('ppg_green', '_green'), ('accel_z', '_accel')],
    },
    'ir': {
        'output_name':  'aligned_8channel.npz',
        'channels': [('ppg_green', '_green'), ('ppg_ir', '_ir')],
    },
}


# ── Participant discovery ────────────────────────────────────────────────────

def get_participants(data_dir):
    """Find participants with files for all 4 devices."""
    participants = {}
    for name in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder) or name.startswith('.'):
            continue
        devices = {}
        for device_label in DEVICES:
            matches = glob.glob(os.path.join(
                folder, f"alignment_windows_*_{device_label}.npz"
            ))
            if matches:
                devices[device_label] = matches[0]
        if len(devices) == 4:
            participants[name] = devices
        else:
            print(f"  Skipping '{name}': only has {list(devices.keys())}")
    return participants


# ── Alignment for one participant ────────────────────────────────────────────

def align_participant(pid, device_files, data_dir, modality):
    print(f"\n── Participant {pid} ──")
    config = MODALITY_CONFIG[modality]

    # Load all 4 device .npz files
    device_data = {}
    for dev, path in device_files.items():
        device_data[dev] = np.load(path, allow_pickle=True)
        print(f"  {dev}: {len(device_data[dev]['t0_ms'])} windows")

    # Base device = the one with fewest windows
    base_dev = min(device_files.keys(),
                   key=lambda d: len(device_data[d]['t0_ms']))
    base_t0  = device_data[base_dev]['t0_ms']

    # For each base window, find closest match across all devices
    aligned_indices = {dev: [] for dev in DEVICES}
    for t in base_t0:
        indices = {}
        valid   = True
        for dev in DEVICES:
            t0   = device_data[dev]['t0_ms']
            diff = np.abs(t0 - t)
            idx  = np.argmin(diff)
            if diff[idx] <= TOLERANCE_MS:
                indices[dev] = idx
            else:
                valid = False
                break
        if valid:
            for dev in DEVICES:
                aligned_indices[dev].append(indices[dev])

    n_aligned = len(aligned_indices[base_dev])
    if n_aligned == 0:
        print("  No aligned windows.")
        return None

    # Pack output dict
    save_dict = {}
    for dev in DEVICES:
        idxs = np.array(aligned_indices[dev])
        for input_key, output_suffix in config['channels']:
            output_key = f'ppg_{dev.lower()}{output_suffix}'
            save_dict[output_key] = device_data[dev][input_key][idxs]

    base_idxs = np.array(aligned_indices[base_dev])
    save_dict['hr_gt'] = device_data[base_dev]['hr_gt'][base_idxs]
    save_dict['t0_ms'] = device_data[base_dev]['t0_ms'][base_idxs]

    save_path = os.path.join(data_dir, pid, config['output_name'])
    np.savez(save_path, **save_dict)
    print(f"  Saved → {save_path} ({n_aligned} windows)")
    return n_aligned


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory with per-participant folders.')
    parser.add_argument('--modality', type=str, default='green',
                        choices=['green', 'accel', 'ir'],
                        help='Which modality to align: '
                             'green=1ch/device, accel=2ch (green+accel_z), '
                             'ir=2ch (green+IR).')
    parser.add_argument('--participants', type=str, default=None,
                        help='Comma-separated list of participant IDs to align '
                             '(e.g. "P1,P2"). Default: all participants.')
    args = parser.parse_args()

    config = MODALITY_CONFIG[args.modality]
    print(f"Modality: {args.modality}")
    print(f"  Input pattern : alignment_windows_*_<Device>.npz")
    print(f"  Output file   : {config['output_name']}")
    print(f"  Channels      : {[k for k, _ in config['channels']]}")

    participants = get_participants(args.data_dir)
    if args.participants:
        keep = set(args.participants.split(','))
        participants = {pid: v for pid, v in participants.items() if pid in keep}
    print(f"\nFound {len(participants)} participants with all 4 devices.")

    total = 0
    for pid, device_files in sorted(participants.items()):
        n = align_participant(pid, device_files, args.data_dir, args.modality)
        if n:
            total += n
    print(f"\nDone. Total aligned windows: {total}")
