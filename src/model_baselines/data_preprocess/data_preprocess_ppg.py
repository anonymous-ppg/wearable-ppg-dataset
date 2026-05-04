# encoding=utf-8
"""
data_preprocess_ppg.py
--------------------------
Loads PPG data for a single participant at a single body position
(one wearable site at a time).

Input files (inside each participant's folder):
    alignment_windows_{PID}_{Device}.npz
    keys: ppg_green (W, L), ppg_ir (W, L), hr_gt (W,), ppg_fs, ...

    W = number of windows, L = original window length (e.g. 1000 samples at 100 Hz)

Processing:
    - Resample L → 200 samples (25 Hz × 8 s)
    - Per-window z-score normalisation for the green channel
    - X: (W, 200, 1)  — green channel only
    - Filter out windows with NaN or HR outside [BPM_MIN, BPM_MAX]
    - y: raw HR in bpm (float32) — no integer offset, since this is regression

Model input shape (after DataLoader):
    sample: (B, 200, 1)   — models permute internally to (B, 1, 200)
    target: (B,)          — bpm float
"""

import os
import numpy as np
import scipy.signal
import torch
from torch.utils.data import DataLoader

from data_preprocess.participants_config import find_device_files, discover_participants
from data_preprocess.data_preprocess_utils import train_val_split
from data_preprocess.base_loader import base_loader


BPM_MIN    = 30
BPM_MAX    = 210
TARGET_LEN = 200    # 25 Hz × 8 s


# ── Load one participant ──────────────────────────────────────────────────────

def load_domain_data(participant_id: str, position: str, data_dir: str, args=None):
    """
    Load and preprocess PPG windows for one participant at one body position.

    Args:
        participant_id : participant folder name, e.g. "P1"
        position       : body position, e.g. "ring"
        data_dir       : root directory containing participant subfolders

    Returns:
        X : np.ndarray, shape (N, 200, 1), float32   — green channel
        y : np.ndarray, shape (N,),        float32   — heart rate in bpm
    """
    folder = os.path.join(data_dir, participant_id)
    result = find_device_files(folder, position)

    if result is None:
        raise FileNotFoundError(
            f"No {position} files found for participant '{participant_id}' "
            f"in {folder}"
        )

    if getattr(args, 'use_preprocess', False):
        p = result.replace('.npz', '_preprocess.npz')
        green_path = p if os.path.exists(p) else result
    else:
        green_path = result
    print(f"  Loading {participant_id} | {position}: {os.path.basename(green_path)}")

    green_data = np.load(green_path, allow_pickle=True)

    ppg_green = green_data['ppg_green'].astype(np.float32)   # (W, L)
    hr_gt     = green_data['hr_gt'].astype(np.float32)       # (W,)

    # align lengths across arrays
    n = min(len(ppg_green), len(hr_gt))
    ppg_green = ppg_green[:n]
    hr_gt     = hr_gt[:n]

    # remove windows with invalid HR or NaN signal values
    valid = (
        np.isfinite(hr_gt) &
        (hr_gt >= BPM_MIN) &
        (hr_gt <= BPM_MAX) &
        (~np.isnan(ppg_green).any(axis=1))
    )
    ppg_green = ppg_green[valid]
    hr_gt     = hr_gt[valid]

    # resample to TARGET_LEN
    ppg_green = scipy.signal.resample(ppg_green, TARGET_LEN, axis=1).astype(np.float32)

    # per-window z-score normalisation
    ppg_green = (ppg_green - ppg_green.mean(axis=1, keepdims=True)) / \
                (ppg_green.std(axis=1,  keepdims=True) + 1e-6)

    X = ppg_green[:, :, np.newaxis].astype(np.float32)   # (N, 200, 1)

    # regression target: raw bpm (float), no integer offset
    y = hr_gt.astype(np.float32)

    print(f"    → {len(y)} valid windows, HR range {hr_gt.min():.0f}–{hr_gt.max():.0f} bpm")
    return X, y


# ── Dataset ───────────────────────────────────────────────────────────────────

class PPGDataset(base_loader):
    """
    PyTorch Dataset for PPG regression (single device).

    __getitem__ returns:
        sample : torch.float32, shape (200, 1)
        target : torch.float32, scalar bpm value
        domain : int (participant index, for reference only)
    """
    def __init__(self, X, y, domain_idx):
        # base_loader expects (samples, labels, domains)
        domains = np.full(len(y), domain_idx, dtype=np.int64)
        super().__init__(X, y, domains)

    def __getitem__(self, index):
        sample = torch.tensor(self.samples[index], dtype=torch.float32)  # (200, 1)
        target = torch.tensor(self.labels[index],  dtype=torch.float32)  # scalar
        domain = self.domains[index]
        return sample, target, domain


# ── Train / val / test split ──────────────────────────────────────────────────

def prep_domains_subject_sp(args):
    """
    Leave-one-subject-out split.

    - args.target_domain : participant ID string to hold out as test, e.g. "P3"
    - All other participants are pooled for training, then split 80/20 into train/val.
    - The held-out participant is used entirely as the test set.
    """
    position = args.position
    all_ids  = discover_participants(args.data_dir, position)

    if not all_ids:
        raise FileNotFoundError(
            f"No participants found in '{args.data_dir}' for position '{position}'."
        )

    if args.target_domain not in all_ids:
        raise ValueError(
            f"target_domain '{args.target_domain}' not found. "
            f"Available participants: {all_ids}"
        )

    source_ids = [pid for pid in all_ids if pid != args.target_domain]

    # ── pool all source participants
    X_list, y_list = [], []
    for i, pid in enumerate(source_ids):
        X, y = load_domain_data(pid, position, args.data_dir, args)
        X_list.append(X)
        y_list.append(y)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    # domain array (not used for training, kept for compatibility with base_loader)
    d_all = np.zeros(len(y_all), dtype=np.int64)

    # ── train / val split (80 / 20)
    X_train, X_val, y_train, y_val, d_train, d_val = train_val_split(
        X_all, y_all, d_all, split_ratio=args.split_ratio
    )

    # ── test: held-out participant
    X_test, y_test = load_domain_data(args.target_domain, position, args.data_dir, args)
    d_test = np.zeros(len(y_test), dtype=np.int64)

    # ── DataLoaders
    train_set = PPGDataset(X_train, y_train, domain_idx=0)
    val_set   = PPGDataset(X_val,   y_val,   domain_idx=0)
    test_set  = PPGDataset(X_test,  y_test,  domain_idx=0)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size,
                              shuffle=False, drop_last=False)

    print(f"  Split — Train: {len(train_set)}  Val: {len(val_set)}  "
          f"Test: {len(test_set)} windows")

    return [train_loader], val_loader, test_loader


# ── Entry point ───────────────────────────────────────────────────────────────

def prep_ppg(args):
    if args.cases == 'subject_val':
        return prep_domains_subject_sp(args)
    else:
        raise ValueError(f"Unknown args.cases: '{args.cases}'. Expected 'subject_val'.")
