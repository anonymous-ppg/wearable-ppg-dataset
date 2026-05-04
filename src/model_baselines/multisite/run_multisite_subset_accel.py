# encoding=utf-8
"""
run_multisite_subset_accel.py
-----------------------------
LOSO sweep for a subset of devices with accel modality.

Trains on a pair (e.g., ring + watch with green + accel_z) and evaluates
on the held-out participant using only those devices.

Usage:
    python run_multisite_subset_accel.py --backbone resnet --devices 0,1
        # 0=earring, 1=ring → trains 4-channel (green+accel each)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader

from multisite.main_supervised_baseline_accel import set_seed, build_model, train, test
from data_preprocess.data_preprocess_multisite_accel import (
    discover_multisite_participants,
    MultisiteAccelDataset,
    DEVICE_NAMES,
    DEVICE_KEYS,
    DEVICE_TO_CHANNELS,
)
from data_preprocess.data_preprocess_utils import train_val_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--devices', type=str, required=True,
                   help='Comma-separated device indices, e.g. "0,1" for earring+ring')
    p.add_argument('--backbone', type=str, default='resnet',
                   choices=['FCN', 'DCL', 'cnn_lstm', 'LSTM', 'Transformer', 'resnet'])
    p.add_argument('--data_dir', type=str, default='../../../anonymous-ppg-dataset/multisite-ppg-submission/ppg_windowed_data')
    p.add_argument('--cuda', type=int, default=0)
    p.add_argument('--n_epoch', type=int, default=20)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--split_ratio', type=float, default=0.2)
    p.add_argument('--logdir', type=str, default='log/')

    args = p.parse_args()
    args.device_indices = [int(x) for x in args.devices.split(',')]
    args.device_tag = '_'.join(DEVICE_NAMES[i] for i in args.device_indices)
    args.n_feature = 2 * len(args.device_indices)
    args.len_sw = 200
    args.dataset = 'multisite_accel'
    args.position = None
    args.target_domain = None
    return args


def load_subset_data(participant_id, data_dir, device_indices):
    """Load aligned_8channel_accel.npz, returning only channels for selected devices."""
    import scipy.signal
    BPM_MIN, BPM_MAX, TARGET_LEN = 30, 210, 200

    path = os.path.join(data_dir, participant_id, "aligned_8channel_accel.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"aligned_8channel_accel.npz not found at {path}.")

    chan_indices = []
    for d in device_indices:
        chan_indices.extend(DEVICE_TO_CHANNELS[d])
    keys = [DEVICE_KEYS[i] for i in chan_indices]

    print(f"  Loading {participant_id} | devices: {[DEVICE_NAMES[d] for d in device_indices]} ({len(keys)} channels)")

    data = np.load(path, allow_pickle=True)
    hr_gt = data['hr_gt'].astype(np.float32)

    channels = []
    for key in keys:
        ppg = data[key].astype(np.float32)
        ppg = scipy.signal.resample(ppg, TARGET_LEN, axis=1).astype(np.float32)
        channels.append(ppg)

    n = min(len(hr_gt), *[len(c) for c in channels])
    hr_gt    = hr_gt[:n]
    channels = [c[:n] for c in channels]

    valid = (
        np.isfinite(hr_gt) & (hr_gt >= BPM_MIN) & (hr_gt <= BPM_MAX) &
        np.stack([~np.isnan(c).any(axis=1) for c in channels], axis=0).all(axis=0)
    )
    hr_gt    = hr_gt[valid]
    channels = [c[valid] for c in channels]

    normed = [(c - c.mean(axis=1, keepdims=True)) / (c.std(axis=1, keepdims=True) + 1e-6)
              for c in channels]

    X = np.stack(normed, axis=2).astype(np.float32)
    y = hr_gt.astype(np.float32)
    print(f"    → {len(y)} valid windows, X shape {X.shape}")
    return X, y


def build_loaders(args, target_pid, all_participants):
    source_ids = [pid for pid in all_participants if pid != target_pid]
    X_list, y_list = [], []
    for pid in source_ids:
        X, y = load_subset_data(pid, args.data_dir, args.device_indices)
        X_list.append(X)
        y_list.append(y)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    d_all = np.zeros(len(y_all), dtype=np.int64)

    X_train, X_val, y_train, y_val, _, _ = train_val_split(
        X_all, y_all, d_all, split_ratio=args.split_ratio
    )

    X_test, y_test = load_subset_data(target_pid, args.data_dir, args.device_indices)

    train_set = MultisiteAccelDataset(X_train, y_train, domain_idx=0)
    val_set   = MultisiteAccelDataset(X_val,   y_val,   domain_idx=0)
    test_set  = MultisiteAccelDataset(X_test,  y_test,  domain_idx=0)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size,
                              shuffle=False, drop_last=False)

    print(f"  Split — Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")
    return [train_loader], val_loader, test_loader


def train_one(args, target_pid, all_participants, seed_idx):
    set_seed(seed_idx * 10 + np.random.randint(0, 10))
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    train_loaders, val_loader, test_loader = build_loaders(args, target_pid, all_participants)

    model = build_model(args).to(device)
    args.target_domain = target_pid
    args.model_name = (
        f"{args.backbone}_multisite_accel_subset-{args.device_tag}"
        f"_target{target_pid}_seed{seed_idx}"
        f"_lr{args.lr}_bs{args.batch_size}"
    )

    os.makedirs(args.logdir, exist_ok=True)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state = train(args, train_loaders, val_loader, model, device, optimizer, criterion)
    model_test = build_model(args).to(device)
    model_test.load_state_dict(best_state)

    return test(test_loader, model_test, device, criterion)


def main():
    args = parse_args()
    set_seed(40)

    participants = discover_multisite_participants(args.data_dir)
    print(f"Found {len(participants)} participants")
    print(f"Device subset: {args.device_tag} (indices {args.device_indices})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results_summary/results_ACCEL_subset-{args.device_tag}_{args.backbone}_{ts}.txt"
    os.makedirs("results_summary", exist_ok=True)

    header_lines = [
        "=" * 70,
        f"Multisite ACCEL Subset Results — {args.device_tag}",
        "=" * 70,
        f"Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Backbone     : {args.backbone}",
        f"Data dir     : {args.data_dir}",
        f"Device subset: {args.device_tag} (indices {args.device_indices}, n_channels={args.n_feature})",
        f"Participants : {len(participants)}",
        "",
        f"{'Participant':<12}  {'MAE':>8}  {'RMSE':>8}  {'R':>8}",
        "-" * 70,
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(header_lines) + '\n')
    print(f"Writing → {out_path}")

    fold_results = []

    for pid in participants:
        print(f"\n  Target: {pid}")
        try:
            mae, rmse, r = train_one(args, pid, participants, seed_idx=0)
        except Exception as e:
            print(f"    ERROR: {e}")
            with open(out_path, 'a') as f:
                f.write(f"{pid:<12}  ERROR: {e}\n")
            continue

        fold_results.append([mae, rmse, r])
        row = f"{pid:<12}  {mae:>8.3f}  {rmse:>8.3f}  {r:>8.4f}"
        print(row)
        with open(out_path, 'a') as f:
            f.write(row + '\n')

    if fold_results:
        arr = np.array(fold_results)
        summary = [
            "-" * 70,
            f"{'MEAN':<12}  {arr[:,0].mean():>8.3f}  {arr[:,1].mean():>8.3f}  {arr[:,2].mean():>8.4f}",
            f"{'STD':<12}  {arr[:,0].std():>8.3f}  {arr[:,1].std():>8.3f}  {arr[:,2].std():>8.4f}",
            "=" * 70,
        ]
        with open(out_path, 'a') as f:
            f.write('\n'.join(summary) + '\n')
        print('\n' + '\n'.join(summary))

    print(f"\nDone → {out_path}")


if __name__ == '__main__':
    main()
