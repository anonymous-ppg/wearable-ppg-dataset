# encoding=utf-8
"""
run_multisite_subset.py
-----------------------
LOSO sweep for a subset of devices.

Trains on a subset (e.g. earring+ring) and evaluates on the held-out participant
using only those devices.

Usage:
    python run_multisite_subset.py --backbone DCL --devices 0,1
        # 0=earring, 1=ring
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader

from supervised.main_supervised_baseline import set_seed, build_model, train, test
from data_preprocess.data_preprocess_multisite import (
    discover_multisite_participants,
    load_domain_data_subset,
    MultisiteDataset,
)
from data_preprocess.data_preprocess_utils import train_val_split


DEVICE_NAMES = ['earring', 'ring', 'watch', 'necklace']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--devices', type=str, required=True,
                   help='Comma-separated device indices, e.g. "0,1" for earring+ring')
    p.add_argument('--backbone',   type=str, default='resnet',
                   choices=['FCN', 'DCL', 'cnn_lstm', 'LSTM', 'Transformer', 'resnet'])
    p.add_argument('--data_dir',   type=str, default='../../../anonymous-ppg-dataset/multisite-ppg-submission/ppg_windowed_data')
    p.add_argument('--cuda',       type=int, default=0)
    p.add_argument('--n_epoch',    type=int, default=15)
    p.add_argument('--patience',   type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr',         type=float, default=5e-4)
    p.add_argument('--split_ratio',type=float, default=0.2)
    p.add_argument('--logdir',     type=str, default='log/')

    args = p.parse_args()
    args.device_indices = [int(x) for x in args.devices.split(',')]
    args.device_tag = '_'.join(DEVICE_NAMES[i] for i in args.device_indices)
    args.n_feature = len(args.device_indices)
    args.len_sw    = 200
    args.dataset   = 'multisite'
    args.position  = None
    return args


def build_loaders(args, target_pid, all_participants):
    source_ids = [pid for pid in all_participants if pid != target_pid]

    X_list, y_list = [], []
    for pid in source_ids:
        X, y = load_domain_data_subset(pid, args.data_dir, args.device_indices)
        X_list.append(X)
        y_list.append(y)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    d_all = np.zeros(len(y_all), dtype=np.int64)

    X_train, X_val, y_train, y_val, _, _ = train_val_split(
        X_all, y_all, d_all, split_ratio=args.split_ratio
    )

    X_test, y_test = load_domain_data_subset(target_pid, args.data_dir, args.device_indices)

    train_set = MultisiteDataset(X_train, y_train, domain_idx=0)
    val_set   = MultisiteDataset(X_val,   y_val,   domain_idx=0)
    test_set  = MultisiteDataset(X_test,  y_test,  domain_idx=0)

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
        f"{args.backbone}_multisite_subset-{args.device_tag}"
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
    print(f"Found {len(participants)} participants: {participants}")
    print(f"Device subset: {args.device_tag} (indices {args.device_indices})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results_summary/results_subset-{args.device_tag}_{args.backbone}_{ts}.txt"
    os.makedirs("results_summary", exist_ok=True)

    header_lines = [
        "=" * 70,
        f"Multisite LOSO Results — device subset ({args.device_tag})",
        "=" * 70,
        f"Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Backbone     : {args.backbone}",
        f"Data dir     : {args.data_dir}",
        f"Device subset: {args.device_tag} (indices {args.device_indices})",
        f"Participants : {len(participants)}  ({', '.join(participants)})",
        "",
        f"{'Participant':<12}  {'MAE':>8}  {'RMSE':>8}  {'R':>8}",
        "-" * 70,
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(header_lines) + '\n')
    print(f"Writing → {out_path}")

    fold_results = []

    for pid in participants:
        print(f"\n  Target participant: {pid}")
        try:
            mae, rmse, r = train_one(args, pid, participants, seed_idx=0)
        except Exception as e:
            print(f"    ERROR for {pid}: {e}")
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
