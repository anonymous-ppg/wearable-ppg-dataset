# encoding=utf-8
"""
main_supervised_baseline_ir.py
------------------------------
Multisite training with green + IR modality (8-channel input).

  * --single_device None  : 8 channels (4 devices × green+IR)
  * --single_device 0-3   : 2 channels (one device's green+IR)
                            0=earring, 1=ring, 2=watch, 3=necklace

Usage:
    # 4-device fusion (8 channels)
    python main_supervised_baseline_ir.py --backbone resnet --target_domain P1

    # Single device + IR (2 channels)
    python main_supervised_baseline_ir.py --backbone resnet --target_domain P1 --single_device 0

    # All participants (LOSO sweep)
    python main_supervised_baseline_ir.py --backbone resnet

Note: Run `python align.py --modality ir --data_dir <path>` first to
generate the aligned_8channel.npz files.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
from copy import deepcopy

from models.backbones import FCN, DeepConvLSTM, cnn_lstm, LSTM, Transformer
from models.models_nc import ResNet1D
from data_preprocess.data_preprocess_multisite_ir import (
    discover_multisite_participants,
    prep_domains_subject_sp,
    DEVICE_NAMES,
)


# ── Argument parser ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description='Supervised baseline (green + IR) — PPG HR regression'
)
parser.add_argument('--cuda',          default=0, type=int)
parser.add_argument('--batch_size',    type=int, default=128)
parser.add_argument('--n_epoch',       type=int, default=20)
parser.add_argument('--lr',            type=float, default=5e-4)
parser.add_argument('--patience',      type=int, default=5)
parser.add_argument('--data_dir',      default='../../../anonymous-ppg-dataset/multisite-ppg-submission/ppg_windowed_data', type=str)
parser.add_argument('--split_ratio',   default=0.2, type=float)
parser.add_argument('--cases',         default='subject_val', type=str)
parser.add_argument('--single_device', type=int, default=None,
                    choices=[None, 0, 1, 2, 3],
                    help='None=all 4 devices (8 channels); '
                         '0=earring, 1=ring, 2=watch, 3=necklace (2 channels)')
parser.add_argument('--backbone',      default='resnet', type=str,
                    choices=['FCN', 'DCL', 'cnn_lstm', 'LSTM', 'Transformer', 'resnet'])
parser.add_argument('--logdir',        default='log/', type=str)
parser.add_argument('--target_domain', default=None, type=str,
                    help='Held-out participant (e.g. P1). '
                         'If unset, run LOSO sweep over all participants.')


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(args):
    n_ch = args.n_feature
    len_ = args.len_sw

    if args.backbone == 'FCN':
        return FCN(n_channels=n_ch, in_dim=len_, n_classes=1,
                   backbone=False, regress=True)
    elif args.backbone == 'DCL':
        return DeepConvLSTM(n_channels=n_ch, n_classes=1,
                            conv_kernels=64, kernel_size=5, LSTM_units=128,
                            backbone=False, regress=True)
    elif args.backbone == 'cnn_lstm':
        return cnn_lstm(n_channels=n_ch, n_classes=1,
                        backbone=False, regress=True)
    elif args.backbone == 'LSTM':
        return LSTM(n_channels=n_ch, n_classes=1, LSTM_units=256,
                    backbone=False, regress=True)
    elif args.backbone == 'Transformer':
        return Transformer(n_channels=n_ch, len_sw=len_, n_classes=1,
                           dim=128, depth=4, heads=4, mlp_dim=64,
                           dropout=0.1, backbone=False, regress=True)
    elif args.backbone == 'resnet':
        return ResNet1D(in_channels=n_ch, base_filters=32, kernel_size=5,
                        stride=2, groups=1, n_block=8, n_classes=1,
                        downsample_gap=2, increasefilter_gap=4,
                        backbone=False, regress=True)
    else:
        raise NotImplementedError(args.backbone)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args, train_loaders, val_loader, model, device, optimizer, criterion,
          save_dir='results/'):
    """Train with early stopping. Returns best model state dict."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, args.model_name + '.pt')

    min_val_loss  = float('inf')
    best_state    = None
    epochs_no_imp = 0

    for epoch in range(args.n_epoch):
        model.train()
        train_loss = 0.0
        for sample, target, _ in train_loaders[0]:
            sample = sample.to(device)
            target = target.to(device)
            out, _ = model(sample)
            loss   = criterion(out.squeeze(1), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loaders[0])

        if val_loader is None:
            best_state = deepcopy(model.state_dict())
            continue

        model.eval()
        val_loss = 0.0
        n        = 0
        with torch.no_grad():
            for sample, target, _ in val_loader:
                sample = sample.to(device)
                target = target.to(device)
                out, _ = model(sample)
                val_loss += criterion(out.squeeze(1), target).item()
                n += 1
        val_loss /= n

        if val_loss < min_val_loss:
            min_val_loss  = val_loss
            best_state    = deepcopy(model.state_dict())
            epochs_no_imp = 0
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
            print(f"    Epoch {epoch+1:>3}/{args.n_epoch} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} *")
        else:
            epochs_no_imp += 1
            print(f"    Epoch {epoch+1:>3}/{args.n_epoch} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            if args.patience > 0 and epochs_no_imp >= args.patience:
                print(f"    Early stop at epoch {epoch + 1}")
                break

    return best_state


# ── Evaluation ────────────────────────────────────────────────────────────────

def test(test_loader, model, device, criterion):
    """Returns (mae, rmse, r) — MAE/RMSE in bpm, Pearson correlation."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for sample, target, _ in test_loader:
            sample = sample.to(device)
            target = target.to(device)
            out, _ = model(sample)
            all_preds.append(out.squeeze(1).cpu())
            all_targets.append(target.cpu())
    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    r    = float(np.corrcoef(preds, targets)[0, 1]) if len(preds) > 1 else 0.0
    if np.isnan(r):
        r = 0.0
    return mae, rmse, r


# ── Configuration ────────────────────────────────────────────────────────────

def configure_args(args):
    args.len_sw    = 200
    args.out_dim   = 200
    # 2 channels per device (green+IR), 8 total for fusion
    args.n_feature = 2 if args.single_device is not None else 8
    return args


# ── Per-participant training run ──────────────────────────────────────────────

def train_sup(args, seed_idx=0):
    set_seed(seed_idx * 10 + np.random.randint(0, 10))
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    train_loaders, val_loader, test_loader = prep_domains_subject_sp(args)
    model = build_model(args).to(device)

    sd_tag = (f"_{DEVICE_NAMES[args.single_device]}IR"
              if args.single_device is not None else "_8chIR")
    args.model_name = (
        f"{args.backbone}_multisite_ir{sd_tag}"
        f"_target{args.target_domain}_seed{seed_idx}"
        f"_lr{args.lr}_bs{args.batch_size}"
    )

    os.makedirs(args.logdir, exist_ok=True)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state = train(args, train_loaders, val_loader, model, device, optimizer, criterion)

    model_test = build_model(args).to(device)
    model_test.load_state_dict(best_state)

    mae, rmse, r = test(test_loader, model_test, device, criterion)
    return mae, rmse, r


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    set_seed(40)
    args = parser.parse_args()
    args = configure_args(args)

    participants = discover_multisite_participants(args.data_dir)
    if not participants:
        raise FileNotFoundError(
            f"No aligned_8channel.npz found in '{args.data_dir}'. "
            f"Run `python align.py --modality ir --data_dir <path>` first."
        )

    # Single participant run
    if args.target_domain is not None:
        if args.target_domain not in participants:
            raise ValueError(
                f"target_domain '{args.target_domain}' not in {participants}"
            )
        print(f"\n=== Target {args.target_domain} ===")
        mae, rmse, r = train_sup(args, seed_idx=0)  # change seed_idx to run a different seed
        print(f"  MAE: {mae:.2f}  RMSE: {rmse:.2f}  R: {r:.4f}")

    # Full LOSO sweep
    else:
        print(f"Found {len(participants)} participants — LOSO sweep")
        fold_results = []
        for pid in participants:
            args.target_domain = pid
            print(f"\n=== Target {pid} ===")
            mae, rmse, r = train_sup(args, seed_idx=0)  # change seed_idx to run a different seed
            print(f"  MAE: {mae:.2f}  RMSE: {rmse:.2f}  R: {r:.4f}")
            fold_results.append([mae, rmse, r])

        mean = np.mean(fold_results, axis=0)
        std  = np.std(fold_results,  axis=0)
        print(f"\n{'='*60}")
        print(f"Final results across {len(fold_results)} LOSO folds")
        print('='*60)
        print(f"  MAE  : {mean[0]:.2f} ± {std[0]:.2f} bpm")
        print(f"  RMSE : {mean[1]:.2f} ± {std[1]:.2f} bpm")
        print(f"  R    : {mean[2]:.4f} ± {std[2]:.4f}")