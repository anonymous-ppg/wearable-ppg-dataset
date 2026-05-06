# encoding=utf-8
"""
main_supervised_baseline.py
----------------------------
Supervised baseline for PPG heart rate regression.

- Always performs regression (predicts bpm directly).
- Participants are discovered automatically from --data_dir subfolders.
- Leave-one-subject-out evaluation.

Usage:
    # Single-device (one wearable position)
    python main_supervised_baseline.py \\
        --dataset ppg --position ring \\
        --backbone DCL --data_dir /path/to/data --cuda 0

    # Multi-site (4 devices fused)
    python main_supervised_baseline.py \\
        --dataset multisite \\
        --backbone DCL --data_dir /path/to/data --cuda 0
"""

import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import argparse
from copy import deepcopy

from models.backbones import FCN, DeepConvLSTM, cnn_lstm, LSTM, Transformer
from models.models_nc import ResNet1D
from data_preprocess.data_prep import setup_dataloaders
from data_preprocess.participants_config import discover_participants
from data_preprocess.data_preprocess_multisite import discover_multisite_participants


# ── Argument parser ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Supervised baseline — PPG HR regression')

# Hardware
parser.add_argument('--cuda', default=0, type=int,
                    help='CUDA device index (0, 1, …). Falls back to CPU if unavailable.')

# Training hyperparameters
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epoch',    type=int, default=20)
parser.add_argument('--lr',         type=float, default=5e-4)
parser.add_argument('--patience',   type=int, default=5)

# Dataset
parser.add_argument('--dataset', default='ppg', type=str,
                    choices=['ppg', 'multisite'],
                    help='Dataset name. dataset=one device per run; multisite=4 devices fused.')
parser.add_argument('--position', default=None, type=str,
                    choices=['ring', 'earring', 'necklace', 'watch', None],
                    help='Wearable body position. Required for ppg, not used for multisite.')
parser.add_argument('--data_dir', default='../../../anonymous-ppg-dataset/multisite-ppg-submission/ppg_windowed_data', type=str,
                    help='Root directory containing per-participant subfolders'
                         '(e.g. P1/, P2/, …).')
parser.add_argument('--split_ratio', default=0.2, type=float,
                    help='Fraction of source-participant windows held out for validation.')
parser.add_argument('--cases', default='subject_val', type=str,
                    choices=['subject_val'],
                    help='Evaluation protocol. Only subject_val (leave-one-out) is supported.')
parser.add_argument('--single_device', type=int, default=None,
                    choices=[None, 0, 1, 2, 3],
                    help='For multisite dataset: None=all 4 devices; '
                         '0=earring, 1=ring, 2=watch, 3=necklace (single device on multisite split)')

# Backbone
parser.add_argument('--backbone', default='DCL', type=str,
                    choices=['FCN', 'DCL', 'cnn_lstm', 'LSTM', 'Transformer', 'resnet'],
                    help='Encoder architecture.')

# Logging
parser.add_argument('--logdir', default='log/', type=str,
                    help='Directory for per-run log files.')
parser.add_argument('--use_preprocess', action='store_true')


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
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
    """
    Instantiate the chosen backbone in regression mode (n_classes=1).
    All models receive n_channels=args.n_feature
    (1 for single, 4 for multisite, 8 for multisite+accel).
    """
    n_ch = args.n_feature
    len_ = args.len_sw      # 200

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
        raise NotImplementedError(f"Unknown backbone: {args.backbone}")


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args, train_loaders, val_loader, model, device, optimizer, criterion,
          save_dir='results/'):
    """
    Train with early stopping based on validation loss.
    Saves the best checkpoint to save_dir/{model_name}.pt.

    Returns the best model state dict.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, args.model_name + '.pt')

    min_val_loss  = float('inf')
    best_state    = None
    epochs_no_imp = 0

    for epoch in range(args.n_epoch):

        # ── training
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

        # ── validation
        if val_loader is None:
            best_state = deepcopy(model.state_dict())
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       checkpoint_path)
            continue

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for sample, target, _ in val_loader:
                sample = sample.to(device)
                target = target.to(device)
                out, _ = model(sample)
                val_loss += criterion(out.squeeze(1), target).item()
                n_val_batches += 1
        val_loss /= n_val_batches

        if val_loss < min_val_loss:
            min_val_loss  = val_loss
            best_state    = deepcopy(model.state_dict())
            epochs_no_imp = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       checkpoint_path)
            print(f"    Epoch {epoch+1:>3}/{args.n_epoch}  |  "
                  f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f} *")
        else:
            epochs_no_imp += 1
            print(f"    Epoch {epoch+1:>3}/{args.n_epoch}  |  "
                  f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
            if args.patience > 0 and epochs_no_imp >= args.patience:
                print(f"    Early stopping at epoch {epoch + 1} "
                      f"(no improvement for {args.patience} epochs)")
                break

    return best_state


# ── Evaluation ────────────────────────────────────────────────────────────────

def test(test_loader, model, device, criterion,
         save_path=None, append=False, participant_id=None):
    """
    Evaluate on the test set.

    Returns:
        mae  : Mean Absolute Error (bpm)
        rmse : Root Mean Squared Error (bpm)
        r    : Pearson correlation coefficient
    """
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
    r    = float(np.corrcoef(preds, targets)[0, 1])
    if np.isnan(r):
        r = 0.0

    # save predictions (appending across participants)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mode = 'a' if append else 'w'
        file_exists = os.path.exists(save_path)

        with open(save_path, mode) as f:
            if mode == 'w' or not file_exists:
                f.write('participant ground_truth_bpm predicted_bpm\n')
            tag = participant_id if participant_id is not None else 'unknown'
            for t, p in zip(targets, preds):
                f.write(f'{tag} {t:.3f} {p:.3f}\n')

    return mae, rmse, r


# ── Per-participant training run ──────────────────────────────────────────────

def train_sup(args, seed_idx: int):
    """
    Full train->test cycle for one (participant, seed) combination.

    args.target_domain must be set to the held-out participant ID (e.g. "P3")
    before calling this function.

    Returns:
        (mae, rmse, r)  — single test result
    """
    set_seed(seed_idx * 10 + np.random.randint(0, 10))

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    train_loaders, val_loader, test_loader = setup_dataloaders(args)

    model = build_model(args).to(device)

    position_tag = args.position if args.dataset == 'ppg' else 'multisite'
    args.model_name = (
        f"{args.backbone}_{args.dataset}_{position_tag}"
        f"_target{args.target_domain}"
        f"_seed{seed_idx}"
        f"_lr{args.lr}_bs{args.batch_size}"
    )

    os.makedirs(args.logdir, exist_ok=True)
    log_path = os.path.join(args.logdir, args.model_name + '.log')
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s %(message)s')

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state = train(args, train_loaders, val_loader, model, device, optimizer, criterion)

    model_test = build_model(args).to(device)
    model_test.load_state_dict(best_state)

    pred_path = f"predictions/{args.backbone}_{args.dataset}_{position_tag}.txt"
    mae, rmse, r = test(test_loader, model_test, device, criterion,
                        save_path=pred_path, append=True,
                        participant_id=args.target_domain)
    return mae, rmse, r


# ── Dataset config ────────────────────────────────────────────────────────────

def configure_dataset(args):
    args.len_sw  = 200
    args.out_dim = 200

    if args.dataset == 'ppg':
        args.n_feature = 1      # green only, single position
        if args.position is None:
            raise ValueError("--position is required for --dataset ppg.")
        participants = discover_participants(args.data_dir, args.position)
        if not participants:
            raise FileNotFoundError(
                f"No valid participant folders found in '{args.data_dir}' "
                f"for position '{args.position}'."
            )
        print(f"Found {len(participants)} participants for '{args.position}': {participants}")

    elif args.dataset == 'multisite':
        # n_feature: 4 if all devices fused, 1 if isolating a single device
        args.n_feature = 1 if args.single_device is not None else 4
        args.position  = None   # not used for multisite
        participants   = discover_multisite_participants(args.data_dir)
        if not participants:
            raise FileNotFoundError(
                f"No participants with aligned_4device.npz found in '{args.data_dir}'. "
                f"Run align.py --modality green first."
            )
        print(f"Found {len(participants)} multisite participants: {participants}")

    else:
        raise ValueError(f"Unknown dataset: '{args.dataset}'.")

    return participants


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    set_seed(40)
    args = parser.parse_args()

    participants = configure_dataset(args)

    all_seed_results = []

    for seed_idx in range(1):  # change to range(3) to run multiple seeds
        print(f"\n{'='*60}")
        print(f"Seed {seed_idx + 1}")
        print('='*60)

        fold_results = []
        for pid in participants:
            print(f"\n  Target participant: {pid}")
            args.target_domain = pid

            mae, rmse, r = train_sup(args, seed_idx)
            fold_results.append([mae, rmse, r])
            print(f"    MAE: {mae:.2f} bpm | RMSE: {rmse:.2f} bpm | R: {r:.4f}")

        fold_mean = np.mean(fold_results, axis=0)
        print(f"\n  Seed {seed_idx + 1} mean — "
              f"MAE: {fold_mean[0]:.2f}  RMSE: {fold_mean[1]:.2f}  R: {fold_mean[2]:.4f}")
        all_seed_results.append(fold_results)

    # ── Final summary across all participants
    flat = np.array(all_seed_results).reshape(-1, 3)   # ← (n_seeds * n_participants, 3)
    overall_mean = flat.mean(axis=0)
    overall_std  = flat.std(axis=0)
    print(f"\n{'='*60}")
    print(f"Final results (mean ± std across {flat.shape[0]} participant-runs)")
