import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from models.backbones import cnn_lstm       # ← reuse existing backbone
from data_preprocess.data_prep import setup_dataloaders
from data_preprocess.participants_config import discover_participants



# ── Parser ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='SimCLR self-supervised for PPG HR')
parser.add_argument('--cuda',           default=0,           type=int)
parser.add_argument('--batch_size',     default=256,         type=int)
parser.add_argument('--n_epoch',        default=60,          type=int,   help='epochs for linear probe')
parser.add_argument('--simclr_epoch',   default=60,         type=int,   help='epochs for SimCLR pretraining')
parser.add_argument('--lr',             default=1e-3,        type=float, help='lr for linear probe')
parser.add_argument('--simclr_lr',      default=0.003,       type=float, help='lr for SimCLR (from WildPPG)')
parser.add_argument('--temperature',    default=0.07,        type=float, help='InfoNCE temperature')
parser.add_argument('--repr_dims',      default=128,         type=int,   help='cnn_lstm LSTM hidden size = 128')
parser.add_argument('--proj_dims',      default=128,         type=int,   help='projector output dimensions')
parser.add_argument('--dataset',        default='ppg',  type=str)
parser.add_argument('--cases',          default='subject_val', type=str)
parser.add_argument('--split_ratio',    default=0.2,         type=float)
parser.add_argument('--target_domain',  default='0',         type=str)
parser.add_argument('--position',       default='ring',
                    choices=['ring', 'earring', 'necklace', 'watch'])
parser.add_argument('--data_dir',       default='../../../anonymous-ppg-dataset/multisite-ppg-submission/ppg_windowed_data', type=str)
parser.add_argument('--logdir',         default='log/',      type=str)
parser.add_argument('--save_dir',       default='results/',  type=str)
parser.add_argument('--use_preprocess', action='store_true')

# ── Helpers ────────────────────────────────────────────────────────────────────
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_domain(args):
    if args.dataset == 'ppg':
        args.out_dim   = 200
        args.n_feature = 1          # green only
        args.len_sw    = 200
        args.fs        = 25
        args.lowest    = 30
        args.n_class   = 1
        args.data_type = 'ppg'
        participants = discover_participants(args.data_dir, args.position)
        args.n_subjects = len(participants)
        return participants


# ── SimCLR Projector: two FC layers (WildPPG spec) ────────────────────────────
class SimCLRProjector(nn.Module):
    def __init__(self, in_dim=128, proj_dims=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dims)
        )
    def forward(self, x):
        return self.net(x)


# ── SimCLR Model ───────────────────────────────────────────────────────────────
class SimCLR(nn.Module):
    """
    SimCLR using the existing cnn_lstm backbone from backbones.py.

    cnn_lstm notes (from backbones.py):
      - Input:  (B, T, C)  — it does permute(0,2,1) internally to (B, C, T)
      - Output: (out, features) where features is (B, 128)  ← LSTM hidden=128
      - backbone flag exists but forward() always returns (out, features)
      - We take the second return value (features) as our representation
    """
    def __init__(self, n_channels=1, repr_dims=128, proj_dims=128):
        super().__init__()
        self.encoder   = cnn_lstm(n_channels=n_channels,
                                  n_classes=1,
                                  backbone=True,
                                  regress=True)
        self.projector = SimCLRProjector(in_dim=repr_dims, proj_dims=proj_dims)

    def forward(self, x):
        # x: (B, T, C) — cnn_lstm permutes internally
        _, features = self.encoder(x)   # features: (B, 128)
        z = self.projector(features)    # z: (B, proj_dims)
        return z

    def encode(self, x):
        _, features = self.encoder(x)
        return features                 # (B, 128)


# ── InfoNCE Loss ───────────────────────────────────────────────────────────────
def info_nce_loss(z1, z2, temperature=0.07):
    B = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    z = torch.cat([z1, z2], dim=0)         # (2B, proj_dims)
    sim = torch.mm(z, z.T) / temperature   # (2B, 2B)

    # mask self-similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # positive pair for z1[i] is z2[i] at index i+B, and vice versa
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B,     device=z.device)
    ])
    return F.cross_entropy(sim, labels)


# ── Augmentations for PPG ──────────────────────────────────────────────────────
def augment(x):
    """
    Input/Output: (B, T, 1) — keep channel-last, cnn_lstm permutes internally.
    View 1: Gaussian noise
    View 2: random amplitude scaling
    """
    x1 = x + 0.05 * torch.randn_like(x)
    scale = 0.8 + 0.4 * torch.rand(x.size(0), 1, 1, device=x.device)
    x2 = x * scale
    return x1, x2


# ── Pretraining ────────────────────────────────────────────────────────────────
def pretrain_simclr(model, train_loader, args, DEVICE):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.simclr_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.simclr_epoch
    )

    model.train()
    for epoch in range(args.simclr_epoch):
        total_loss = 0
        n_batches  = 0
        for batch in train_loader:
            sample, _, _ = batch
            x = sample.float().to(DEVICE)
            x = x[:, :, 0:1]               # green only → (B, T, 1)

            x1, x2 = augment(x)
            z1 = model(x1)
            z2 = model(x2)
            loss = info_nce_loss(z1, z2, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        print(f"  Epoch #{epoch}: loss={total_loss/n_batches:.6f}  lr={scheduler.get_last_lr()[0]:.6f}")

    return model


# ── Extract representations ────────────────────────────────────────────────────
def extract_all_repr(model, loader, DEVICE):
    model.eval()
    all_repr, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            sample, target, _ = batch
            x = sample.float().to(DEVICE)
            x = x[:, :, 0:1]               # green only → (B, T, 1)

            features = model.encode(x)     # (B, 128)
            all_repr.append(features.cpu().numpy())
            all_y.append(target.numpy())
    return np.concatenate(all_repr), np.concatenate(all_y)


# ── Linear Probe ──────────────────────────────────────────────────────────────
class LinearProbe(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


# def train_probe(X_train, y_train, X_val, y_val, repr_dims, args, DEVICE):
#     train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
#     val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val))
#     train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
#     val_dl   = DataLoader(val_ds,   batch_size=args.batch_size)

#     probe     = LinearProbe(in_dim=repr_dims).to(DEVICE)
#     optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
#     criterion = nn.L1Loss()

#     best_val, best_state = 1e8, None
#     for epoch in range(args.n_epoch):
#         probe.train()
#         for xb, yb in train_dl:
#             xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#             loss = criterion(probe(xb), yb)
#             optimizer.zero_grad(); loss.backward(); optimizer.step()

#         probe.eval()
#         with torch.no_grad():
#             val_loss = sum(criterion(probe(xb.to(DEVICE)), yb.to(DEVICE)).item()
#                           for xb, yb in val_dl)
#         if val_loss < best_val:
#             best_val   = val_loss
#             best_state = deepcopy(probe.state_dict())

#     probe.load_state_dict(best_state)
#     return probe

def train_probe(X_train, y_train, X_val, y_val, repr_dims, args, DEVICE):
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size)

    probe     = LinearProbe(in_dim=repr_dims).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    best_val, best_state = 1e8, None
    for epoch in range(args.n_epoch):
        probe.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(probe(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(criterion(probe(xb), yb).item())
            val_loss = np.mean(val_losses)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = deepcopy(probe.state_dict())

    probe.load_state_dict(best_state)
    return probe


def test_probe(probe, X_test, y_test, DEVICE):
    probe.eval()
    with torch.no_grad():
        preds = probe(torch.FloatTensor(X_test).to(DEVICE)).cpu()
        trgs  = torch.FloatTensor(y_test)
        mae   = torch.mean(torch.abs(preds - trgs)).item()
        rmse  = torch.sqrt(torch.mean((preds - trgs) ** 2)).item()
        mr    = np.corrcoef(preds.numpy(), trgs.numpy())[0, 1]
        if np.isnan(mr): mr = 0.0
    return mae, rmse, mr


# ── Main training loop ─────────────────────────────────────────────────────────
def train_simclr(args, i):
    set_seed(np.random.randint(i * 10, (i + 1) * 10))
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {DEVICE}")

    train_loaders, val_loader, test_loader = setup_dataloaders(args)
    train_loader = train_loaders[0]

    model = SimCLR(
        n_channels=1,               # green only
        repr_dims=args.repr_dims,   # 128
        proj_dims=args.proj_dims,   # 128
    ).to(DEVICE)

    print(f"\n[SimCLR] Pretraining for {args.simclr_epoch} epochs ...")
    print(f"  Backbone: cnn_lstm (backbones.py), feature dim={args.repr_dims}")
    model = pretrain_simclr(model, train_loader, args, DEVICE)

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.encoder.state_dict(), os.path.join(
        args.save_dir,
        f"simclr_{args.dataset}_{args.position}_domain{args.target_domain}.pt"
    ))

    print("[SimCLR] Extracting representations ...")
    X_train_repr, y_train_repr = extract_all_repr(model, train_loader, DEVICE)
    X_val_repr,   y_val_repr   = extract_all_repr(model, val_loader,   DEVICE)
    X_test_repr,  y_test_repr  = extract_all_repr(model, test_loader,  DEVICE)

    print("[Probe] Training linear probe ...")
    probe = train_probe(
        X_train_repr, y_train_repr,
        X_val_repr,   y_val_repr,
        args.repr_dims, args, DEVICE
    )

    mae, rmse, mr = test_probe(probe, X_test_repr, y_test_repr, DEVICE)
    print(f"  → MAE={mae:.2f} bpm  RMSE={rmse:.2f} bpm  R={mr:.4f}")
    return mae, rmse, mr


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    set_seed(40)
    args = parser.parse_args()
    domain = set_domain(args)

    # Single seed; mean ± std reported across participants
    per_participant = {}
    for k in domain:
        setattr(args, 'target_domain', k)
        setattr(args, 'cases', 'subject_val')
        mae, rmse, mr = train_simclr(args, 0)
        per_participant[k] = [mae, rmse, mr]
        print(f'  {k} → MAE: {mae:.3f}, RMSE: {rmse:.3f}, R: {mr:.4f}')

    participant_results = np.array([per_participant[pid] for pid in domain])
    overall_mean = participant_results.mean(axis=0)
    overall_std  = participant_results.std(axis=0)

    print('\n===== Final Results (SimCLR) =====')
    print(f'Across {len(domain)} participants (1 seed)')
    print(f'  MAE  : {overall_mean[0]:.3f} ± {overall_std[0]:.3f} bpm')
    print(f'  RMSE : {overall_mean[1]:.3f} ± {overall_std[1]:.3f} bpm')
    print(f'  R    : {overall_mean[2]:.4f} ± {overall_std[2]:.4f}')

    # Save summary
    os.makedirs("results", exist_ok=True)
    summary_path = f"results/summary_simclr_{args.dataset}_{args.position}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Method       : SimCLR\n")
        f.write(f"Dataset      : {args.dataset}\n")
        f.write(f"Position     : {args.position}\n")
        f.write(f"Participants : {len(domain)}\n")
        f.write(f"Seeds        : 1\n")
        f.write(f"\n")
        f.write(f"MAE  : {overall_mean[0]:.3f} ± {overall_std[0]:.3f} bpm\n")
        f.write(f"RMSE : {overall_mean[1]:.3f} ± {overall_std[1]:.3f} bpm\n")
        f.write(f"R    : {overall_mean[2]:.4f} ± {overall_std[2]:.4f}\n")
    print(f"Summary saved to {summary_path}")
