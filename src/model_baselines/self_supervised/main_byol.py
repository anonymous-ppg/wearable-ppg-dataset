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
parser = argparse.ArgumentParser(description='BYOL self-supervised for PPG HR')
parser.add_argument('--cuda',          default=0,           type=int)
parser.add_argument('--batch_size',    default=256,         type=int)
parser.add_argument('--n_epoch',       default=60,          type=int,   help='epochs for linear probe')
parser.add_argument('--byol_epoch',    default=60,         type=int,   help='epochs for BYOL pretraining')
parser.add_argument('--lr',            default=1e-3,        type=float, help='lr for linear probe')
parser.add_argument('--byol_lr',       default=0.001,        type=float, help='lr for BYOL (from WildPPG)')
parser.add_argument('--ema_decay',     default=0.99,       type=float, help='EMA decay for target network')
parser.add_argument('--proj_size',     default=128,         type=int,   help='projector size (from WildPPG)')
parser.add_argument('--weight_decay',  default=1.5e-6,      type=float, help='weight decay (from WildPPG)')
parser.add_argument('--repr_dims',     default=128,         type=int,   help='cnn_lstm LSTM hidden size = 128')
parser.add_argument('--dataset',       default='ppg',  type=str)
parser.add_argument('--cases',         default='subject_val', type=str)
parser.add_argument('--split_ratio',   default=0.2,         type=float)
parser.add_argument('--target_domain', default='0',         type=str)
parser.add_argument('--position',      default='ring',
                    choices=['ring', 'earring', 'necklace', 'watch'])
parser.add_argument('--data_dir',      default='../../../anonymous-ppg-dataset/multisite-ppg-submission/ppg_windowed_data', type=str)
parser.add_argument('--logdir',        default='log/',      type=str)
parser.add_argument('--save_dir',      default='results/',  type=str)
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


# ── BYOL MLP (projector & predictor) ──────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)


# ── BYOL Model ─────────────────────────────────────────────────────────────────
class BYOL(nn.Module):
    """
    BYOL using the existing cnn_lstm backbone from backbones.py.

    cnn_lstm notes (from backbones.py):
      - Input:  (B, T, C)  — it does permute(0,2,1) internally to (B, C, T)
      - Output: (out, features) where features is (B, 128)  ← LSTM hidden=128
      - We take the second return value (features) as our representation
    """
    def __init__(self, n_channels=1, repr_dims=128,
                 proj_size=128, ema_decay=0.996):
        super().__init__()
        self.ema_decay = ema_decay

        # Online network
        self.online_encoder   = cnn_lstm(n_channels=n_channels,
                                         n_classes=1,
                                         backbone=True,
                                         regress=True)
        self.online_projector = MLP(repr_dims, proj_size * 2, proj_size)
        self.predictor        = MLP(proj_size, proj_size * 2, proj_size)

        # Target network (EMA of online, no grad)
        self.target_encoder   = cnn_lstm(n_channels=n_channels,
                                         n_classes=1,
                                         backbone=True,
                                         regress=True)
        self.target_projector = MLP(repr_dims, proj_size * 2, proj_size)

        # Copy weights and freeze target
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        self.target_projector.load_state_dict(self.online_projector.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        """EMA update of target network."""
        for o, t in zip(self.online_encoder.parameters(),
                        self.target_encoder.parameters()):
            t.data = self.ema_decay * t.data + (1 - self.ema_decay) * o.data
        for o, t in zip(self.online_projector.parameters(),
                        self.target_projector.parameters()):
            t.data = self.ema_decay * t.data + (1 - self.ema_decay) * o.data

    def forward(self, x1, x2):
        # x1, x2: (B, T, 1) — cnn_lstm permutes internally
        # Online branch
        _, f1_online = self.online_encoder(x1)   # (B, 128)
        _, f2_online = self.online_encoder(x2)
        z1_online = self.predictor(self.online_projector(f1_online))
        z2_online = self.predictor(self.online_projector(f2_online))

        # Target branch (no grad)
        with torch.no_grad():
            _, f1_target = self.target_encoder(x1)
            _, f2_target = self.target_encoder(x2)
            z1_target = self.target_projector(f1_target)
            z2_target = self.target_projector(f2_target)

        return z1_online, z2_online, z1_target.detach(), z2_target.detach()

    def encode(self, x):
        """Extract representations for downstream use."""
        _, features = self.online_encoder(x)
        return features                           # (B, 128)


def byol_loss(p, z):
    """Normalized MSE loss (cosine similarity)."""
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()


# ── Augmentations for PPG ──────────────────────────────────────────────────────
def augment_one_view(x):
    """
    Apply stochastic augmentations to one view of PPG data.
    x: (B, T, 1) — keep channel-last, cnn_lstm permutes internally.
    """
    # 1. Gaussian noise (always, varying strength)
    noise_std = 0.02 + 0.13 * torch.rand(1).item()  # 0.02-0.15
    x = x + noise_std * torch.randn_like(x)

    # 2. Random amplitude scaling (60% prob, larger range)
    if torch.rand(1).item() < 0.6:
        scale = 0.6 + 0.8 * torch.rand(x.size(0), 1, 1, device=x.device)  # 0.6-1.4
        x = x * scale

    # 3. Random time shift (50% prob)
    if torch.rand(1).item() < 0.5:
        shift = torch.randint(-15, 16, (1,)).item()
        x = torch.roll(x, shifts=shift, dims=1)

    # 4. Random masking (40% prob)
    if torch.rand(1).item() < 0.4:
        mask_len = torch.randint(10, 30, (1,)).item()
        if x.size(1) - mask_len > 0:
            mask_start = torch.randint(0, x.size(1) - mask_len, (1,)).item()
            x[:, mask_start:mask_start + mask_len, :] = 0

    # 5. Random DC shift (30% prob)
    if torch.rand(1).item() < 0.3:
        dc = (torch.rand(x.size(0), 1, 1, device=x.device) - 0.5) * 0.3
        x = x + dc

    return x


def augment(x):
    """
    Two independent augmented views for BYOL.
    Input/Output: (B, T, 1)
    """
    return augment_one_view(x.clone()), augment_one_view(x.clone())


# ── Pretraining ────────────────────────────────────────────────────────────────
def pretrain_byol(model, train_loader, args, DEVICE):
    optimizer = torch.optim.Adam(
        list(model.online_encoder.parameters()) +
        list(model.online_projector.parameters()) +
        list(model.predictor.parameters()),
        lr=args.byol_lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  # ← add
        optimizer, T_max=args.byol_epoch
    )
    
    model.train()
    for epoch in range(args.byol_epoch):
        total_loss = 0
        n_batches  = 0
        for batch in train_loader:
            sample, _, _ = batch
            x = sample.float().to(DEVICE)
            x = x[:, :, 0:1]                           # green only → (B, T, 1)
            
            # no permute — cnn_lstm handles it internally
            x1, x2 = augment(x)
            z1_online, z2_online, z1_target, z2_target = model(x1, x2)

            loss = (byol_loss(z1_online, z2_target) +
                    byol_loss(z2_online, z1_target)) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # ← add
            optimizer.step()
            model.update_target()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # Monitor representation collapse
        if epoch % 5 == 0 or epoch == args.byol_epoch - 1:
            model.eval()
            with torch.no_grad():
                first_batch = next(iter(train_loader))
                x_test = first_batch[0][:8].float().to(DEVICE)
                x_test = x_test[:, :, 0:1]
                _, f_test = model.online_encoder(x_test)
                f_std = f_test.std(dim=0).mean().item()
            warning = " ⚠️ COLLAPSE WARNING" if f_std < 0.05 else ""
            print(f"        Epoch {epoch}: representation std={f_std:.4f}{warning}")
            model.train()  # ← add
        print(f"  Epoch #{epoch}: loss={total_loss/n_batches:.6f}")

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
            
            # no permute — cnn_lstm handles it internally
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


def train_probe(X_train, y_train, X_val, y_val, repr_dims, args, DEVICE):
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
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
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_loss = sum(criterion(probe(xb.to(DEVICE)), yb.to(DEVICE)).item()
                          for xb, yb in val_dl)
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
def train_byol(args, i):
    set_seed(np.random.randint(i * 10, (i + 1) * 10))
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {DEVICE}")

    train_loaders, val_loader, test_loader = setup_dataloaders(args)
    train_loader = train_loaders[0]
    # Recreate train_loader with drop_last=True for BYOL pretrain (BatchNorm requires batch>1)
    train_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True, drop_last=True)

    model = BYOL(
        n_channels=1,              # green only
        repr_dims=args.repr_dims,  # 128 (cnn_lstm LSTM hidden size)
        proj_size=args.proj_size,  # 128 from WildPPG
        ema_decay=args.ema_decay   # 0.996 from WildPPG
    ).to(DEVICE)

    print(f"\n[BYOL] Pretraining for {args.byol_epoch} epochs ...")
    print(f"  Backbone: cnn_lstm (backbones.py), feature dim={args.repr_dims}")
    model = pretrain_byol(model, train_loader, args, DEVICE)

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.online_encoder.state_dict(), os.path.join(
        args.save_dir,
        f"byol_{args.dataset}_{args.position}_domain{args.target_domain}.pt"
    ))

    print("[BYOL] Extracting representations ...")
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
        mae, rmse, mr = train_byol(args, 0)
        per_participant[k] = [mae, rmse, mr]
        print(f'  {k} → MAE: {mae:.3f}, RMSE: {rmse:.3f}, R: {mr:.4f}')

    participant_results = np.array([per_participant[pid] for pid in domain])
    overall_mean = participant_results.mean(axis=0)
    overall_std  = participant_results.std(axis=0)

    print('\n===== Final Results (BYOL) =====')
    print(f'Across {len(domain)} participants (1 seed)')
    print(f'  MAE  : {overall_mean[0]:.3f} ± {overall_std[0]:.3f} bpm')
    print(f'  RMSE : {overall_mean[1]:.3f} ± {overall_std[1]:.3f} bpm')
    print(f'  R    : {overall_mean[2]:.4f} ± {overall_std[2]:.4f}')

    # Save summary
    os.makedirs("results", exist_ok=True)
    summary_path = f"results/summary_byol_{args.dataset}_{args.position}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Method       : BYOL\n")
        f.write(f"Dataset      : {args.dataset}\n")
        f.write(f"Position     : {args.position}\n")
        f.write(f"Participants : {len(domain)}\n")
        f.write(f"Seeds        : 1\n")
        f.write(f"\n")
        f.write(f"MAE  : {overall_mean[0]:.3f} ± {overall_std[0]:.3f} bpm\n")
        f.write(f"RMSE : {overall_mean[1]:.3f} ± {overall_std[1]:.3f} bpm\n")
        f.write(f"R    : {overall_mean[2]:.4f} ± {overall_std[2]:.4f}\n")
    print(f"Summary saved to {summary_path}")
