# Supervised Baselines for PPG Heart Rate Estimation

## Overview

Supervised learning baselines for heart rate (HR) regression from PPG signals collected across four wearable devices: **ring, earring, necklace, and watch**.

Models are trained end-to-end with HR labels using a leave-one-subject-out (LOSO) evaluation protocol.

---

## Set up environment

```bash
cd src/model_baselines/
```
```bash
conda env create -f environment.yaml
conda activate water
```
> If `conda activate` fails, run `conda init` once and restart your shell.


---

## Quick Start

Run from the `src/model_baselines/` directory with `PYTHONPATH=.`:

```bash
cd src/model_baselines/
PYTHONPATH=. python quickstart.py        # 4 smoke tests, only use sample_data
```

Or run individual experiments directly.

**No alignment needed — single device:**
```bash
# --position can be: earring, ring, watch, necklace
# --n_epoch default is 20; reduce for a quick test (e.g. --n_epoch 5)
PYTHONPATH=. python supervised/main_supervised_baseline.py \
    --dataset ppg --position earring --backbone resnet --n_epoch 20

# Batch run (all backbones × positions) — edit BACKBONES/POSITIONS/DATA_DIR at top of file first:
PYTHONPATH=. python supervised/run_supervised_all.py
```

**Requires alignment first** (run once per dataset):
```bash
PYTHONPATH=. python multisite/aligned_4device.py --data_dir <path>                  # green
PYTHONPATH=. python multisite/aligned_4device.py --data_dir <path> --modality accel # green + accel_z
PYTHONPATH=. python multisite/aligned_4device.py --data_dir <path> --modality ir    # green + IR
```

Then run multisite experiments:
```bash
# Common options for all commands below:
#   --backbone    FCN | DCL | cnn_lstm | LSTM | Transformer | resnet (default: resnet)
#   --n_epoch     number of training epochs (default: 20)
#   --cuda        GPU index (default: 0)

# 4-device fusion — green
PYTHONPATH=. python supervised/main_supervised_baseline.py \
    --dataset multisite --backbone resnet --n_epoch 20

# 4-device fusion — green + accel_z
PYTHONPATH=. python multisite/main_supervised_baseline_accel.py --backbone resnet --n_epoch 20

# 4-device fusion — green + IR
PYTHONPATH=. python multisite/main_supervised_baseline_ir.py --backbone resnet --n_epoch 20

# Single device with accel or IR (--single_device: 0=earring 1=ring 2=watch 3=necklace):
PYTHONPATH=. python multisite/main_supervised_baseline_accel.py --backbone resnet --single_device 0
PYTHONPATH=. python multisite/main_supervised_baseline_ir.py    --backbone resnet --single_device 0

# Device-subset sweep (--devices: comma-separated indices, e.g. 0,1 = earring+ring):
PYTHONPATH=. python multisite/run_multisite_subset.py       --backbone resnet --devices 0,1
PYTHONPATH=. python multisite/run_multisite_subset_accel.py --backbone resnet --devices 0,1
```

**Data directory** (relative to `src/model_baselines/`):
```
# sample dataset (P7, P8 only):
../../../anonymous-ppg-dataset/multisite-ppg-submission/sample_data/ppg_windowed_data

# full dataset (all participants):
../../../anonymous-ppg-dataset/multisite-ppg-submission/ppg_windowed_data
```
Pass `--data_dir <path>` to override. `quickstart.py` defaults to the sample path.

---

## Repository Structure

```
src/model_baselines/
├── quickstart.py                         # Smoke-test runner for reviewers
├── environment.yaml                      # Conda environment
├── requirements.txt                      # Pip dependencies
│
├── data_preprocess/
│   ├── data_prep.py                      # Dataset loader dispatcher
│   ├── base_loader.py                    # Base PyTorch Dataset class
│   ├── participants_config.py            # Participant & device file discovery
│   ├── data_preprocess_dataset.py        # Single-device (per-position) loader
│   ├── data_preprocess_multisite.py      # 4-device green-channel loader
│   ├── data_preprocess_multisite_accel.py  # 4-device green+accel loader
│   ├── data_preprocess_multisite_ir.py     # 4-device green+IR loader
│   ├── data_preprocess_utils.py          # Train/val split utilities
│   └── run_preprocess_ppg.py             # Optional bandpass preprocessing
│
├── models/
│   ├── backbones.py                      # FCN, DCL, CNN-LSTM, Transformer
│   ├── models_nc.py                      # ResNet1D
│   └── attention.py                      # Attention module
│
├── supervised/
│   ├── main_supervised_baseline.py       # Single-device & green multisite training
│   └── run_supervised_all.py             # Batch runner (all backbones × positions)
│
├── multisite/
│   ├── aligned_4device.py                # Multi-device window alignment utility
│   ├── main_supervised_baseline_accel.py # Multisite green+accel training
│   ├── main_supervised_baseline_ir.py    # Multisite green+IR training
│   ├── run_multisite_subset.py           # LOSO sweep over a device subset (green)
│   └── run_multisite_subset_accel.py     # LOSO sweep over a device subset (accel)
│
└── self_supervised/                      # SSL baselines (BYOL, SimCLR) — see its README
```

---

## Data Format

**Input:** Green channel (optionally + IR or accel_z), resampled to 200 samples/window at 25 Hz (8-second windows).

Participants are auto-discovered from subfolders:
```
data_dir/
├── P1/
│   ├── alignment_windows_P1_Earring.npz
│   ├── alignment_windows_P1_Ring.npz
│   ├── alignment_windows_P1_Watch.npz
│   └── alignment_windows_P1_Necklace.npz
├── P2/
│   └── ...
```

Multisite runs also require pre-aligned files per participant (generated by `aligned_4device.py` — see alignment step in Quick Start):
```
P1/aligned_4device.npz           # green only (4 devices)
P1/aligned_8channel_accel.npz    # green + accel_z (4 devices × 2 channels)
P1/aligned_8channel.npz          # green + IR (4 devices × 2 channels)
```

---

## Backbone Models

| `--backbone` | Architecture |
|---|---|
| `FCN` | Fully Convolutional Network |
| `DCL` | DeepConvLSTM (64 conv kernels, kernel size 5, 128 LSTM units) |
| `cnn_lstm` | Lightweight CNN + LSTM |
| `LSTM` | Stacked LSTM (256 units) |
| `Transformer` | Transformer encoder (dim=128, depth=4, heads=4) |
| `resnet` | 1D ResNet (8 blocks, base filters=32, kernel size=5) |

---

## Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | `ppg` (single device) or `multisite` (4-device green) | `ppg` |
| `--position` | Wearable position — required for `ppg` | `None` |
| `--backbone` | Model architecture | `resnet` |
| `--data_dir` | Path to participant data folders | `../../../anonymous-ppg-dataset/...` |
| `--single_device` | `0–3` to isolate one device; omit for full fusion | `None` |
| `--cuda` | GPU device index | `0` |
| `--n_epoch` | Training epochs | `20` |
| `--lr` | Learning rate | `5e-4` |
| `--batch_size` | Training batch size | `128` |
| `--patience` | Early stopping patience | `5` |

---

## Evaluation Protocol

- **Split:** Leave-one-subject-out — each participant is held out as test; all others are pooled for training (80% train / 20% val).
- **Loss:** L1 (MAE).
- **Optimizer:** Adam.

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error (bpm) |
| **RMSE** | Root Mean Squared Error (bpm) |
| **R** | Pearson correlation coefficient |

Results are printed per participant and summarised as mean ± std across all LOSO folds.

---

## Self-Supervised Baselines

The `self_supervised/` subdirectory contains SSL methods (BYOL, SimCLR) that use the same backbone architectures and data pipeline. See [`self_supervised/README.md`](self_supervised/README.md) for details.
