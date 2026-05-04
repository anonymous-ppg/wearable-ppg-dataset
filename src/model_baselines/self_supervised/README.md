# Self-Supervised Learning for PPG Heart Rate Estimation

## Overview

This module implements and benchmarks self-supervised learning (SSL) frameworks for heart rate (HR) regression from PPG signals collected across four wearable device positions: **ring, earring, necklace, and watch**.

All methods follow a two-stage paradigm:
1. **SSL Pre-training** — learn representations from PPG windows without HR labels.
2. **Linear Probe** — freeze the encoder and train a linear regression head on labelled data.

The shared CNN-LSTM backbone (`models/backbones.py: cnn_lstm`) is reused across all methods for a fair comparison.

---

## Repository Structure

```
self_supervised/
├── main_byol.py              # BYOL pre-training + linear probe
├── main_simclr.py            # SimCLR pre-training + linear probe
└── run_selfsupervised_all.py # Batch runner across methods & positions
```

> **Shared dependencies** (in `src/`):
> - `models/backbones.py` — CNN-LSTM backbone
> - `data_preprocess/data_preprocess_dataset.py` — single-device data loader
> - `data_prep.py` — dataset dispatcher

---

## Methods

### 1. BYOL (`main_byol.py`)
Bootstrap Your Own Latent — momentum-based SSL that learns without negative pairs.

| Hyperparameter | Default |
|---|---|
| Pre-training epochs | 60 |
| BYOL learning rate | 0.001 |
| EMA decay | 0.99 |
| Projector size | 128 |
| Weight decay | 1.5e-6 |
| Linear probe epochs | 60 |
| Linear probe LR | 1e-3 |

### 2. SimCLR (`main_simclr.py`)
Contrastive learning with InfoNCE loss over augmented PPG window pairs.

| Hyperparameter | Default |
|---|---|
| Pre-training epochs | 120 |
| SimCLR learning rate | 0.003 |
| Temperature (InfoNCE) | 0.07 |
| Projector output dims | 128 |
| Linear probe epochs | 60 |
| Linear probe LR | 1e-3 |

---

## Data

**Signal:** Green-channel PPG only (`n_feature = 1`), 200-sample windows at 25 Hz (8 seconds).

**Default data directory:** `anonymous-ppg-dataset/multisite-ppg-submission/ppg_windowed_data`

**Evaluation protocol:** Leave-one-subject-out (`subject_val`) — each participant is held out as the test domain in turn.

---

## Running the Experiments

Run from `src/self_supervised/` with `PYTHONPATH=..`.

**BYOL:**
```bash
python main_byol.py --dataset ppg --position ring --cuda 0
```

**SimCLR:**
```bash
python main_simclr.py --dataset ppg --position ring --cuda 0
```

**Batch run** (edit `METHODS`, `POSITIONS`, and `DATA_DIR` at the top of the script):
```bash
python run_selfsupervised_all.py
```

Logs are saved to `logs/{position}_{method}.txt`.

### Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--position` | Wearable position (`ring`, `earring`, `necklace`, `watch`) | `ring` |
| `--dataset` | Dataset name | `ppg` |
| `--data_dir` | Path to participant data folders | see above |
| `--cuda` | GPU index | `0` |
| `--n_epoch` | Linear probe epochs | `60` |
| `--save_dir` | Model checkpoint output | `results/` |

---

## Output & Metrics

Each script reports three metrics across all leave-one-out folds:

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error (bpm) |
| **RMSE** | Root Mean Squared Error (bpm) |
| **R** | Pearson correlation coefficient |

Final summary (mean ± std across LOSO folds) is printed at the end of each run.
