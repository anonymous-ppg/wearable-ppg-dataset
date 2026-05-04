# Multi-site PPG: An In-the-Wild Physiological Dataset from Emerging Multi-site Wearables

## Quick Start

This repository provides baselines for heart rate estimation from PPG signals collected across four wearable sites (ring, earring, necklace, watch) in an in-the-wild setting. You can run heuristic baselines, supervised deep learning baselines, or optionally re-prepare the windowed dataset with a different window size.

- **Dataset** is available on [HuggingFace](https://huggingface.co/datasets/anonymous-ppg-dataset/multisite-ppg-submission).
- **Data download and preprocessing code** is included in this repository.


## Installation

```bash
git clone https://github.com/yourname/yourrepo.git
cd yourrepo
```

## Download Dataset

```bash
# for only sample data
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='anonymous-ppg-dataset/multisite-ppg-submission',
    repo_type='dataset',
    local_dir='anonymous-ppg-dataset/multisite-ppg-submission',
    allow_patterns="sample_data/*"
)
"

# for all 20 participant data
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='anonymous-ppg-dataset/multisite-ppg-submission',
    repo_type='dataset',
    local_dir='anonymous-ppg-dataset/multisite-ppg-submission'
)
"
```


## Running Experiments

### 1. Run Heuristic Baselines

All commands below are run from `src/heuristic_baselines/`:

```bash
cd src/heuristic_baselines/
```

#### 1.1 Download `ppg_windowed_data`

Use **Download Dataset** above. To fetch only that subtree, use `allow_patterns='ppg_windowed_data/*'` in `snapshot_download`; otherwise use the full download and read files under `ppg_windowed_data/`.

#### 1.2 Create `inputs/` and place windowed NPZ

Under this directory, create **`inputs/`**, then **`P1`**, **`P2`**, … inside it. Copy **windowed** alignment NPZ (after the windowing pipeline; filenames **`alignment_windows_*`**) so each file is **`inputs/<Px>/alignment_windows_<Px>_<Role>.npz`** (example: **`inputs/P1/alignment_windows_P1_Earring.npz`**).

#### 1.3 Edit `config.py`

Set **`HEURISTIC_PIPELINE_PARTICIPANTS`** and **`HEURISTIC_DEVICE_ROLES`** to match which participants you run and which `alignment_windows_*` files exist.

#### 1.4 Run

```bash
pip install -r requirements.txt
python runner.py
```


---

### 2. Run Supervised Baselines

All commands below are run from `src/model_baselines/` with `PYTHONPATH=.`:
```bash
cd src/model_baselines/
```
For a quick smoke test on sample data:
```bash
PYTHONPATH=. python quickstart.py
```

#### 2.1 Supervised Baseline on Single Device Dataset

```bash
# --position: earring | ring | watch | necklace
# --backbone: FCN | DCL | cnn_lstm | LSTM | Transformer | resnet
PYTHONPATH=. python supervised/main_supervised_baseline.py \
    --dataset ppg --position earring --backbone resnet --n_epoch 20

# Batch run (all backbones × positions):
PYTHONPATH=. python supervised/run_supervised_all.py
```

#### 2.2 Self-Supervised Baseline on Single Device Dataset

```bash
# BYOL
PYTHONPATH=. python self_supervised/main_byol.py --dataset ppg --position ring --cuda 0

# SimCLR
PYTHONPATH=. python self_supervised/main_simclr.py --dataset ppg --position ring --cuda 0

# Batch run (all methods × positions):
PYTHONPATH=. python self_supervised/run_selfsupervised_all.py
```

See [`src/model_baselines/self_supervised/README.md`](src/model_baselines/self_supervised/README.md) for BYOL and SimCLR hyperparameters.

#### 2.3 Multi-Site Results on 4-Device Aligned Dataset

Run alignment once per dataset first:
```bash
PYTHONPATH=. python multisite/aligned_4device.py --data_dir <path>
```

Then run 4-device fusion:
```bash
# 4-device green channel
PYTHONPATH=. python supervised/main_supervised_baseline.py \
    --dataset multisite --backbone resnet --n_epoch 20

# Device-subset sweep (0=earring 1=ring 2=watch 3=necklace):
PYTHONPATH=. python multisite/run_multisite_subset.py --backbone resnet --devices 0,1
```

#### 2.4 PPG-Motion Fusion Experiment

Run alignment with the corresponding modality first:
```bash
PYTHONPATH=. python multisite/aligned_4device.py --data_dir <path> --modality accel
PYTHONPATH=. python multisite/aligned_4device.py --data_dir <path> --modality ir
```

Then run fusion:
```bash
# 4-device green + accel_z
PYTHONPATH=. python multisite/main_supervised_baseline_accel.py --backbone resnet --n_epoch 20

# 4-device green + IR
PYTHONPATH=. python multisite/main_supervised_baseline_ir.py --backbone resnet --n_epoch 20

# Single device (--single_device: 0=earring 1=ring 2=watch 3=necklace):
PYTHONPATH=. python multisite/main_supervised_baseline_accel.py --backbone resnet --single_device 0
PYTHONPATH=. python multisite/main_supervised_baseline_ir.py    --backbone resnet --single_device 0
```

See [`src/model_baselines/README.md`](src/model_baselines/README.md) for the full list of arguments.


---

### 3. (Optional) Prepare Windowed Dataset with Different Window Size

Use this only if you want to **re-window raw recordings** (e.g. change window length or stride). All commands below are run from `src/prepare_windowed_dataset/`:

```bash
cd src/prepare_windowed_dataset/
```

#### 3.1 Get raw NPZ for windowing

Obtain **raw** PPG multi-sensor NPZ and ECG NPZ per participant (e.g. from the dataset’s **`raw_data/`** tree via **Download Dataset**, or `allow_patterns='raw_data/*'` if you only pull that subtree).

#### 3.2 Create `inputs/` and place raw NPZ

Under this directory, create **`inputs/`**, then **`P1`**, **`P2`**, … inside it. For each participant **`Px`**, add:

- **`inputs/<Px>/<Px>_<Role>_raw.npz`** for each **`Role`** you will run (names must match **`WEARABLE_DEVICE_ROLES`** in `config.py`, e.g. `Earring` → `inputs/P1/P1_Earring_raw.npz`).
- **`inputs/<Px>/<Px>_ecg_raw.npz`** once per participant.

#### 3.3 Edit `config.py`

Set **`PIPELINE_PARTICIPANTS`** and **`WEARABLE_DEVICE_ROLES`**. To change windowing, edit **`ALIGNMENT_WINDOW_SEC`** and **`ALIGNMENT_WINDOW_STRIDE_SEC`** (other fields in the manual block only if you need them).

#### 3.4 Run

```bash
pip install -r requirements.txt
python run_pipeline.py
```

---

## License

This project uses the **MIT License**: you may use, modify, and redistribute the code (including in commercial products) if you keep the copyright and permission notice. The software is provided **as is**, without warranty. Full text: [`LICENSE`](LICENSE).
