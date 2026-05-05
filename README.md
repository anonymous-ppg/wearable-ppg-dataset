# Multi-site PPG: An In-the-Wild Physiological Dataset from Emerging Multi-site Wearables

## Quick Start

This repository provides baselines for heart rate estimation from PPG signals collected across four wearable sites (ring, earring, necklace, watch) in an in-the-wild setting. You can run heuristic baselines, supervised deep learning baselines, or optionally re-prepare the windowed dataset with a different window size.

- **Dataset** is available on [HuggingFace](https://huggingface.co/datasets/anonymous-ppg-dataset/multisite-ppg-submission).
- **Data download and preprocessing code** is included in this repository.


## Installation

Run 
```bash
pip install huggingface_hub
```
in your terminal if you don't have it installed yet. If it doesn’t allow you to install due to “error: externally-managed-environment,” you can override this, at the risk of breaking your Python installation or OS, by passing 
```bash
pip install huggingface_hub --break-system-packages
```

Then, clone our repo locally:
```bash
git clone https://github.com/anonymous-ppg/wearable-ppg-dataset.git
cd wearable-ppg-dataset
```

## Download Dataset

```bash
# for only sample data
python -c '
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="anonymous-ppg-dataset/multisite-ppg-submission",
    repo_type="dataset",
    local_dir="anonymous-ppg-dataset/multisite-ppg-submission",
    allow_patterns="sample_data/*"
)
'
```

```bash
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

#### 1.1 Dataset download

Use **Download Dataset** above.

#### 1.2 Code + dataset folders

Put **`wearable-ppg-dataset-main`** (the repo from GitHub) and **`anonymous-ppg-dataset`** (from HuggingFace) **in the same parent folder**. Don’t move or rename anything inside them—the runner finds the data automatically.

#### 1.3 Edit `config.py` (optional)

Set these fields in `src/heuristic_baselines/config.py`:

- **`HEURISTIC_DATA_SOURCE`**: **`"sample"`** (default) for the sample windowed data, or **`"full"`** for the full **`ppg_windowed_data`** tree.
- **`HEURISTIC_PIPELINE_PARTICIPANTS`**: which participants to run (default `["P3", "P4"]`).
- **`HEURISTIC_DEVICE_ROLES`**: devices to run (`Earring`, `Ring`, `Necklace`, `Watch`).
- **`HEURISTIC_PPG_CHANNELS`**: channels (`ppg_green`, `ppg_ir`).
- **`HEURISTIC_ALGORITHMS`**: heuristic methods among `pwd`, `msptd`, `fft`, `autocorr`, `heartpy`, `neurokit`, `qppgfast`.
- **`HEURISTIC_RUN_PREPROCESS`**: if **`True`** (default), apply detrend + bandpass and write cached **`outputs/<Px>/<stem>_preprocess.npz`** next to CSVs; if **`False`**, algorithms read the raw window NPZ fields directly.

With the layout in **1.2**, the default **`config.py`** is enough: **`pip install -r requirements.txt`**, then **`python runner.py`**.

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

**Set environment via Conda :**
```bash
conda env create -f environment.yaml
```
```bash
conda init
conda activate water
```

For a **quick smoke test** on sample data:
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

Set these fields in `src/prepare_windowed_dataset/config.py`:

- **`PIPELINE_PARTICIPANTS`**: choose which participants to run (e.g. `["P1", "P2"]`).
- **`WEARABLE_DEVICE_ROLES`**: choose devices to run (`Earring`, `Ring`, `Necklace`, `Watch`); names must match your raw NPZ filenames.
- **`ALIGNMENT_WINDOW_SEC`**: window length in seconds.
- **`ALIGNMENT_WINDOW_STRIDE_SEC`**: window stride in seconds.

If needed, also adjust:

- **`WINDOW_INPUT_ROOT`**: where `inputs/<Px>/` raw NPZ are read from.
- **`WINDOW_OUTPUT_ROOT`**: where `outputs/<Px>/` windowed NPZ are written.
- **`ECG_FS`**, **`PPG_WINDOW_GAP_THRESHOLD_MS`**, **`ECG_MIN_SAMPLES_FRAC`**: ECG/PPG quality and overlap settings used by the pipeline.

#### 3.4 Run

```bash
pip install -r requirements.txt
python run_pipeline.py
```

---

## License

This project uses the **MIT License**: you may use, modify, and redistribute the code (including in commercial products) if you keep the copyright and permission notice. The software is provided **as is**, without warranty. Full text: [`LICENSE`](LICENSE).
