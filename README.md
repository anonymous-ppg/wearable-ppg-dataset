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

### for only sample data
```bash
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

### for all 20 participant data
```bash
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

Put **`wearable-ppg-dataset`** (the repo from GitHub) and **`anonymous-ppg-dataset`** (from HuggingFace) **in the same parent folder**. Don’t move or rename anything inside them—the runner finds the data automatically.

#### 1.3 Edit `config.py`

Set these fields in `src/heuristic_baselines/config.py`:

- **`HEURISTIC_DATA_SOURCE`**: **`"sample"`** (default) for the sample windowed data, or **`"full"`** for the full **`ppg_windowed_data`** tree.
- **`HEURISTIC_PIPELINE_PARTICIPANTS`**: which participants to run (default `["P7", "P8"]`).
- **`HEURISTIC_DEVICE_ROLES`**: devices to run (`Earring`, `Ring`, `Necklace`, `Watch`).
- **`HEURISTIC_PPG_CHANNELS`**: channels (`ppg_green`, `ppg_ir`).
- **`HEURISTIC_ALGORITHMS`**: heuristic methods among `pwd`, `msptd`, `fft`, `autocorr`, `heartpy`, `neurokit`, `qppgfast` (default runs only `neurokit`).
- **`HEURISTIC_RUN_PREPROCESS`**: if **`True`** (default), apply detrend + bandpass and write cached **`outputs/<Px>/<stem>_preprocess.npz`** next to CSVs; if **`False`**, algorithms read the raw window NPZ fields directly.

With the layout in **1.2**, the default **`config.py`** is enough: **`pip install -r requirements.txt`**, then **`python runner.py`**.

#### 1.4 Run

```bash
pip install -r requirements.txt
python runner.py
```
For the default sample-data setup, a full heuristic run typically takes about **16 minutes** in total.

---

### 2. Run Supervised Baselines

All commands below are run from `src/model_baselines/` with `PYTHONPATH=.`:
```bash
cd src/model_baselines/
```

#### Set up environment
```bash
conda env create -f environment.yaml
conda activate water
```

> If `conda activate` fails, run `conda init` once and restart your shell.

#### Quick smoke test

Run an end-to-end check on the sample data (DCL method, earring site):

```bash
PYTHONPATH=. python quickstart.py
```
Expected runtime: ~5 minutes on a single NVIDIA H100.

**Setting `--data_dir` (commands 2.1–2.4).** 
The default `--data_dir` points to the full dataset, so **if you have not downloaded the full dataset**, you must override it to use the sample data:
>
> ```
> --data_dir ../../../anonymous-ppg-dataset/multisite-ppg-submission/sample_data/ppg_windowed_data
> ```
>
> If you omit `--data_dir`, the script defaults to the full-dataset path.

#### 2.1 Supervised Baseline on Single Device Dataset

```bash
# --position: earring | ring | watch | necklace
# --backbone: FCN | DCL | cnn_lstm | LSTM | Transformer | resnet
PYTHONPATH=. python supervised/main_supervised_baseline.py \
    --dataset ppg --position earring --backbone DCL --data_dir <path>

```

#### 2.2 Self-Supervised Baseline on Single Device Dataset

```bash
# BYOL
PYTHONPATH=. python self_supervised/main_byol.py --dataset ppg --position earring --cuda 0 --data_dir <path>
```
```bash
# SimCLR
PYTHONPATH=. python self_supervised/main_simclr.py --dataset ppg --position earring --cuda 0 --data_dir <path>

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
    --dataset multisite --backbone DCL --data_dir <path>
```
```bash
# Device-subset sweep (0=earring 1=ring 2=watch 3=necklace):
PYTHONPATH=. python multisite/run_multisite_subset.py --backbone DCL --devices 0,1 --data_dir <path>
```

#### 2.4 PPG-Motion Fusion Experiment

Run alignment with the corresponding modality first:
```bash
PYTHONPATH=. python multisite/aligned_4device.py --data_dir <path> --modality accel
```
Then run fusion:
```bash
# 4-device green + accel_z
PYTHONPATH=. python multisite/main_supervised_baseline_accel.py --backbone resnet --data_dir <path>
```

See [`src/model_baselines/README.md`](src/model_baselines/README.md) for the full list of arguments.


---

### 3. (Optional) Prepare Windowed Dataset with Different Window Size

Use this only if you want to **re-window raw recordings** (e.g. change window length or stride). All commands below are run from `src/prepare_windowed_dataset/`:

```bash
cd src/prepare_windowed_dataset/
```

#### 3.1 Dataset download

Use **Download Dataset** above.

#### 3.2 Code + dataset folders

Put **`wearable-ppg-dataset`** (the repo from GitHub) and **`anonymous-ppg-dataset`** (from HuggingFace) **in the same parent folder**. Don’t move or rename anything inside them—the runner resolves raw-data paths automatically.

#### 3.3 Edit `config.py`

Set these fields in `src/prepare_windowed_dataset/config.py`:

- **`WINDOW_DATA_SOURCE`**: **`"sample"`** (default) uses `sample_data/raw_data`; **`"full"`** uses `raw_data`.
- **`PIPELINE_PARTICIPANTS`**: participants to run (default `["P7", "P8"]`).
- **`WEARABLE_DEVICE_ROLES`**: devices to run (default `("Earring",)`).
- **`ALIGNMENT_WINDOW_SEC`**: window length in seconds.
- **`ALIGNMENT_WINDOW_STRIDE_SEC`**: window stride in seconds.
- **`ECG_FS`**, **`PPG_WINDOW_GAP_THRESHOLD_MS`**, **`ECG_MIN_SAMPLES_FRAC`**: ECG/PPG quality and overlap settings used by the pipeline.

With the layout in **3.2**, the default **`config.py`** is enough for a sample run.

#### 3.4 Run

```bash
pip install -r requirements.txt
python run_pipeline.py
```

---

## License

This project uses the **MIT License**: you may use, modify, and redistribute the code (including in commercial products) if you keep the copyright and permission notice. The software is provided **as is**, without warranty. Full text: [`LICENSE`](LICENSE).

## Acknowledgments

Baseline training code builds on the [WildPPG](https://github.com/eth-siplab/WildPPG) repository (Meier et al., NeurIPS 2024 Datasets and Benchmarks Track).
