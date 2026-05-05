# =============================================================================
# quickstart.py — Quick verification of the codebase
#
# Run from the src/ directory:
#   PYTHONPATH=. python quickstart.py
#
# Step 0: aligns windows for P3 and P4 only (skips files that already exist).
# Step 1: single-device full LOSO over all participants, 5 epochs.
# Steps 2-4: multisite — train on P4, test on P3 (5 epochs).
#
# For the full LOSO sweep over all participants:
#   - remove --participants from the alignment commands
#   - remove --target_domain from the training commands
#   - set --n_epoch back to 20
# =============================================================================

import subprocess
import sys
import os

# ── Configuration — edit these two lines ─────────────────────────────────────
DATA_DIR = "../../../anonymous-ppg-dataset/multisite-ppg-submission/sample_data/ppg_windowed_data"  # sample dataset (P3, P4)
# DATA_DIR = "../../../anonymous-ppg-dataset/multisite-ppg-submission/ppg_windowed_data"       # full dataset (all participants)
CUDA     = 0
# ─────────────────────────────────────────────────────────────────────────────

env = {**os.environ, "PYTHONPATH": os.path.abspath(".")}

# ── Step 0: alignment (run once, skips participants that are already aligned) ─
ALIGN = [
    {
        "label": "[0a/4] Align windows — green (P3, P4 only)",
        "cmd": [sys.executable, "multisite/aligned_4device.py",
                "--data_dir", DATA_DIR, "--participants", "P3,P4"],
    },
    {
        "label": "[0b/4] Align windows — green + accel_z (P3, P4 only)",
        "cmd": [sys.executable, "multisite/aligned_4device.py",
                "--data_dir", DATA_DIR, "--modality", "accel",
                "--participants", "P3,P4"],
    },
    {
        "label": "[0c/4] Align windows — green + IR (P3, P4 only)",
        "cmd": [sys.executable, "multisite/aligned_4device.py",
                "--data_dir", DATA_DIR, "--modality", "ir",
                "--participants", "P3,P4"],
    },
]

# ── Steps 1-4: training tests ─────────────────────────────────────────────────
TESTS = [
    {
        "label": "[1/4] Single-device supervised  (earring, full LOSO)",
        "cmd": [
            sys.executable, "supervised/main_supervised_baseline.py",
            "--dataset",   "ppg",
            "--position",  "earring",
            "--backbone",  "resnet",
            "--data_dir",  DATA_DIR,
            "--cuda",      str(CUDA),
            "--n_epoch",   "5",
        ],
    },
    {
        "label": "[2/4] Multisite green  (4-device fusion, target=P3)",
        "cmd": [
            sys.executable, "supervised/main_supervised_baseline.py",
            "--dataset",       "multisite",
            "--backbone",      "resnet",
            "--target_domain", "P3",
            "--data_dir",      DATA_DIR,
            "--cuda",          str(CUDA),
            "--n_epoch",       "5",
        ],
    },
    {
        "label": "[3/4] Multisite green+accel  (8-channel fusion, target=P3)",
        "cmd": [
            sys.executable, "multisite/main_supervised_baseline_accel.py",
            "--backbone",      "resnet",
            "--target_domain", "P3",
            "--data_dir",      DATA_DIR,
            "--cuda",          str(CUDA),
            "--n_epoch",       "5",
        ],
    },
    {
        "label": "[4/4] Multisite green+IR  (8-channel fusion, target=P3)",
        "cmd": [
            sys.executable, "multisite/main_supervised_baseline_ir.py",
            "--backbone",      "resnet",
            "--target_domain", "P3",
            "--data_dir",      DATA_DIR,
            "--cuda",          str(CUDA),
            "--n_epoch",       "5",
        ],
    },
]


def run(test):
    print(f"\n{'='*60}")
    print(f" {test['label']}")
    print("="*60)
    result = subprocess.run(test["cmd"], env=env)
    if result.returncode != 0:
        print(f"\n  FAILED — see output above.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    print("\nQuick Start — PPG Heart Rate Baselines")
    print(f"Data dir   : {DATA_DIR}")
    print(f"PYTHONPATH : {os.path.abspath('.')}")

    print("\n── Step 0: aligning windows (skips if already done) ──")
    for step in ALIGN:
        run(step)

    print("\n── Steps 1-4: training tests ──")
    for test in TESTS:
        run(test)

    print(f"\n{'='*60}")
    print(" All quick tests passed.")
    print(" To run the full LOSO sweep, remove --target_domain")
    print(" and set --n_epoch back to 20.")
    print("="*60)
