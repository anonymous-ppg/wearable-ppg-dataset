# =============================================================================
# quickstart.py — Quick verification of the codebase
#
# Run from the src/ directory:
#   PYTHONPATH=. python quickstart.py
#
# Runs single-device supervised baseline (earring, 20 epochs) on P7 & P8.
# No alignment step required.
#
# For the full LOSO sweep over all participants, change to full dataset.
# =============================================================================

import subprocess
import sys
import os

# ── Configuration — edit these two lines ─────────────────────────────────────
DATA_DIR = "../../../anonymous-ppg-dataset/multisite-ppg-submission/sample_data/ppg_windowed_data"  # sample dataset (P7, P8)
# DATA_DIR = "../../../anonymous-ppg-dataset/multisite-ppg-submission/ppg_windowed_data"       # full dataset (all participants)
CUDA     = 0
# ─────────────────────────────────────────────────────────────────────────────

env = {**os.environ, "PYTHONPATH": os.path.abspath(".")}

TESTS = [
    {
        "label": "[1/1] Single-device supervised  (earring, full LOSO)",
        "cmd": [
            sys.executable, "supervised/main_supervised_baseline.py",
            "--dataset",   "ppg",
            "--position",  "earring",
            "--backbone",  "DCL",
            "--data_dir",  DATA_DIR,
            "--cuda",      str(CUDA),
            "--n_epoch",   "20",
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

    for test in TESTS:
        run(test)

    print(f"\n{'='*60}")
    print(" Quick test passed.")
    print("="*60)

