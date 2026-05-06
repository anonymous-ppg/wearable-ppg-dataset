# encoding=utf-8
"""
run_selfsupervised_all.py
-------------------------
Batch runner — trains all SSL methods / position combinations in sequence.
Logs for each run are written to logs/{position}_{method}.txt.

Edit METHODS, POSITIONS, and DATA_DIR before running.

Usage:
    PYTHONPATH=. python self_supervised/run_selfsupervised_all.py
"""

import os
import subprocess
import sys

# ── Configuration ─────────────────────────────────────────────────────────────

METHODS   = ['byol']
# ['byol', 'simclr', 'ts2vec']
POSITIONS = ['earring']
# POSITIONS = ['ring', 'earring', 'necklace', 'watch']
DATA_DIR  = '../../../anonymous-ppg-dataset/multisite-ppg-submission/ppg_windowed_data'

N_EPOCH   = 60    # linear probe epochs
CUDA      = 0
USE_PREPROCESS = False  # ← change this to switch

# ── Run ───────────────────────────────────────────────────────────────────────

os.makedirs('logs', exist_ok=True)

for position in POSITIONS:
    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"Position: {position} | Method: {method}")
        print('='*60)

        log_path = f"logs/{position}_{method}.txt"

        with open(log_path, 'w', buffering=1) as f:
            result = subprocess.run(
                [
                    sys.executable,
                    '-u', 
                    f'self_supervised/main_{method}.py',
                    '--dataset',  'ppg',
                    '--position', position,
                    '--data_dir', DATA_DIR,
                    '--cuda',     str(CUDA),
                    '--n_epoch',  str(N_EPOCH),
                ] + (['--use_preprocess'] if USE_PREPROCESS else []),
                stdout=f,
                stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONPATH': os.path.abspath('.')},
            )
        if result.returncode != 0:
            print(f"  ✗ FAILED — see {log_path}")
        else:
            print(f"  ✓ Done   — log saved to {log_path}")
