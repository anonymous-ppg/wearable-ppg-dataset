import os
import subprocess
import sys

# BACKBONES = ['resnet', 'DCL', 'FCN', 'LSTM', 'Transformer']
BACKBONES = ['cnn_lstm']
# POSITIONS = ['earring', 'ring', 'watch', 'necklace']
POSITIONS = ['earring']
DATA_DIR  = '../../../anonymous-ppg-dataset/multisite-ppg-submission/ppg_windowed_data'

N_EPOCHS  = 20
PATIENCE  = 5
CUDA      = 0

# ── Preprocessing (set to True/False) ─────────────────────────────────────────
USE_PREPROCESS = False  # ← change this to switch

# ── Run ───────────────────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)

for position in POSITIONS:
    for backbone in BACKBONES:
        print(f"\n{'='*60}", flush=True)
        print(f"Position: {position} | Backbone: {backbone}", flush=True)
        print('='*60, flush=True)

        log_path = f"logs/{position}_{backbone}.txt"

        with open(log_path, 'w', buffering=1) as f:
            result = subprocess.run(
                [
                    sys.executable,
                    '-u',
                    'supervised/main_supervised_baseline.py',
                    '--dataset', 'ppg',
                    '--position', position,
                    '--backbone', backbone,
                    '--data_dir', DATA_DIR,
                    '--cuda', str(CUDA),
                    '--n_epoch', str(N_EPOCHS),
                    '--patience', str(PATIENCE),
                ] + (['--use_preprocess'] if USE_PREPROCESS else []),
                stdout=f,
                stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONPATH': os.path.abspath('.')},
            )


        if result.returncode != 0:
            print(f"  FAILED — see {log_path}", flush=True)
        else:
            print(f"  Done — log saved to {log_path}", flush=True)
