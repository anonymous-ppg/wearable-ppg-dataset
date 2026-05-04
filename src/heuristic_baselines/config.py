"""
Configuration for ``heuristic_baselines`` (PPG preprocess + heuristic HR models).

Layout (this package root = ``PACKAGE_ROOT``)::

    inputs/<Px>/alignment_windows_<Px>_<Role>.npz   (read)
    outputs/<Px>/model_results_*.csv                (write)

Which participants run is ``HEURISTIC_PIPELINE_PARTICIPANTS``. Put window NPZ under
``inputs/<Px>/`` (e.g. copy from ``prepare_windowed_dataset`` outputs).

Run from ``src/heuristic_baselines`` (directory that contains this ``config.py``)::

    python runner.py
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

# =============================================================================
# Manual settings (edit here only)
# =============================================================================
HEURISTIC_PIPELINE_PARTICIPANTS: list[str] = ["P1"]
HEURISTIC_DEVICE_ROLES: tuple[str, ...] = ("Earring", "Ring", "Necklace", "Watch")

# Read merged window NPZ from inputs/<Px>/; write CSV under outputs/<Px>/
HEURISTIC_WINDOWS_ROOT: Path = PACKAGE_ROOT / "inputs"
HEURISTIC_RESULT_ROOT: Path = PACKAGE_ROOT / "outputs"

HEURISTIC_RUN_PREPROCESS: bool = True

# Single channel: one-element tuple, e.g. ("ppg_green","ppg_ir") — trailing comma required.
HEURISTIC_PPG_CHANNELS: tuple[str, ...] = ("ppg_green","ppg_ir")

HEURISTIC_ALGORITHMS: tuple[str, ...] = (
    "pwd",
    "msptd",
    "fft",
    "autocorr",
    "heartpy",
    "neurokit",
    "qppgfast",
)

KNOWN_HEURISTIC_ALGORITHMS: frozenset[str] = frozenset(
    ("pwd", "msptd", "fft", "autocorr", "heartpy", "neurokit", "qppgfast")
)
