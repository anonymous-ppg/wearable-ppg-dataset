"""
Configuration for ``heuristic_baselines`` (PPG preprocess + heuristic HR models).

HuggingFace window NPZ is read from the **sibling** dataset tree (fixed layout)::

    <parent>/
      <this-repo>/      # clone name may vary, e.g. wearable-ppg-dataset or Multi-site-PPG-Dataset-anon
                         # .../src/heuristic_baselines/ = PACKAGE_ROOT
      anonymous-ppg-dataset/
        multisite-ppg-submission/
          ppg_windowed_data/<Px>/...                    (``HEURISTIC_DATA_SOURCE="full"``)
          sample_data/ppg_windowed_data/<Px>/...         (``HEURISTIC_DATA_SOURCE="sample"``)

Outputs (CSV, ``*_preprocess.npz``) go under this package: ``outputs/<Px>/``.

Run from this directory::

    python runner.py
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
# Repository root (parent of ``src/``); folder name does not matter.
REPO_ROOT = PACKAGE_ROOT.parent.parent
# Sibling of the code repo: anonymous-ppg-dataset/multisite-ppg-submission/
HEURISTIC_HF_SUBMISSION_ROOT: Path = (
    REPO_ROOT.parent / "anonymous-ppg-dataset" / "multisite-ppg-submission"
).resolve()

# =============================================================================
# Manual settings (edit here only)
# =============================================================================
# Which windowed tree under ``HEURISTIC_HF_SUBMISSION_ROOT`` to use.
# "full" -> .../ppg_windowed_data/<Px>/...
# "sample" -> .../sample_data/ppg_windowed_data/<Px>/...
HEURISTIC_DATA_SOURCE: str = "sample"

HEURISTIC_PIPELINE_PARTICIPANTS: list[str] = ["P3", "P4"]
HEURISTIC_DEVICE_ROLES: tuple[str, ...] = ("Earring", "Ring", "Necklace", "Watch")

HEURISTIC_RESULT_ROOT: Path = PACKAGE_ROOT / "outputs"

HEURISTIC_RUN_PREPROCESS: bool = True

# Single channel: one-element tuple, e.g. ("ppg_green","ppg_ir") — trailing comma required.
HEURISTIC_PPG_CHANNELS: tuple[str, ...] = ("ppg_green", "ppg_ir")

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


def _resolve_heuristic_windows_root() -> Path:
    key = HEURISTIC_DATA_SOURCE.strip().lower()
    base = HEURISTIC_HF_SUBMISSION_ROOT
    if key == "full":
        return (base / "ppg_windowed_data").resolve()
    if key == "sample":
        return (base / "sample_data" / "ppg_windowed_data").resolve()
    raise ValueError(
        f'HEURISTIC_DATA_SOURCE must be "full" or "sample", got {HEURISTIC_DATA_SOURCE!r}'
    )


HEURISTIC_WINDOWS_ROOT: Path = _resolve_heuristic_windows_root()
