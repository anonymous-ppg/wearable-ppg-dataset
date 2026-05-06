"""
Configuration for ``prepare_windowed_dataset`` (windowing + Pan-Tompkins batch only).

Raw NPZ is read from the sibling HuggingFace dataset tree (fixed layout)::

    <parent>/
      <this-repo>/src/prepare_windowed_dataset/    (PACKAGE_ROOT)
      anonymous-ppg-dataset/
        multisite-ppg-submission/
          raw_data/<Px>/...               (``WINDOW_DATA_SOURCE="full"``)
          sample_data/raw_data/<Px>/...   (``WINDOW_DATA_SOURCE="sample"``)

Window outputs are written under this package::

    outputs/<Px>/alignment_windows_<Px>_<Role>.npz, ...

Run from ``src/prepare_windowed_dataset`` (directory containing this ``config.py``)::

    python run_pipeline.py
    python run_pipeline.py --help
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
WINDOW_HF_SUBMISSION_ROOT: Path = (
    REPO_ROOT.parent / "anonymous-ppg-dataset" / "multisite-ppg-submission"
).resolve()

# =============================================================================
# Manual settings (edit here only)
# =============================================================================
# Participants to process in batch mode.
PIPELINE_PARTICIPANTS: list[str] = ["P7", "P8"]

# Wearables (must match filenames; available: Earring, Ring, Necklace, Watch;
# e.g. P1_Earring_raw.npz -> "Earring").
WEARABLE_DEVICE_ROLES: tuple[str, ...] = ("Earring",)

# Which HF raw_data tree to read.
# "sample" -> .../sample_data/raw_data/<Px>/...
# "full" -> .../raw_data/<Px>/...
WINDOW_DATA_SOURCE: str = "sample"


def _resolve_window_input_root() -> Path:
    key = WINDOW_DATA_SOURCE.strip().lower()
    base = WINDOW_HF_SUBMISSION_ROOT
    if key == "sample":
        return (base / "sample_data" / "raw_data").resolve()
    if key == "full":
        return (base / "raw_data").resolve()
    raise ValueError(f'WINDOW_DATA_SOURCE must be "sample" or "full", got {WINDOW_DATA_SOURCE!r}')


WINDOW_INPUT_ROOT = _resolve_window_input_root()
WINDOW_OUTPUT_ROOT = PACKAGE_ROOT / "outputs"

# Back-compat names used by run_pipeline / helpers
DATALOADER_INPUT_ROOT = WINDOW_INPUT_ROOT
DATALOADER_OUTPUT_ROOT = WINDOW_OUTPUT_ROOT
DATALOADER_PIPELINE_PARTICIPANTS = PIPELINE_PARTICIPANTS
DATA_LOADER_ROOT = PACKAGE_ROOT

# Windowing and ECG constants
ALIGNMENT_WINDOW_SEC = 8.0
ALIGNMENT_WINDOW_STRIDE_SEC = 1.0
ECG_FS = 130.0
PPG_WINDOW_GAP_THRESHOLD_MS = 15.0
ECG_MIN_SAMPLES_FRAC = 0.95


# =============================================================================
# Used by scripts; batch runs set RUN_* env vars — normally do not edit
# =============================================================================


def _dir_from_env(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    if not v:
        return default
    p = Path(v)
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


DEVICE = os.environ.get("RUN_DEVICE", "Earring")
PPG_CHANNEL = os.environ.get("RUN_PPG_CHANNEL", "ppg_green")
SUBJECT = os.environ.get("RUN_SUBJECT", "unknown")

OUTPUTS_DIR = _dir_from_env("RUN_OUTPUTS_DIR", PACKAGE_ROOT / "outputs" / "dataset")

ALIGNMENT_WINDOWS_BASE = f"alignment_windows_{DEVICE}_{PPG_CHANNEL}"


def normalize_participant_id(name: str) -> str:
    s = str(name).strip()
    if not s:
        raise ValueError("Empty participant id")
    if s.upper().startswith("P"):
        num = s[1:]
    else:
        num = s
    if not num.isdigit():
        raise ValueError(f"Invalid participant id: {name}")
    return f"P{int(num)}"


def dataloader_ppg_raw_npz(participant_id: str, role: str) -> Path:
    pid = normalize_participant_id(participant_id)
    return WINDOW_INPUT_ROOT / pid / f"{pid}_{role}_raw.npz"


def dataloader_ecg_raw_npz(participant_id: str) -> Path:
    pid = normalize_participant_id(participant_id)
    return WINDOW_INPUT_ROOT / pid / f"{pid}_polar_ecg_raw.npz"


def participant_output_device(participant_id: str, role: str) -> str:
    return f"{participant_id}_{role}"
