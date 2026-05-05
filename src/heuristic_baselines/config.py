"""
Configuration for ``heuristic_baselines`` (PPG preprocess + heuristic HR models).

Window NPZ lives **next to** this package (same parent directory as ``heuristic_baselines/``), e.g. under ``src/``::

    ../ppg_windowed_data/<Px>/alignment_windows_<Px>_<Role>.npz   (``HEURISTIC_DATA_SOURCE="full"``)
    ../sample_data/ppg_windowed_data/<Px>/...                    (``HEURISTIC_DATA_SOURCE="sample"``)
    outputs/<Px>/model_results_*.csv, *_preprocess.npz            (write, under this package)

Run from this directory::

    python runner.py
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
# Parent of ``heuristic_baselines/`` (e.g. ``src/``): HF trees sit here as siblings.
HEURISTIC_DATA_PARENT = PACKAGE_ROOT.parent

# =============================================================================
# Manual settings (edit here only)
# =============================================================================
# Which tree to read (folders **next to** ``heuristic_baselines/``, not inside it).
# "full" -> <parent>/ppg_windowed_data/<Px>/...
# "sample" -> <parent>/sample_data/ppg_windowed_data/<Px>/...
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
    base = HEURISTIC_DATA_PARENT
    if key == "full":
        return (base / "ppg_windowed_data").resolve()
    if key == "sample":
        return (base / "sample_data" / "ppg_windowed_data").resolve()
    raise ValueError(
        f'HEURISTIC_DATA_SOURCE must be "full" or "sample", got {HEURISTIC_DATA_SOURCE!r}'
    )


HEURISTIC_WINDOWS_ROOT: Path = _resolve_heuristic_windows_root()
