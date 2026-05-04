"""Paths and NPZ helpers for heuristic batch."""
from __future__ import annotations

from pathlib import Path

import numpy as np


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


def participant_device_id(participant_id: str, role: str) -> str:
    pid = normalize_participant_id(participant_id)
    return f"{pid}_{role}"


def merged_windows_npz(windows_root: Path, participant_id: str, role: str) -> Path:
    pid = normalize_participant_id(participant_id)
    dev = participant_device_id(pid, role)
    return (windows_root / pid / f"alignment_windows_{dev}.npz").resolve()


def ppg_key_for_channel(ppg_channel: str) -> str:
    if ppg_channel == "ppg_ir":
        return "ppg_ir"
    if ppg_channel == "ppg_green":
        return "ppg_green"
    return "ppg"


def ppg_from_npz(data: dict | object, ppg_channel: str) -> np.ndarray:
    key = ppg_key_for_channel(ppg_channel)
    if isinstance(data, dict):
        if key in data:
            return np.asarray(data[key])
        return np.asarray(data["ppg"])
    if key in data.files:
        return np.asarray(data[key])
    return np.asarray(data["ppg"])
