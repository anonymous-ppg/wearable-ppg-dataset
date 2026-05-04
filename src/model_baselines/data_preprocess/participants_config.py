# encoding=utf-8
"""
participants_config.py
----------------------
Auto-discovers participant folders and their device files.

Expected data directory layout:
    data_dir/
    ├── P1/
    │   ├── alignment_windows_P1_Ring.npz
    │   ├── alignment_windows_P1_Earring.npz
    │   ├── alignment_windows_P1_Necklace.npz
    │   └── alignment_windows_P1_Watch.npz
    ├── P2/
    │   └── ...
    └── P20/
        └── ...
Each .npz contains keys: ppg_green, ppg_ir, hr_gt, ppg_fs, ...
"""

import os
import glob


POSITION_TO_DEVICE = {
    'ring':     'Ring',
    'earring':  'Earring',
    'necklace': 'Necklace',
    'watch':    'Watch',
}


def find_device_files(participant_dir: str, position: str):
    """
    Find the green and IR .npz files for a given position inside one
    participant's folder. The device number is unknown, so we glob for it.

    Returns:
        (green_path)  if green files exist
        None          if green file is missing

    Raises:
        ValueError  if more than one device of the same type is found
                    (e.g. two Ring files in the same folder)
    """
    device = POSITION_TO_DEVICE[position]

    green_matches = sorted(glob.glob(
        os.path.join(participant_dir, f"alignment_windows_*_{device}.npz")
    ))

    if len(green_matches) == 0:
        return None

    if len(green_matches) > 1:
        raise ValueError(
            f"Multiple {device} green files found in {participant_dir}:\n"
            f"  {green_matches}\n"
            f"Expected exactly one per participant."
        )

    return green_matches[0]


def discover_participants(data_dir: str, position: str) -> list:
    """
    Scan data_dir for participant subfolders that have both green and IR
    files for the given position (any device number).

    Returns a sorted list of folder names, e.g. ["P1", "P2", ..., "P18"].
    Folders missing the required files are skipped with a warning.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    participants = []
    for name in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder):
            continue
        result = find_device_files(folder, position)
        if result is not None:
            participants.append(name)
        else:
            print(f"  Skipping '{name}': no {position} files found.")

    return participants
