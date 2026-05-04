"""Per-algorithm HR estimators (lazy import per algorithm name)."""
from __future__ import annotations

from typing import Callable

import numpy as np

AlgorithmFn = Callable[[np.ndarray, float], float]


def resolve_algorithm(name: str) -> tuple[AlgorithmFn, str]:
    n = name.strip().lower()
    if n == "pwd":
        from . import pwd

        return pwd.hr_from_pwd, "hr_pwd"
    if n == "msptd":
        from . import msptd

        return msptd.hr_from_msptd, "hr_msptd"
    if n == "fft":
        from . import fft

        return fft.hr_from_fft, "hr_fft"
    if n == "autocorr":
        from . import autocorr

        return autocorr.hr_from_autocorr, "hr_autocorr"
    if n == "heartpy":
        from . import heartpy

        return heartpy.hr_from_heartpy, "hr_heartpy"
    if n == "neurokit":
        from . import neurokit

        return neurokit.hr_from_neurokit_ppg, "hr_neurokit"
    if n == "qppgfast":
        from . import qppgfast

        return qppgfast.hr_from_qppgfast, "hr_qppgfast"
    raise KeyError(f"unknown algorithm: {name!r}")
