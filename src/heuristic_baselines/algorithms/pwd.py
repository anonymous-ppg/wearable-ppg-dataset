"""
PWD: Pulse Wave Delineator HR from PPG (per-window).

Reference: B. N. Li et al., Biomedical Signal Processing and Control, 2010.
Ported from ppg-beats MATLAB (2-Clause BSD).
"""
import numpy as np
from scipy.signal import bessel, filtfilt
from scipy.ndimage import uniform_filter1d

HR_MIN_BPM = 30
HR_MAX_BPM = 200


def _smooth(x: np.ndarray, size: int = 5) -> np.ndarray:
    """Moving-average smooth, similar to MATLAB smooth()."""
    size = max(1, min(size, len(x)))
    if size % 2 == 0:
        size += 1
    return uniform_filter1d(x.astype(np.float64), size=size, mode="nearest")


def _seeklocales(
    tempsig: np.ndarray, tempbegin: int, tempend: int
) -> tuple[float, int, float, int]:
    """Min/max and positions in [tempbegin, tempend]; 0-based indices."""
    seg = tempsig[tempbegin : tempend + 1]
    minip = tempbegin + int(np.argmin(seg))
    maxip = tempbegin + int(np.argmax(seg))
    return float(np.min(seg)), minip, float(np.max(seg)), maxip


def _seekdicrotic(tempdiff: np.ndarray) -> int:
    """
    Find dicrotic notch in a derivative segment.
    Returns offset within segment (0-based); 0 if not found.
    """
    tempdiff = _smooth(tempdiff)
    temp_len = len(tempdiff) - 3
    if temp_len < 3:
        return 0

    tzc_min = []
    for itemp in range(2, temp_len):
        if tempdiff[itemp] * tempdiff[itemp + 1] <= 0:
            if tempdiff[itemp - 2] < 0 and tempdiff[itemp + 2] >= 0:
                tzc_min.append(itemp)

    izc_min = len(tzc_min)
    if izc_min == 0:
        itemp_min = 2 + int(np.argmin(tempdiff[2:temp_len]))
        for itemp in range(itemp_min + 1, temp_len):
            if tempdiff[itemp + 1] <= tempdiff[itemp - 1]:
                return itemp
        return 0
    if izc_min == 1:
        return tzc_min[0]
    itemp_max = 2 + int(np.argmax(tempdiff[2:temp_len]))
    for i in range(izc_min - 1, -1, -1):
        if tzc_min[i] < itemp_max:
            return tzc_min[i]
    return 0


def _delineator(abpsig: np.ndarray, abpfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """
    PWD delineator core. Returns (onsets, peaks) as 0-based indices.
    """
    abpsig = np.asarray(abpsig, dtype=np.float64)
    od = 3
    step_win = int(2 * abpfreq)
    close_win = max(1, int(0.1 * abpfreq))
    sig_len = len(abpsig)

    if sig_len < close_win + 10:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # Bessel low-pass ~25 Hz (normalized to sampling rate)
    coh = 25.0 * 2 / abpfreq
    coh = min(0.99, max(0.01, coh))
    b, a = bessel(od, coh, btype="low")
    abpsig = filtfilt(b, a, abpsig)
    abpsig = 10.0 * abpsig
    abpsig = _smooth(abpsig)

    # First derivative
    ttp = np.diff(abpsig)
    diff1 = np.zeros(sig_len, dtype=np.float64)
    diff1[1:] = ttp * 100
    diff1[0] = diff1[1]
    diff1 = _smooth(diff1)

    # Adaptive threshold
    if sig_len > 12 * abpfreq:
        tk = 10
    elif sig_len > 7 * abpfreq:
        tk = 5
    elif sig_len > 4 * abpfreq:
        tk = 2
    else:
        tk = 1

    if tk > 1:
        tatom = sig_len // (tk + 2)
        temp_th = []
        for ji in range(tk):
            sig_idx = (ji + 1) * tatom
            temp_idx = min(sig_idx + int(abpfreq), sig_len - 1)
            temp_min, _, temp_max, _ = _seeklocales(abpsig, sig_idx, temp_idx)
            temp_th.append(temp_max - temp_min)
        abp_max_th = float(np.mean(temp_th))
    else:
        temp_min, _, temp_max, _ = _seeklocales(abpsig, close_win, sig_len - 1)
        abp_max_th = temp_max - temp_min

    abp_max_lt = 0.4 * abp_max_th

    peak_list = []
    onset_list = []
    dicro_list = []
    peak_index = 0
    onset_index = 0
    dicro_index = 0

    diff_index = close_win
    while diff_index < sig_len - 1:
        temp_min = abpsig[diff_index]
        temp_max = abpsig[diff_index]
        temp_index = diff_index
        tpeakp = diff_index
        tonsetp = diff_index

        while temp_index < sig_len - 1:
            if (temp_index - diff_index) > step_win:
                temp_index = diff_index
                abp_max_th = 0.6 * abp_max_th
                if abp_max_th <= abp_max_lt:
                    abp_max_th = 2.5 * abp_max_lt
                break

            if diff1[temp_index - 1] * diff1[temp_index + 1] <= 0:
                jk = min(temp_index + 5, sig_len - 1)
                jj = max(0, temp_index - 5)

                if (jk - temp_index) >= 5:
                    ttk = temp_index
                    while ttk <= jk and ttk < sig_len:
                        if diff1[ttk] != 0:
                            break
                        ttk += 1
                    if ttk > jk:
                        break

                if diff1[jj] < 0 and diff1[jk] > 0:
                    temp_mini, tmin, _, _ = _seeklocales(abpsig, jj, jk)
                    if abs(tmin - temp_index) <= 2:
                        temp_min = temp_mini
                        tonsetp = tmin
                elif diff1[jj] > 0 and diff1[jk] < 0:
                    _, _, temp_maxi, tmax = _seeklocales(abpsig, jj, jk)
                    if abs(tmax - temp_index) <= 2:
                        temp_max = temp_maxi
                        tpeakp = tmax

                if (temp_max - temp_min) > 0.4 * abp_max_th and (temp_max - temp_min) < 2 * abp_max_th:
                    if tpeakp > tonsetp:
                        ttemp_min = abpsig[tonsetp]
                        ttonsetp = tonsetp
                        for ttk in range(tpeakp, tonsetp, -1):
                            if abpsig[ttk] < ttemp_min:
                                ttemp_min = abpsig[ttk]
                                ttonsetp = ttk
                        temp_min = ttemp_min
                        tonsetp = ttonsetp

                        if peak_index > 0:
                            if (tonsetp - peak_list[peak_index - 1]) < (3 * close_win):
                                temp_index = diff_index
                                abp_max_th = 2.5 * abp_max_lt
                                break

                            if (tpeakp - peak_list[peak_index - 1]) > step_win:
                                peak_index -= 1
                                onset_index -= 1
                                peak_list.pop()
                                onset_list.pop()
                                if dicro_index > 0:
                                    dicro_index -= 1
                                    dicro_list.pop()

                            if peak_index > 0:
                                peak_index += 1
                                onset_index += 1
                                peak_list.append(tpeakp)
                                onset_list.append(tonsetp)
                                tf = tonsetp - onset_list[peak_index - 2]
                                to_off = min(int(abpfreq / 20), max(1, int(0.1 * tf)))
                                to = peak_list[peak_index - 2] + to_off
                                te_off = int(abpfreq / 2)
                                tff_te = max(1, int(0.5 * tf))
                                if tff_te < te_off:
                                    te_off = tff_te
                                te = min(peak_list[peak_index - 2] + te_off, sig_len)
                                to = min(to, te - 1)
                                seg = diff1[to:te]
                                tff = _seekdicrotic(seg)
                                if tff == 0:
                                    tff = (te - to) // 3
                                dicro_index += 1
                                dicro_list.append(to + tff)
                                temp_index = temp_index + close_win
                                break

                        if peak_index == 0:
                            peak_index += 1
                            onset_index += 1
                            peak_list.append(tpeakp)
                            onset_list.append(tonsetp)
                            temp_index = temp_index + close_win
                            break

            temp_index += 1

        diff_index = temp_index + 1

    if len(peak_list) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # filtfilt is zero-phase; no group-delay correction like MATLAB filter()
    peakp = np.array(peak_list, dtype=np.int64)
    onsetp = np.array(onset_list, dtype=np.int64)
    return onsetp, peakp


def _pwd_beat_detector(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """PWD entry: returns (peaks, onsets) as 0-based indices."""
    onsets, peaks = _delineator(sig, fs)
    return peaks, onsets


def _merge_close_peaks(peaks: np.ndarray, ppg: np.ndarray, ppg_fs: float) -> np.ndarray:
    """If two peaks are closer than min_dist, keep the larger amplitude."""
    min_dist = max(1, int(ppg_fs * 60.0 / HR_MAX_BPM))
    if len(peaks) < 2:
        return np.sort(np.asarray(peaks, dtype=np.int64))
    keep = [peaks[0]]
    for i in range(1, len(peaks)):
        if peaks[i] - keep[-1] < min_dist:
            if ppg[peaks[i]] > ppg[keep[-1]]:
                keep[-1] = peaks[i]
        else:
            keep.append(peaks[i])
    return np.sort(np.asarray(keep, dtype=np.int64))


def hr_from_pwd(ppg: np.ndarray, ppg_fs: float) -> float:
    """HR (bpm) from one PPG window via PWD; returns np.nan on failure."""
    if len(ppg) < 50:
        return np.nan
    try:
        peaks, _ = _pwd_beat_detector(ppg, ppg_fs)
        peaks = _merge_close_peaks(peaks, ppg, ppg_fs)
        if len(peaks) < 2:
            return np.nan
        ibi_s = np.diff(peaks) / ppg_fs
        ibi_s = ibi_s[(ibi_s >= 60.0 / HR_MAX_BPM) & (ibi_s <= 60.0 / HR_MIN_BPM)]
        if len(ibi_s) == 0:
            return np.nan
        hr = 60.0 / np.median(ibi_s)
        if hr < HR_MIN_BPM or hr > HR_MAX_BPM:
            return np.nan
        return float(hr)
    except Exception:
        return np.nan
