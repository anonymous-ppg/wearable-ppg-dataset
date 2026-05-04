"""
qppgfast / adapted onset detector for HR from PPG.

Reference: Vest et al., Physiological Measurement, 2018.
"""
import numpy as np

HR_MIN_BPM = 30
HR_MAX_BPM = 200

EYE_CLS = 0.34
SLPW = 0.17
NDP = 2.5
TmDEF = 5
INVALID_DATA = -32768


def _slope_sum(data: np.ndarray, slp_window: int) -> np.ndarray:
    """
    Slope Sum Function: dy = max(0, data[t]-data[t-1]), summed in a sliding window.
    Returns lbuf with the same length as data.
    """
    dy = np.zeros(len(data), dtype=np.float64)
    dy[1:] = np.maximum(0, np.diff(data))
    kernel = np.ones(slp_window, dtype=np.float64)
    lbuf = np.convolve(dy, kernel, mode="full")[: len(data)]
    return lbuf


def _rescale_data(data: np.ndarray, fs: float) -> np.ndarray:
    """Rescale PPG to roughly ±2000."""
    data = np.asarray(data, dtype=np.float64)
    if data[0] <= INVALID_DATA + 10:
        data = data.copy()
        data[0] = np.mean(data)
    inv = np.where(data <= INVALID_DATA + 10)[0]
    for i in inv:
        if i > 0:
            data[i] = data[i - 1]

    if len(data) < 5 * 60 * fs:
        dmin, dmax = np.min(data), np.max(data)
        if dmax - dmin > 1e-10:
            data = (data - dmin) / (dmax - dmin) * 4000 - 2000
        else:
            data = np.zeros_like(data)
    else:
        chunk = int(5 * 60 * fs)
        max_vals = []
        min_vals = []
        for i in range(0, len(data), chunk):
            end = min(i + chunk, len(data))
            max_vals.append(np.max(data[i:end]))
            min_vals.append(np.min(data[i:end]))
        dmin = np.median(min_vals)
        dmax = np.median(max_vals)
        if dmax - dmin > 1e-10:
            data = (data - dmin) / (dmax - dmin) * 4000 - 2000
        else:
            data = np.zeros_like(data)
    return data


def _qppg_fast(data: np.ndarray, fs: float) -> np.ndarray:
    """
    qppg_fast core: returns onset sample indices (0-based).
    Uses a shorter learning period when the window is short.
    """
    data = _rescale_data(data, fs)
    sig_len = len(data)
    sps = fs

    eye_closing = max(1, int(sps * EYE_CLS))
    expect_period = int(sps * NDP)
    slp_window = max(1, int(sps * SLPW))

    lperiod = int(8 * sps)
    if sig_len < 12 * sps:
        lperiod = min(int(4 * sps), sig_len // 3)
    lperiod = max(eye_closing, lperiod)

    lbuf = _slope_sum(data, slp_window)

    from_idx = 0
    to_idx = sig_len - 1
    t1 = min(from_idx + lperiod, to_idx)

    valid_lbuf = lbuf[from_idx : t1 + 1]
    valid_lbuf = valid_lbuf[valid_lbuf > INVALID_DATA + 10]
    T0 = float(np.mean(valid_lbuf)) if len(valid_lbuf) > 0 else TmDEF
    if np.isnan(T0) or T0 <= 0:
        T0 = TmDEF
    Ta = 3 * T0
    Tm = TmDEF

    ppg_onsets = []
    beat_n = 0
    learning = True
    timer = 0
    t = from_idx

    while t <= to_idx:
        if learning:
            if t > from_idx + lperiod:
                learning = False
                T1 = T0
                t = from_idx
                continue
            else:
                T1 = 2 * T0

        temp = lbuf[t]

        if temp > T1:
            timer = 0
            maxd = temp
            mind = maxd
            tmax = t

            for tt in range(t + 1, min(t + eye_closing, to_idx + 1)):
                temp2 = lbuf[tt]
                if temp2 > maxd:
                    maxd = temp2
                    tmax = tt

            if maxd == temp:
                t += 1
                continue

            range_start = max(0, t - eye_closing // 2)
            for tt in range(tmax, range_start - 1, -1):
                temp2 = lbuf[tt]
                if temp2 < mind:
                    mind = temp2

            if maxd <= mind + 10:
                t += 1
                continue

            onset = (maxd - mind) / 100 + 2
            tpq = t - int(0.04 * fs)
            tpq = max(0, tpq)
            maxmin_23 = (maxd - mind) * 2.0 / 3

            tt = tmax
            for tt in range(tmax, range_start - 1, -1):
                temp2 = lbuf[tt]
                if temp2 < maxmin_23:
                    break

            tt_end = max(range_start, t - eye_closing // 2 + int(0.024 * fs))
            for tt in range(tt, tt_end - 1, -1):
                if tt - int(0.024 * fs) < 0:
                    break
                temp2 = lbuf[tt]
                temp3 = lbuf[tt - int(0.024 * fs)]
                if temp2 - temp3 < onset:
                    tpq = tt - int(0.016 * fs)
                    tpq = max(0, tpq)
                    break

            valley_v = int(round(tpq))
            lo = max(1, int(tpq - 0.20 * fs))
            hi = min(int(tpq + 0.05 * fs), sig_len - 2)
            for valley_i in range(lo, hi + 1):
                if valley_v <= 0:
                    break
                if (
                    data[valley_v] > data[valley_i]
                    and data[valley_i] <= data[valley_i - 1]
                    and data[valley_i] <= data[valley_i + 1]
                ):
                    valley_v = valley_i
            valley_v = max(0, min(valley_v, sig_len - 1))

            if valley_v <= 0:
                t += 1
                continue

            if not learning:
                if beat_n == 0:
                    if valley_v > 0:
                        ppg_onsets.append(valley_v)
                        beat_n += 1
                else:
                    if valley_v > ppg_onsets[beat_n - 1]:
                        ppg_onsets.append(valley_v)
                        beat_n += 1

            Ta = Ta + (maxd - Ta) / 10
            T1 = Ta / 3
            t = min(tpq + eye_closing, to_idx + 1)
        else:
            if not learning:
                timer += 1
                if timer > expect_period and Ta > Tm:
                    Ta = max(Tm, Ta - 1)
                    T1 = Ta / 3
            t += 1

    onsets = np.array(ppg_onsets, dtype=np.int64)
    if len(onsets) > 1:
        onsets = onsets[1:]  # Discard first beat: trace-back can stick to an early local minimum
    return onsets


def _pulse_peaks_from_onsets(sig: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """Peak = argmax between consecutive onsets; also one peak from last onset to end."""
    if len(onsets) == 0:
        return np.array([], dtype=np.int64)
    peaks = []
    for i in range(len(onsets) - 1):
        seg = sig[onsets[i] : onsets[i + 1]]
        if len(seg) > 0:
            idx = np.argmax(seg)
            peaks.append(onsets[i] + idx)
    last_seg = sig[onsets[-1] :]
    if len(last_seg) > 0:
        idx = np.argmax(last_seg)
        peaks.append(onsets[-1] + idx)
    return np.array(peaks, dtype=np.int64)


def _merge_close_peaks(
    peaks: np.ndarray, ppg: np.ndarray, ppg_fs: float
) -> np.ndarray:
    """If two peaks are closer than min_dist, keep the one with larger amplitude."""
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


def _qppgfast_beat_detector(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """qppgfast entry: returns (peaks, onsets) as 0-based sample indices."""
    onsets = _qppg_fast(sig, fs)
    onsets = np.unique(onsets)
    peaks = _pulse_peaks_from_onsets(sig, onsets)
    return peaks, onsets


def hr_from_qppgfast(ppg: np.ndarray, ppg_fs: float) -> float:
    """HR (bpm) from one PPG window via qppgfast; returns np.nan on failure."""
    if len(ppg) < 50:
        return np.nan
    try:
        peaks, _ = _qppgfast_beat_detector(ppg, ppg_fs)
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
