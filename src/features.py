"""ECG feature extraction used by the Streamlit app.

This is the same code as section 6 of the notebook (the notebook checks that
both give identical numbers). Hand-crafted, clinically interpretable ECG features.

Features are computed *per lead*, and every column is named
``<LEAD>__<feature>``. A reduced-lead model is then just the 12-lead
feature table restricted to the columns of the leads it can see, so all
lead configurations share one identical, reproducible extraction step.

Per-lead feature groups (48 features/lead):
  * rhythm      - R-peak based heart rate and RR variability
  * statistics  - amplitude distribution of the filtered signal
  * spectral    - relative band powers and spectral entropy (Welch)
  * morphology  - median-beat P / Q / R / S / ST / T measurements, QRS width
  * template    - the median beat compressed to 14 bins (shape descriptor)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from scipy.stats import kurtosis, skew

from . import config as C

PRE, POST = 25, 45          # median-beat window: 250 ms before / 450 ms after R (at 100 Hz)
N_TEMPLATE_BINS = 14

_SOS_CACHE: dict = {}


def bandpass(x: np.ndarray, fs: int = C.FS) -> np.ndarray:
    key = (fs, C.BANDPASS)
    if key not in _SOS_CACHE:
        hi = min(C.BANDPASS[1], 0.45 * fs)
        _SOS_CACHE[key] = butter(3, [C.BANDPASS[0], hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(_SOS_CACHE[key], x, axis=0)


def detect_r_peaks(x: np.ndarray, fs: int = C.FS) -> np.ndarray:
    """Polarity-independent QRS detector (derivative energy + adaptive threshold)."""
    d = np.diff(x, prepend=x[0])
    env = np.convolve(d ** 2, np.ones(int(0.1 * fs)) / int(0.1 * fs), mode="same")
    if env.max() <= 0:
        return np.array([], dtype=int)
    thr = 0.3 * np.percentile(env, 99)
    cand, _ = find_peaks(env, height=thr, distance=int(0.3 * fs))
    # refine each candidate to the largest absolute deflection nearby
    w = int(0.08 * fs)
    centred = x - np.median(x)
    peaks = []
    for c in cand:
        lo, hi = max(0, c - w), min(len(x), c + w + 1)
        peaks.append(lo + int(np.argmax(np.abs(centred[lo:hi]))))
    peaks = np.unique(peaks)
    # drop near-duplicates created by refinement
    if len(peaks) > 1:
        keep = np.insert(np.diff(peaks) > int(0.25 * fs), 0, True)
        peaks = peaks[keep]
    return peaks


def _rhythm(peaks: np.ndarray, fs: int) -> dict:
    f = {"n_beats": float(len(peaks))}
    if len(peaks) >= 3:
        rr = np.diff(peaks) / fs * 1000.0  # ms
        drr = np.diff(rr)
        f.update(hr=60000.0 / np.mean(rr), rr_std=np.std(rr), rr_cv=np.std(rr) / np.mean(rr),
                 rmssd=np.sqrt(np.mean(drr ** 2)) if len(drr) else np.nan,
                 pnn50=np.mean(np.abs(drr) > 50) if len(drr) else np.nan,
                 rr_range=(rr.max() - rr.min()))
    else:
        f.update(hr=np.nan, rr_std=np.nan, rr_cv=np.nan, rmssd=np.nan, pnn50=np.nan, rr_range=np.nan)
    return f


def _stats(x: np.ndarray) -> dict:
    return {
        "std": np.std(x), "ptp": np.ptp(x), "skew": float(skew(x)), "kurt": float(kurtosis(x)),
        "p05": np.percentile(x, 5), "p95": np.percentile(x, 95),
        "abs_mean": np.mean(np.abs(x)),
        "zcr": np.mean(np.abs(np.diff(np.sign(x - np.mean(x)))) > 0),
    }


def _spectral(x: np.ndarray, fs: int) -> dict:
    f, p = welch(x, fs=fs, nperseg=min(256, len(x)))
    total = p[(f >= 0.5) & (f <= 45)].sum() + 1e-12

    def band(a, b):
        return p[(f >= a) & (f < b)].sum() / total

    pn = p / (p.sum() + 1e-12)
    return {
        "bp_low": band(0.5, 5), "bp_mid": band(5, 15), "bp_high": band(15, 45),
        "spec_entropy": float(-(pn * np.log(pn + 1e-12)).sum()),
        "peak_freq": float(f[np.argmax(p)]),
    }


MORPH_KEYS = ["p_amp", "q_amp", "r_amp", "s_amp", "qrs_ptp", "qrs_area", "qrs_width",
              "st_level", "st_slope", "t_amp", "t_max", "t_min", "t_area", "beat_corr"]


def _morphology(x: np.ndarray, peaks: np.ndarray, fs: int) -> dict:
    nan = {k: np.nan for k in MORPH_KEYS}
    nan.update({f"tmpl_{i}": np.nan for i in range(N_TEMPLATE_BINS)})
    peaks = peaks[(peaks - PRE >= 0) & (peaks + POST <= len(x))]
    if len(peaks) < 2:
        return nan
    beats = np.stack([x[p - PRE:p + POST] for p in peaks])
    t = np.median(beats, axis=0)
    t = t - np.median(t[PRE - 12:PRE - 5])          # PR-segment baseline
    r = PRE
    qrs = t[r - 8:r + 9]
    dt = np.abs(np.diff(t[r - 15:r + 16]))
    above = np.where(dt > 0.15 * dt.max())[0] if dt.max() > 0 else np.array([])
    width_ms = (above[-1] - above[0] + 1) / fs * 1000 if len(above) else np.nan
    tw = t[r + 15:r + 40]
    corr = [np.corrcoef(b, t)[0, 1] for b in beats if np.std(b) > 0]
    f = {
        "p_amp": t[r - 25:r - 10][np.argmax(np.abs(t[r - 25:r - 10]))],
        "q_amp": t[r - 8:r].min(),
        "r_amp": t[r],
        "s_amp": t[r:r + 8].min(),
        "qrs_ptp": np.ptp(qrs),
        "qrs_area": np.abs(t[r - 6:r + 7]).sum() / fs,
        "qrs_width": width_ms,
        "st_level": t[r + 9:r + 13].mean(),
        "st_slope": t[r + 14] - t[r + 8],
        "t_amp": tw[np.argmax(np.abs(tw))],
        "t_max": tw.max(), "t_min": tw.min(),
        "t_area": tw.sum() / fs,
        "beat_corr": float(np.nanmean(corr)) if corr else np.nan,
    }
    bins = np.array_split(t, N_TEMPLATE_BINS)
    f.update({f"tmpl_{i}": float(b.mean()) for i, b in enumerate(bins)})
    return f


def lead_features(x_raw: np.ndarray, fs: int = C.FS) -> dict:
    x = bandpass(x_raw, fs)
    peaks = detect_r_peaks(x, fs)
    f = {}
    f.update(_rhythm(peaks, fs))
    f.update(_stats(x))
    f.update(_spectral(x, fs))
    f.update(_morphology(x, peaks, fs))
    return f


def record_features(sig: np.ndarray, leads: list[str] = C.ALL_LEADS, fs: int = C.FS) -> dict:
    """Features for one ECG. `sig` has shape (n_samples, len(leads))."""
    out = {}
    for j, lead in enumerate(leads):
        for k, v in lead_features(sig[:, j], fs).items():
            out[f"{lead}__{k}"] = float(v) if v is not None else np.nan
    return out
