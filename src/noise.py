"""Simulated wearable-device noise.

Consumer wearables (watches, patches, handheld electrodes) suffer from three
dominant artefacts, which we synthesise and mix at a target SNR:

1. Baseline wander   - slow drift from breathing and electrode movement (0.05-0.5 Hz)
2. Muscle (EMG) noise - broadband high-frequency noise from arm / chest muscles
3. Motion artefacts  - short, large transients when the wearer moves

The mixture is scaled per lead so that 10*log10(P_signal / P_noise) = SNR dB.
Noise is seeded by record id, so results are fully reproducible.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import config as C

# Relative power share of each artefact in the mixture
MIX = {"baseline": 0.45, "emg": 0.35, "motion": 0.20}


def _baseline(n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / fs
    x = np.zeros(n)
    for _ in range(3):
        f = rng.uniform(0.05, 0.5)
        x += rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return x


def _emg(n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    sos = butter(4, [15, 0.45 * fs], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, rng.standard_normal(n + 200))[100:-100]


def _motion(n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n)
    t = np.arange(n)
    for _ in range(rng.integers(1, 4)):
        c = rng.integers(0, n)
        width = rng.uniform(0.1, 0.6) * fs
        x += rng.choice([-1, 1]) * rng.uniform(0.5, 1.0) * np.exp(-0.5 * ((t - c) / width) ** 2)
    return x


def _unit_power(x: np.ndarray) -> np.ndarray:
    p = np.mean(x ** 2)
    return x / np.sqrt(p) if p > 0 else x


def add_wearable_noise(sig: np.ndarray, snr_db: float, seed: int, fs: int = C.FS) -> np.ndarray:
    """Return a noisy copy of `sig` (shape (n_samples, n_leads))."""
    rng = np.random.default_rng(seed)
    n, n_leads = sig.shape
    out = sig.astype(np.float64).copy()
    for j in range(n_leads):
        noise = (np.sqrt(MIX["baseline"]) * _unit_power(_baseline(n, fs, rng))
                 + np.sqrt(MIX["emg"]) * _unit_power(_emg(n, fs, rng))
                 + np.sqrt(MIX["motion"]) * _unit_power(_motion(n, fs, rng)))
        noise = _unit_power(noise)
        centred = sig[:, j] - np.mean(sig[:, j])
        p_sig = np.mean(centred ** 2)
        if p_sig <= 0:
            continue
        p_noise = p_sig / (10 ** (snr_db / 10))
        out[:, j] += noise * np.sqrt(p_noise)
    return out.astype(np.float32)


def noise_seed(ecg_id: int, snr_db: float) -> int:
    return int(ecg_id) * 1000 + int(snr_db)
