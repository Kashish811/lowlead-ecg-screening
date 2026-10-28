"""Unit tests for the helpers the Streamlit app uses.   Run:  python -m pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.app_model import AppModel, pretty
from src.features import bandpass, detect_r_peaks, record_features
from src.noise import add_wearable_noise


def fake_ecg(heart_rate=60, n=1000):
    """A simple synthetic heartbeat train (R and T waves) at 100 Hz."""
    t = np.arange(n) / C.FS
    x = np.zeros(n)
    for beat in np.arange(0.3, 10, 60 / heart_rate):
        x += np.exp(-0.5 * ((t - beat) / 0.012) ** 2) + 0.25 * np.exp(-0.5 * ((t - beat - 0.3) / 0.045) ** 2)
    return x


def test_r_peak_detection_counts_beats():
    assert abs(len(detect_r_peaks(bandpass(fake_ecg(75)))) - 12) <= 1


def test_heart_rate_feature():
    features = record_features(np.tile(fake_ecg(80)[:, None], (1, 12)))
    assert abs(features["II__hr"] - 80) < 3


def test_48_features_per_lead():
    features = record_features(np.tile(fake_ecg()[:, None], (1, 12)))
    assert len(features) == 48 * 12


@pytest.mark.parametrize("snr", [24, 12, 6])
def test_noise_hits_target_snr(snr):
    clean = np.tile(fake_ecg()[:, None], (1, 3)).astype(np.float32)
    noisy = add_wearable_noise(clean, snr, seed=1)
    signal = clean[:, 0] - clean[:, 0].mean()
    measured = 10 * np.log10(np.mean(signal ** 2) / np.mean((noisy[:, 0] - clean[:, 0]) ** 2))
    assert abs(measured - snr) < 0.5


def test_noise_is_reproducible():
    clean = np.tile(fake_ecg()[:, None], (1, 2)).astype(np.float32)
    assert np.allclose(add_wearable_noise(clean, 12, 5), add_wearable_noise(clean, 12, 5))


def test_app_model_applies_platt_calibration(tmp_path):
    """A saved model must give sigmoid(a * logit(raw) + b), exactly as in the notebook."""
    from lightgbm import LGBMClassifier
    rng = np.random.default_rng(0)
    cols = [f"I__f{i}" for i in range(4)]
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=cols)
    y = (X.iloc[:, 0] > 0).astype(int)
    booster = LGBMClassifier(n_estimators=20, verbose=-1).fit(X, y)
    a, b = 0.8, -0.2
    saved = {"lead_set": "1-lead", "leads": ["I"], "feature_columns": cols, "classes": C.CLASSES,
             "per_class": {c: {"booster": booster.booster_.model_to_string(), "platt_a": a, "platt_b": b,
                               "threshold": 0.5} for c in C.CLASSES}}
    (tmp_path / "m.json").write_text(json.dumps(saved))
    raw = np.clip(booster.predict_proba(X.iloc[[3]])[0, 1], 1e-6, 1 - 1e-6)
    expected = 1 / (1 + np.exp(-(a * np.log(raw / (1 - raw)) + b)))
    assert abs(AppModel(tmp_path / "m.json").predict_proba(X.iloc[[3]])["MI"] - expected) < 1e-9


def test_pretty_names():
    assert pretty("AVF__q_amp") == "aVF: Q depth"
