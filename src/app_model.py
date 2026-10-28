"""Loads the calibrated LightGBM models saved by the notebook (section 13).

The Streamlit app must load models trained on a different machine (your
laptop -> Streamlit Cloud). Pickles break when library versions differ, so
each model is exported as plain JSON:

  * the LightGBM booster as its native text format (stable across versions)
  * the Platt-scaling coefficients  p_cal = sigmoid(a * logit(p_raw) + b)
  * the F1-optimal decision threshold

LightGBM's own `pred_contrib=True` gives exact TreeSHAP values, so the app
gets per-ECG explanations without needing the `shap` package.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class AppModel:
    """Lightweight predictor used by the Streamlit app."""

    def __init__(self, path: Path):
        import lightgbm as lgb
        d = json.loads(Path(path).read_text())
        self.lead_set, self.leads = d["lead_set"], d["leads"]
        self.feature_columns, self.classes = d["feature_columns"], d["classes"]
        self.boosters, self.platt, self.thresholds = {}, {}, {}
        for c, pc in d["per_class"].items():
            self.boosters[c] = lgb.Booster(model_str=pc["booster"])
            self.platt[c] = (pc["platt_a"], pc["platt_b"])
            self.thresholds[c] = pc["threshold"]

    def _X(self, feats: pd.DataFrame) -> np.ndarray:
        return feats.reindex(columns=self.feature_columns).to_numpy(dtype=float)

    def predict_proba(self, feats: pd.DataFrame) -> dict:
        X = self._X(feats)
        out = {}
        for c in self.classes:
            raw = self.boosters[c].predict(X)
            a, b = self.platt[c]
            out[c] = float(1 / (1 + np.exp(-(a * _logit(raw)[0] + b))))
        return out

    def explain(self, feats: pd.DataFrame, cls: str, top: int = 10) -> pd.DataFrame:
        """Exact TreeSHAP contributions (log-odds of the raw model) for one ECG."""
        X = self._X(feats)
        contrib = self.boosters[cls].predict(X, pred_contrib=True)[0][:-1]   # last = bias
        df = pd.DataFrame({"feature": self.feature_columns, "shap": contrib, "value": X[0]})
        return df.reindex(df["shap"].abs().sort_values(ascending=False).index).head(top)


PRETTY = {
    "hr": "heart rate", "rr_std": "RR std", "rr_cv": "RR variability", "rmssd": "RMSSD",
    "pnn50": "pNN50", "rr_range": "RR range", "n_beats": "beat count",
    "std": "signal std", "ptp": "peak-to-peak", "skew": "skewness", "kurt": "kurtosis",
    "p05": "5th pct amplitude", "p95": "95th pct amplitude", "abs_mean": "mean |amplitude|",
    "zcr": "zero-crossing rate", "bp_low": "low-band power", "bp_mid": "mid-band power",
    "bp_high": "high-band power", "spec_entropy": "spectral entropy", "peak_freq": "peak frequency",
    "p_amp": "P amplitude", "q_amp": "Q depth", "r_amp": "R amplitude", "s_amp": "S depth",
    "qrs_ptp": "QRS amplitude", "qrs_area": "QRS area", "qrs_width": "QRS width",
    "st_level": "ST level", "st_slope": "ST slope", "t_amp": "T amplitude", "t_max": "T max",
    "t_min": "T min", "t_area": "T area", "beat_corr": "beat consistency",
}


def pretty(col: str) -> str:
    lead, f = col.split("__", 1)
    lead = lead.replace("AV", "aV")
    if f.startswith("tmpl_"):
        return f"{lead}: beat shape bin {f.split('_')[1]}"
    return f"{lead}: {PRETTY.get(f, f)}"
