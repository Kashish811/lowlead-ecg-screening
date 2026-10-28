"""Create a small SYNTHETIC dataset in the exact PTB-XL file format.

Purpose: smoke-test / CI only. It lets anyone check the whole pipeline
(feature extraction -> training -> figures -> app -> report) in a few
minutes without downloading PTB-XL. Numbers produced from it are NOT
scientific results; the folder is tagged with SYNTHETIC_DATA.txt and the
report/slides generators print a warning banner when they detect it.

    python scripts/make_synthetic_ptbxl.py --n 2000
    (then set USE_SYNTHETIC_DATA = True in the notebook's settings cell)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

FS, N = C.FS, C.FS * C.SIGNAL_SECONDS

# approximate lead axes (3-D unit vectors: x=left, y=inferior, z=anterior)
_FRONTAL = {"I": 0, "II": 60, "III": 120, "AVR": -150, "AVL": -30, "AVF": 90}
_PRECORD = {"V1": 115, "V2": 95, "V3": 75, "V4": 55, "V5": 30, "V6": 0}


def lead_vectors():
    v = {}
    for l, a in _FRONTAL.items():
        r = np.deg2rad(a); v[l] = np.array([np.cos(r), np.sin(r), 0.0])
    for l, a in _PRECORD.items():
        r = np.deg2rad(a); v[l] = np.array([np.cos(r), 0.25, np.sin(r)])
    return np.stack([v[l] / np.linalg.norm(v[l]) for l in C.ALL_LEADS])


LV = lead_vectors()


def gauss(t, mu, sd):
    return np.exp(-0.5 * ((t - mu) / sd) ** 2)


def synth(labels: set, rng) -> np.ndarray:
    t = np.arange(N) / FS
    hr = rng.uniform(52, 105)
    rr = 60 / hr
    irregular = "CD" in labels and rng.random() < 0.3
    beats, x = [], rng.uniform(0.1, 0.4)
    while x < 10:
        beats.append(x); x += rr * (rng.uniform(0.75, 1.25) if irregular else rng.uniform(0.97, 1.03))
    axis = np.deg2rad(rng.normal(55, 20))
    vR = np.array([np.cos(axis), np.sin(axis), rng.normal(0.2, 0.15)]) * rng.uniform(0.9, 1.4)
    vQ = -0.15 * vR; vS = np.array([0.1, -0.2, -0.9]) * rng.uniform(0.4, 0.7)
    vT = 0.28 * vR + np.array([0, 0, 0.15]); vP = 0.12 * vR
    qrs_sd = 0.012 * (rng.uniform(1.9, 2.6) if "CD" in labels else rng.uniform(0.9, 1.15))
    st = np.zeros(12); t_scale = np.ones(12); q_extra = np.zeros(12)
    if "HYP" in labels:
        vR = vR * rng.uniform(1.5, 1.9); vS = vS * rng.uniform(1.5, 2.0)
    if "MI" in labels:
        terr = [6, 7, 8, 9] if rng.random() < 0.5 else [1, 2, 5]   # anterior or inferior
        q_extra[terr] = rng.uniform(0.15, 0.35); st[terr] += rng.uniform(0.05, 0.15)
        t_scale[terr] *= rng.uniform(-0.8, 0.3)
    if "STTC" in labels:
        idx = rng.choice(12, size=rng.integers(4, 9), replace=False)
        st[idx] -= rng.uniform(0.05, 0.12); t_scale[idx] *= rng.uniform(-0.7, 0.4)
    sig = np.zeros((N, 12))
    for b in beats:
        comp = {
            "P": (b - 0.16, 0.022, vP), "Q": (b - 0.025, 0.010, vQ),
            "R": (b, qrs_sd, vR), "S": (b + 0.03 + 1.2 * (qrs_sd - 0.012), 0.012, vS),
            "T": (b + 0.30, 0.045, vT),
        }
        for k, (mu, sd, vec) in comp.items():
            amp = LV @ vec
            if k == "T":
                amp = amp * t_scale
            sig += np.outer(gauss(t, mu, sd), amp)
        sig += np.outer(gauss(t, b - 0.03, 0.012), -q_extra)
        sig += np.outer(((t > b + 0.05) & (t < b + 0.22)).astype(float), st)
    sig += 0.02 * rng.standard_normal(sig.shape)
    sig += np.outer(0.05 * np.sin(2 * np.pi * rng.uniform(0.1, 0.3) * t), rng.uniform(-1, 1, 12))
    return sig


def main():
    import wfdb
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=C.DATA_DIR / "synthetic_ptbxl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "SYNTHETIC_DATA.txt").write_text("Synthetic smoke-test data in PTB-XL format. Not real ECGs.\n")

    code_for = {"NORM": "NORM", "MI": "IMI", "STTC": "NST_", "CD": "CLBBB", "HYP": "LVH"}
    stmts = pd.DataFrame({
        "description": ["normal ECG", "inferior MI", "non-specific ST changes",
                        "complete LBBB", "left ventricular hypertrophy", "sinus rhythm"],
        "diagnostic": [1, 1, 1, 1, 1, np.nan],
        "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP", np.nan],
    }, index=["NORM", "IMI", "NST_", "CLBBB", "LVH", "SR"])
    stmts.to_csv(a.out / "scp_statements.csv")

    rows = []
    for i in range(1, a.n + 1):
        r = rng.random()
        if r < 0.42:
            labels = {"NORM"}
        else:
            labels = {rng.choice(["MI", "STTC", "CD", "HYP"], p=[0.3, 0.3, 0.25, 0.15])}
            if rng.random() < 0.3:
                labels.add(rng.choice(["MI", "STTC", "CD", "HYP"]))
        sig = synth(labels, rng)
        sub = f"records100/{(i // 1000) * 1000:05d}"
        (a.out / sub).mkdir(parents=True, exist_ok=True)
        name = f"{i:05d}_lr"
        wfdb.wrsamp(name, fs=FS, units=["mV"] * 12, sig_name=list(C.ALL_LEADS),
                    p_signal=sig, fmt=["16"] * 12, adc_gain=[1000.0] * 12, baseline=[0] * 12,
                    write_dir=str(a.out / sub))
        codes = {code_for[l]: 100.0 for l in labels}
        codes["SR"] = 0.0
        rows.append({"ecg_id": i, "patient_id": float(rng.integers(1, int(a.n * 0.9))),
                     "age": float(rng.integers(25, 90)), "sex": int(rng.integers(0, 2)),
                     "height": np.nan, "weight": np.nan, "scp_codes": str(codes),
                     "strat_fold": int(rng.integers(1, 11)),
                     "filename_lr": f"{sub}/{name}", "filename_hr": f"{sub}/{name}"})
        if i % 500 == 0:
            print(f"  {i}/{a.n}")
    pd.DataFrame(rows).to_csv(a.out / "ptbxl_database.csv", index=False)
    print(f"Synthetic PTB-XL-format dataset written to {a.out}")


if __name__ == "__main__":
    main()
