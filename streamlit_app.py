"""Low-Lead ECG Screening - interactive demo.

    python -m streamlit run streamlit_app.py

Pick a heart recording, choose a device (hospital 12-lead down to a 1-lead smartwatch),
add wearable noise, and see what the model concludes and why - in plain language.
A screening-support prototype for education, not a diagnostic tool.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import resample

from src import config as C
from src.app_model import AppModel
from src.features import record_features
from src.noise import add_wearable_noise

st.set_page_config(page_title="Low-Lead ECG Screening", page_icon="🫀", layout="wide",
                   initial_sidebar_state="expanded")

# =====================================================================================
# Design tokens
# =====================================================================================
# Theme: the bedside heart monitor. Deep teal "monitor glass", glowing phosphor traces,
# and one colour per signal, as on a real ward monitor.
BG = "#061A20"           # page background
PANEL = "#0D252C"        # cards and chart panels
INK = "#E8F3F1"          # main text
MUTED = "#8FB3AF"        # secondary text
LINE = "#1D3F46"         # borders and chart grid
SURFACE = "#0A2026"
PAPER = "#03141A"        # ECG monitor screen
GRID_MAJOR = "#14404A"
GRID_MINOR = "#0A2A31"
GREEN = "#3EE89A"        # ECG phosphor green
RED = "#FF5D6C"          # alarm red
AMBER = "#FFB547"
CYAN = "#4CC9F0"
TEAL, CRIMSON = GREEN, RED
FONT = "Figtree, system-ui, sans-serif"
DISPLAY = "'Chakra Petch', 'Figtree', sans-serif"

CLASS_COLORS = {"NORM": GREEN, "MI": RED, "STTC": AMBER, "CD": CYAN, "HYP": "#C792EA"}
DEVICE_COLORS = {"12-lead": "#E8F3F1", "6-lead": CYAN, "3-lead": GREEN, "1-lead": AMBER}
MODEL_COLORS = {"LogReg": "#6F8F8C", "RandomForest": CYAN, "LightGBM": GREEN}
HEAT = [[0, "#7A2232"], [0.5, "#7A5A1E"], [1, "#147A55"]]     # dark red -> amber -> green, white text stays legible

# =====================================================================================
# Plain-language vocabulary (used everywhere so a newcomer can follow)
# =====================================================================================
SHORT = {"NORM": "Normal", "MI": "Heart attack signs", "STTC": "ST/T changes",
         "CD": "Conduction delay", "HYP": "Thickened muscle"}
LONG = {
    "NORM": "Normal ECG",
    "MI": "Heart attack signs (MI)",
    "STTC": "ST/T changes (STTC)",
    "CD": "Conduction delay (CD)",
    "HYP": "Thickened heart muscle (HYP)",
}
MEANING = {
    "NORM": "The heart's electrical pattern looks healthy.",
    "MI": "Myocardial infarction: a heart attack, now or in the past. Part of the muscle was starved of blood "
          "and leaves tell-tale marks, such as deep Q dips or a raised ST segment.",
    "STTC": "Changes in the ST segment and T wave, the part of each beat where the heart recovers. "
            "Often linked to reduced blood flow or strain.",
    "CD": "The electrical signal travels through the heart more slowly or by a detour, for example a "
          "bundle branch block. The main spike (QRS) becomes wider.",
    "HYP": "Hypertrophy: the heart muscle has thickened, often from long-term high blood pressure. "
           "The main spikes become unusually tall.",
}
DEVICE_NAME = {"12-lead": "Hospital ECG (12 leads)", "6-lead": "Handheld device (6 leads)",
               "3-lead": "Chest patch (3 leads)", "1-lead": "Smartwatch (1 lead)"}
DEVICE_NOTE = {
    "12-lead": "The full clinical recording: 10 electrodes on chest, arms and legs.",
    "6-lead": "Only the six limb leads (I, II, III, aVR, aVL, aVF), as recorded by handheld devices you hold "
              "with both hands and rest on your leg.",
    "3-lead": "Leads I, II and one chest lead (V2), like an adhesive chest patch.",
    "1-lead": "Lead I only: the signal between your two arms, which is what a smartwatch records.",
}
LEAD_LABEL = lambda l: l.replace("AV", "aV")

GLOSSARY = {
    "ECG (electrocardiogram)": "A recording of the heart's electrical activity, drawn as a wavy line over time. "
                               "Each heartbeat makes the same pattern of waves.",
    "Lead": "The heart's signal viewed from one direction, measured between particular electrodes. More leads "
            "means more viewing angles, like more cameras around the heart.",
    "12-lead ECG": "The standard hospital test. 10 electrodes give 12 views: 6 from the limbs (I, II, III, aVR, "
                   "aVL, aVF) and 6 across the chest (V1-V6).",
    "Lead I": "The signal between the left and right arm. It is the only lead a smartwatch can record.",
    "P wave": "The small first bump of a heartbeat: the upper chambers (atria) contracting.",
    "QRS complex": "The tall, sharp main spike: the main pumping chambers (ventricles) contracting. "
                   "A wide QRS suggests a conduction delay; a very tall one can mean thickened muscle.",
    "Q wave": "The small dip just before the main spike. Deep Q waves can be a mark of an old heart attack.",
    "ST segment": "The flat stretch right after the main spike. If it is raised or lowered, the heart muscle "
                  "may be short of blood.",
    "T wave": "The rounded bump after the spike, when the ventricles recover. Flat or upside-down T waves can "
              "signal strain or reduced blood flow.",
    "RR interval": "The time between two heartbeats (from one main spike to the next). It gives the heart rate "
                   "and shows how regular the rhythm is.",
    "Heart rate (bpm)": "Beats per minute. A typical resting value is about 60-100.",
    "mV (millivolt)": "The unit of ECG voltage. A typical main spike is about 1 mV tall.",
    "Myocardial infarction (MI)": MEANING["MI"],
    "ST/T change (STTC)": MEANING["STTC"],
    "Conduction disturbance (CD)": MEANING["CD"],
    "Hypertrophy (HYP)": MEANING["HYP"],
    "Probability": "How likely the model thinks a condition is. The probabilities here are calibrated: 30% means "
                   "that among many similar ECGs, about 30 in 100 really had the condition.",
    "Flag line (threshold)": "The probability above which a condition is flagged for review. Each condition has "
                             "its own line, chosen on data the model never saw while learning.",
    "Screening": "Spotting people who should get a proper check. A screening flag is a prompt to see a doctor, "
                 "not a diagnosis.",
    "Noise / SNR": "Wearables pick up interference from movement, muscles and breathing. SNR (signal-to-noise "
                   "ratio, in dB) measures how clean the signal is: 24 dB is clean, 6 dB is very noisy.",
    "SHAP explanation": "A way to see why the model decided something: it splits a prediction into the push each "
                        "measurement gave towards or away from a condition.",
    "AUROC": "How well a model ranks people with a condition above people without it, from 0.5 (coin toss) to "
             "1.0 (perfect). 0.9 is excellent.",
    "Sensitivity": "Of the people who really have the condition, the share the model flags.",
    "Specificity": "Of the people who do not have the condition, the share the model correctly leaves unflagged.",
    "Retention": "How much of the 12-lead accuracy a smaller device keeps, as a percentage.",
    "PTB-XL": "The public dataset used here: 21,799 hospital ECGs from 18,869 patients, labelled by cardiologists.",
    "LightGBM": "The machine-learning model used in this app: many small decision trees, each correcting the "
                "previous ones.",
}

FEATURE_PLAIN = {
    "hr": ("heart rate", "bpm"), "rr_std": ("variation in time between beats", "ms"),
    "rr_cv": ("rhythm irregularity", ""), "rmssd": ("beat-to-beat timing change", "ms"),
    "pnn50": ("share of big beat-to-beat changes", ""), "rr_range": ("spread of beat timings", "ms"),
    "n_beats": ("number of beats", ""), "std": ("overall signal strength", "mV"),
    "ptp": ("total height of the signal", "mV"), "skew": ("lopsidedness of the signal", ""),
    "kurt": ("spikiness of the signal", ""), "p05": ("lowest typical voltage", "mV"),
    "p95": ("highest typical voltage", "mV"), "abs_mean": ("average voltage size", "mV"),
    "zcr": ("how often the signal crosses its middle", ""), "bp_low": ("share of slow wave energy", ""),
    "bp_mid": ("share of medium-speed wave energy", ""), "bp_high": ("share of fast wave energy", ""),
    "spec_entropy": ("signal complexity", ""), "peak_freq": ("dominant frequency", "Hz"),
    "p_amp": ("P wave height", "mV"), "q_amp": ("Q dip depth", "mV"), "r_amp": ("main spike height", "mV"),
    "s_amp": ("S dip depth", "mV"), "qrs_ptp": ("main spike size (QRS)", "mV"),
    "qrs_area": ("main spike area (QRS)", "mV·s"), "qrs_width": ("main spike width (QRS)", "ms"),
    "st_level": ("ST segment level", "mV"), "st_slope": ("ST segment slope", "mV"),
    "t_amp": ("T wave height", "mV"), "t_max": ("T wave top", "mV"), "t_min": ("T wave bottom", "mV"),
    "t_area": ("T wave area", "mV·s"), "beat_corr": ("how alike the beats are", ""),
}


def shape_region(i: int) -> str:
    """Name the part of the heartbeat covered by template bin i (each bin = 50 ms, from 250 ms before the spike)."""
    start = -250 + 50 * i
    if start < -100:
        part = "P wave"
    elif start < -50:
        part = "before the main spike"
    elif start < 50:
        part = "main spike (QRS)"
    elif start < 150:
        part = "ST segment"
    elif start < 400:
        part = "T wave"
    else:
        part = "end of the beat"
    return f"beat shape at {start:+d} to {start + 50:+d} ms ({part})"


def describe_feature(col: str) -> tuple[str, str]:
    lead, f = col.split("__", 1)
    if f.startswith("tmpl_"):
        return f"Lead {LEAD_LABEL(lead)}: {shape_region(int(f.split('_')[1]))}", "mV"
    name, unit = FEATURE_PLAIN.get(f, (f, ""))
    return f"Lead {LEAD_LABEL(lead)}: {name}", unit


def fmt_value(v: float, unit: str) -> str:
    if v is None or np.isnan(v):
        return "not measurable"
    if unit in ("ms", "bpm"):
        return f"{v:.0f} {unit}"
    if unit == "Hz":
        return f"{v:.1f} Hz"
    if unit:
        return f"{v:.2f} {unit}"
    return f"{v:.2f}"


# =====================================================================================
# Styles
# =====================================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=Figtree:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{ font-family: {FONT}; }}

/* page: monitor glass with a faint grid and a soft glow */
.stApp {{
  background:
    radial-gradient(1100px 480px at 78% -12%, rgba(62,232,154,.13), transparent 60%),
    radial-gradient(800px 420px at -5% 10%, rgba(76,201,240,.09), transparent 60%),
    linear-gradient(180deg, #071E25 0%, {BG} 45%, #04141A 100%);
}}
.stApp::before {{
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    linear-gradient(rgba(62,232,154,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(62,232,154,.035) 1px, transparent 1px);
  background-size: 28px 28px;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px; position: relative; z-index: 1; }}

h1, h2, h3, h4 {{ font-family: {DISPLAY} !important; color: {INK}; letter-spacing: .01em; }}
h2 {{ font-weight: 600 !important; font-size: 1.55rem !important; }}
h3 {{ font-weight: 600 !important; font-size: 1.3rem !important; }}
h4 {{ font-weight: 600 !important; font-size: 1.15rem !important; margin-top: 1.4rem; color: {INK}; }}
p, li {{ color: {INK}; line-height: 1.6; }}
a {{ color: {CYAN}; }}

/* sidebar: the control panel */
section[data-testid="stSidebar"] {{ background: rgba(3,17,22,.94); border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] h3 {{ font-size: 1.02rem !important; color: {GREEN}; margin-top: .5rem; }}
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {{ color: {MUTED}; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.8rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-family: {DISPLAY}; font-size: 1.02rem; font-weight: 600; color: {MUTED};
                               padding: .65rem 0; }}
.stTabs [aria-selected="true"] {{ color: {GREEN} !important; }}

/* hero: a ward monitor with a live trace */
.hero {{ position: relative; overflow: hidden; border: 1px solid {LINE}; border-radius: 16px;
         background: linear-gradient(180deg, rgba(13,37,44,.92), rgba(3,20,26,.92));
         padding: 1.5rem 1.7rem 1.35rem; margin-bottom: 1rem; }}
.hero h1 {{ font-size: 2.7rem !important; font-weight: 700 !important; line-height: 1.05 !important;
            margin: 0 0 .45rem 0 !important; padding: 0 !important; }}
.hero p {{ color: {MUTED}; font-size: 1.08rem; max-width: 64ch; margin: 0; position: relative; z-index: 2; }}
.hero .trace {{ position: absolute; left: 0; right: 0; bottom: 6px; height: 92px; width: 100%; z-index: 1;
                opacity: .9; }}
.hero .trace path {{ fill: none; stroke: {GREEN}; stroke-width: 2.2; stroke-linejoin: round;
                     filter: drop-shadow(0 0 6px rgba(62,232,154,.75));
                     stroke-dasharray: 2600; stroke-dashoffset: 2600; animation: sweep 5.5s linear infinite; }}
.hero .trace path.ghost {{ stroke: rgba(62,232,154,.13); filter: none; animation: none; stroke-dashoffset: 0; }}
.hero .content {{ position: relative; z-index: 2; padding-bottom: 70px; }}
@keyframes sweep {{ 0% {{ stroke-dashoffset: 2600; }} 85% {{ stroke-dashoffset: 0; opacity: 1; }}
                    100% {{ stroke-dashoffset: 0; opacity: 0; }} }}

/* vital-signs strip */
.vitals {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; margin: .2rem 0 1.1rem 0; }}
.vital {{ border: 1px solid {LINE}; border-radius: 12px; padding: .7rem 1rem .8rem; background: rgba(3,20,26,.75); }}
.vital .k {{ color: {MUTED}; font-size: .86rem; }}
.vital .v {{ font-family: {DISPLAY}; font-size: 1.75rem; font-weight: 600; line-height: 1.15; margin-top: .15rem;
             text-shadow: 0 0 14px currentColor; }}
.vital .u {{ font-size: .95rem; color: {MUTED}; text-shadow: none; margin-left: .25rem; }}
.vital .beat {{ display: inline-block; animation: pulse var(--beat, 1s) ease-out infinite; }}
@keyframes pulse {{ 0% {{ transform: scale(1.25); }} 30% {{ transform: scale(1); }} 100% {{ transform: scale(1); }} }}

.notice {{ border: 1px solid rgba(255,181,71,.35); background: rgba(255,181,71,.07); padding: .6rem .9rem;
           border-radius: 10px; color: #FFD9A0; font-size: .93rem; margin-bottom: .8rem; }}
.notice b {{ color: #FFE3B8; }}
.notice.synthetic {{ border-color: rgba(255,93,108,.45); background: rgba(255,93,108,.08); color: #FFC2C8; }}

/* result cards */
.verdict {{ display: flex; gap: 1rem; align-items: stretch; flex-wrap: wrap; margin: .3rem 0 1.1rem 0; }}
.verdict .main {{ flex: 3 1 420px; border-radius: 14px; padding: 1.1rem 1.3rem; background: {PANEL};
                  border: 1px solid {LINE}; border-left: 4px solid var(--accent); 
                  box-shadow: inset 0 0 40px -20px var(--accent); }}
.verdict .main .lbl, .verdict .side .lbl {{ font-size: .88rem; color: {MUTED}; margin-bottom: .3rem; }}
.verdict .main .big {{ font-family: {DISPLAY}; font-size: 1.65rem; font-weight: 600; line-height: 1.2;
                       color: var(--accent); }}
.verdict .main .sub, .verdict .side .sub {{ font-size: .95rem; color: {INK}; opacity: .85; margin-top: .35rem; }}
.verdict .side {{ flex: 2 1 260px; border: 1px solid {LINE}; border-radius: 14px; padding: 1.1rem 1.3rem;
                  background: rgba(3,20,26,.7); }}
.verdict .side .big {{ font-family: {DISPLAY}; font-size: 1.3rem; font-weight: 600; color: {INK}; }}

.chip {{ display: inline-block; padding: .12rem .65rem; border-radius: 999px; font-weight: 700; font-size: .84rem;
         color: #04141A; margin: 0 .3rem .3rem 0; }}
.explain {{ background: rgba(76,201,240,.06); border: 1px solid rgba(76,201,240,.18); border-radius: 10px;
            padding: .75rem 1rem; color: {INK}; font-size: .95rem; margin: .3rem 0 1rem 0; }}
.explain b {{ color: {CYAN}; font-weight: 600; }}

.reason {{ display: grid; grid-template-columns: 1fr auto; gap: .2rem 1rem; padding: .7rem 0;
           border-bottom: 1px solid {LINE}; }}
.reason .what {{ font-weight: 600; color: {INK}; }}
.reason .val {{ color: {MUTED}; font-size: .92rem; }}
.reason .eff {{ font-family: {DISPLAY}; font-weight: 600; text-align: right; white-space: nowrap; }}
.reason .why {{ grid-column: 1 / -1; color: {MUTED}; font-size: .9rem; }}

.term {{ padding: .8rem 0; border-bottom: 1px solid {LINE}; }}
.term .t {{ font-family: {DISPLAY}; font-weight: 600; color: {GREEN}; }}
.term .d {{ color: {INK}; margin-top: .15rem; opacity: .9; }}

.stat {{ border: 1px solid {LINE}; border-radius: 12px; padding: .8rem 1rem; background: rgba(3,20,26,.7); }}
.stat .n {{ font-family: {DISPLAY}; font-size: 2.1rem; font-weight: 600; color: {GREEN}; line-height: 1.1;
            text-shadow: 0 0 16px rgba(62,232,154,.45); }}
.stat .l {{ color: {MUTED}; font-size: .92rem; }}

.lede {{ color: {MUTED}; font-size: 1.05rem; max-width: 70ch; margin: 0 0 1rem 0; }}
div[data-testid="stExpander"] details {{ border: 1px solid {LINE}; border-radius: 12px; background: rgba(13,37,44,.6); }}
div[data-testid="stPlotlyChart"] {{ border: 1px solid {LINE}; border-radius: 14px; background: rgba(13,37,44,.55);
                                    padding: .4rem .3rem .2rem; }}
*:focus-visible {{ outline: 2px solid {GREEN} !important; outline-offset: 2px; }}

@media (prefers-reduced-motion: reduce) {{
  .hero .trace path {{ animation: none; stroke-dashoffset: 0; }}
  .vital .beat {{ animation: none; }}
}}
@media (max-width: 760px) {{
  .hero h1 {{ font-size: 2rem !important; }}
  .vitals {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .verdict .main .big {{ font-size: 1.35rem; }}
}}
</style>
""", unsafe_allow_html=True)

PLOT_CONFIG = {"displaylogo": False, "scrollZoom": True,
               "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
               "toImageButtonOptions": {"format": "png", "scale": 2}}


def base_layout(fig, height, **kw):
    """Shared chart styling; any keyword passed in overrides the default."""
    layout = dict(height=height, font=dict(family=FONT, size=13, color=INK), paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=20, t=30, b=40),
                  hoverlabel=dict(bgcolor="#0B2229", bordercolor=GREEN, font=dict(family=FONT, size=13, color=INK)),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=12)))
    layout.update(kw)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=INK),
                     title_font=dict(color=MUTED))
    return fig


def explain_box(html: str):
    if st.session_state.get("plain", True):
        st.markdown(f'<div class="explain">{html}</div>', unsafe_allow_html=True)


# =====================================================================================
# Data loading
# =====================================================================================
@st.cache_resource
def load_models():
    models = {ls: AppModel(C.MODEL_DIR / f"{ls}.json") for ls in C.LEAD_SET_ORDER
              if (C.MODEL_DIR / f"{ls}.json").exists()}
    manifest_path = C.MODEL_DIR / "manifest.json"
    return models, (json.loads(manifest_path.read_text()) if manifest_path.exists() else {})


@st.cache_data
def load_samples():
    out = {}
    for f in sorted(C.SAMPLE_DIR.glob("*.npz")):
        d = np.load(f)
        labels = [c for c, v in zip(C.CLASSES, d["labels"]) if v]
        name = f"ECG {int(d['ecg_id'])}: {' + '.join(SHORT[c] for c in labels)}"
        out[name] = {"signal": d["signal"], "labels": labels, "age": float(d["age"]), "sex": int(d["sex"])}
    return out


@st.cache_data
def load_results():
    s, m = C.RESULTS_DIR / "summary.json", C.RESULTS_DIR / "metrics.csv"
    return (json.loads(s.read_text()) if s.exists() else None,
            pd.read_csv(m, dtype={"noise": str}) if m.exists() else None)


def parse_upload(file, fs_in: int):
    df = pd.read_csv(io.BytesIO(file.getvalue()))
    df.columns = [str(c).strip().upper() for c in df.columns]
    present = [l for l in C.ALL_LEADS if l in df.columns]
    if not present:
        raise ValueError("no lead columns found. The first row must name the leads, e.g. I, II, V1.")
    n = C.FS * C.SIGNAL_SECONDS
    sig = np.full((n, 12), np.nan, dtype=np.float32)
    for l in present:
        x = df[l].astype(float).to_numpy()[: fs_in * C.SIGNAL_SECONDS]
        if fs_in != C.FS:
            x = resample(x, int(round(len(x) * C.FS / fs_in)))
        sig[:, C.ALL_LEADS.index(l)] = np.pad(x, (0, max(0, n - len(x))), mode="edge")[:n]
    return sig, present


def features_for(sig, leads):
    idx = [C.ALL_LEADS.index(l) for l in leads]
    return pd.DataFrame([record_features(sig[:, idx], leads=leads)])


# =====================================================================================
# Charts
# =====================================================================================
def ecg_paper(sig, leads, t0, t1, highlight):
    """ECG drawn like a bedside monitor: glowing trace on a dark grid (0.2 s / 0.5 mV major squares)."""
    rows = len(leads)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.012 if rows > 4 else 0.04)
    t = np.arange(sig.shape[0]) / C.FS
    for r, lead in enumerate(leads, start=1):
        x = sig[:, C.ALL_LEADS.index(lead)]
        x = x - np.nanmedian(x)
        on = lead in highlight
        if on:   # soft glow behind the trace
            fig.add_trace(go.Scattergl(x=t, y=x, mode="lines", hoverinfo="skip",
                                       line=dict(color="rgba(62,232,154,.22)", width=6)), row=r, col=1)
        fig.add_trace(go.Scattergl(
            x=t, y=x, mode="lines", line=dict(color=GREEN if on else "#4E6E6B", width=1.6 if on else 1.1),
            name=LEAD_LABEL(lead),
            hovertemplate=f"<b>Lead {LEAD_LABEL(lead)}</b><br>time %{{x:.2f}} s<br>voltage %{{y:.2f}} mV<extra></extra>"),
            row=r, col=1)
        lo, hi = np.nanmin(x), np.nanmax(x)
        pad = max(0.25, (hi - lo) * 0.15)
        fig.update_yaxes(range=[lo - pad, hi + pad], row=r, col=1)
        fig.add_annotation(x=t0, y=hi + pad, xanchor="left", yanchor="top", showarrow=False,
                           text=f"<b>{LEAD_LABEL(lead)}</b>" + ("" if on else " (not seen by this device)"),
                           font=dict(size=12, color=GREEN if on else "#6F8F8C", family=DISPLAY), bgcolor="rgba(3,20,26,.85)",
                           row=r, col=1)
    grid = dict(showgrid=True, gridcolor=GRID_MAJOR, gridwidth=1, zeroline=False,
                minor=dict(showgrid=True, gridcolor=GRID_MINOR, gridwidth=0.5))
    fig.update_xaxes(range=[t0, t1], dtick=0.2, minor_dtick=0.04, showticklabels=False, **grid)
    fig.update_xaxes(showticklabels=True, tickmode="linear", tick0=0, dtick=1 if t1 - t0 > 3 else 0.2,
                     title_text="Time (seconds)", row=rows, col=1)
    fig.update_yaxes(dtick=0.5, minor_dtick=0.1, showticklabels=False, fixedrange=False, **grid)
    base_layout(fig, height=max(240, (110 if rows <= 6 else 88) * rows + 70), showlegend=False, plot_bgcolor=PAPER,
                margin=dict(l=10, r=10, t=10, b=45))
    fig.update_xaxes(gridcolor=GRID_MAJOR, minor=dict(gridcolor=GRID_MINOR))
    fig.update_yaxes(gridcolor=GRID_MAJOR, minor=dict(gridcolor=GRID_MINOR))
    return fig


def probability_chart(probs, thresholds, truth):
    cls = C.CLASSES[::-1]
    flagged = [probs[c] >= thresholds[c] for c in cls]
    fig = go.Figure(go.Bar(
        y=[LONG[c] for c in cls], x=[probs[c] for c in cls], orientation="h",
        marker=dict(color=[CLASS_COLORS[c] if f else "#2C4C53" for c, f in zip(cls, flagged)]),
        text=[f"<b>{probs[c]:.0%}</b>" for c in cls], textposition="outside", cliponaxis=False,
        textfont=dict(size=14, color=INK),
        customdata=[[thresholds[c], "Flagged" if f else "Not flagged", MEANING[c],
                     "yes" if c in (truth or []) else ("no" if truth else "unknown")] for c, f in zip(cls, flagged)],
        hovertemplate="<b>%{y}</b><br>Chance: %{x:.0%}<br>Flag line: %{customdata[0]:.0%} → %{customdata[1]}"
                      "<br>Cardiologist said: %{customdata[3]}<br><br>%{customdata[2]}<extra></extra>"))
    for i, c in enumerate(cls):
        fig.add_shape(type="line", x0=thresholds[c], x1=thresholds[c], y0=i - 0.42, y1=i + 0.42,
                      line=dict(color=INK, width=2, dash="dot"))
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", gridcolor=LINE, title_text="Calibrated chance",
                     dtick=0.25, zeroline=False)
    fig.update_yaxes(tickfont=dict(size=13))
    return base_layout(fig, 330, showlegend=False, bargap=0.35, margin=dict(l=10, r=30, t=10, b=45))


def reasons_chart(ex, cls):
    ex = ex.iloc[::-1]
    names = [describe_feature(f)[0] for f in ex.feature]
    vals = [fmt_value(v, describe_feature(f)[1]) for f, v in zip(ex.feature, ex.value)]
    fig = go.Figure(go.Bar(
        y=names, x=ex.shap, orientation="h",
        marker_color=[CRIMSON if s > 0 else TEAL for s in ex.shap],
        text=[f"{s:+.2f}" for s in ex.shap], textposition="outside", cliponaxis=False,
        customdata=vals,
        hovertemplate="<b>%{y}</b><br>Measured: %{customdata}<br>Push: %{x:+.2f} "
                      f"(positive = more likely {SHORT[cls].lower()})<extra></extra>"))
    span = max(0.1, ex.shap.abs().max() * 1.35)
    fig.update_xaxes(range=[-span, span], zeroline=True, zerolinecolor=INK, zerolinewidth=1.5, gridcolor=LINE,
                     title_text=f"← makes {SHORT[cls].lower()} less likely      more likely →")
    fig.update_yaxes(tickfont=dict(size=12))
    return base_layout(fig, 60 + 34 * len(ex), showlegend=False, margin=dict(l=10, r=40, t=10, b=50))


# =====================================================================================
# Page
# =====================================================================================
models, manifest = load_models()
summary, metrics = load_results()

with st.sidebar:
    st.markdown("### Step 1: Choose a recording")
    samples = load_samples()
    source = st.radio("Source", ["Example from the study", "Upload my own CSV"], label_visibility="collapsed",
                      help="Examples are real hospital ECGs the model never saw while learning.")
    sig, available, truth, who = None, list(C.ALL_LEADS), None, ""
    if source == "Example from the study":
        if samples:
            key = st.selectbox("Example ECG", list(samples),
                               help="Each example was read by a cardiologist; their diagnosis is shown after the number "
                                    "so you can compare it with the model.")
            sig, truth = samples[key]["signal"], samples[key]["labels"]
            age, sex = samples[key]["age"], samples[key]["sex"]
            if not np.isnan(age):
                who = f"{'Man' if sex == 0 else 'Woman'}, {'90 or older' if age >= 90 else f'{age:.0f} years'}"
                st.caption(who)
        else:
            st.info("No examples found. Run the notebook to create the app_samples folder.")
    else:
        up = st.file_uploader("ECG file (CSV)", type="csv",
                              help="One column per lead, with the lead names (I, II, III, AVR, AVL, AVF, V1-V6) "
                                   "in the first row. 10 seconds are used.")
        fs_in = st.number_input("Samples per second in the file", 50, 2000, 100, step=50,
                                help="The sampling rate of your recording. It is converted to 100 per second.")
        if up is not None:
            try:
                sig, available = parse_upload(up, int(fs_in))
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read this file: {e}")

    st.markdown("### Step 2: Choose a device")
    usable = [ls for ls in C.LEAD_SET_ORDER if ls in models and set(C.LEAD_SETS[ls]) <= set(available)]
    lead_set = st.radio("Device", usable or ["-"], format_func=lambda s: DEVICE_NAME.get(s, s),
                        label_visibility="collapsed",
                        help="Fewer leads means fewer viewing angles of the heart. The model for each device only "
                             "sees that device's leads.")
    if lead_set in DEVICE_NOTE:
        st.caption(DEVICE_NOTE[lead_set])

    st.markdown("### Step 3: Add movement noise")
    snr = st.select_slider("Noise level", options=["clean", 24, 18, 12, 6],
                           format_func=lambda v: {"clean": "None", 24: "Light", 18: "Moderate",
                                                  12: "Strong", 6: "Very strong"}[v],
                           help="Simulates a wearable worn while moving: breathing drift, muscle twitches and "
                                "movement bumps. The model was trained on clean hospital ECGs.")
    seed = 7
    if snr != "clean":
        st.caption(f"Signal-to-noise ratio {snr} dB. Lower dB means noisier.")

    st.divider()
    st.toggle("Show plain-English explanations", value=True, key="plain",
              help="Turn off for a cleaner view once you know your way around.")

# -------------------------------------------------------------------- header


def ecg_path(width=1600, height=92, beats=9):
    """SVG path of a stylised ECG trace for the header."""
    pts, base, step = [], height * 0.62, width / beats
    for b in range(beats):
        x0 = b * step
        shape = [(0, 0), (.16, 0), (.2, -.07), (.24, 0), (.33, 0), (.355, .1), (.38, -.62), (.405, .26),
                 (.43, 0), (.56, 0), (.64, -.18), (.72, 0), (1, 0)]
        pts += [(x0 + fx * step, base + fy * height) for fx, fy in shape]
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)


_path = ecg_path()
st.markdown(f'''<div class="hero">
  <svg class="trace" viewBox="0 0 1600 92" preserveAspectRatio="none" aria-hidden="true">
    <path class="ghost" d="{_path}"/><path d="{_path}"/></svg>
  <div class="content"><h1>Low-Lead ECG Screening</h1>
  <p>How much can a smartwatch tell about your heart compared with a hospital ECG? Pick a real heart recording,
  choose a device, and see what the model notices and why.</p></div>
</div>''', unsafe_allow_html=True)
st.markdown('<div class="notice"><b>For education only.</b> This is a screening-support prototype, not a medical '
            'device. It cannot diagnose anything; always see a doctor about heart concerns.</div>',
            unsafe_allow_html=True)
if manifest.get("synthetic"):
    st.markdown('<div class="notice synthetic"><b>Test models loaded.</b> These were trained on synthetic data. '
                'Run the notebook on PTB-XL to load the real models.</div>', unsafe_allow_html=True)
if not models:
    st.error("No trained models found in the models folder. Run the notebook first (section 13 saves them).")
    st.stop()

with st.expander("New here? How to use this page", expanded=False):
    st.markdown("""
1. **Choose a recording** in the left panel. Each example is a real 10-second ECG with the diagnosis a cardiologist gave it.
2. **Choose a device.** *Hospital ECG* sees the heart from 12 angles; *Smartwatch* sees it from only one.
3. **Optionally add noise** to imitate a wearable worn while moving.
4. **Read the result** in the *Check an ECG* tab: what the model flags, how sure it is, and which parts of the heartbeat made it decide.
5. **Compare devices** to see what a smartwatch would miss, and open **Learn the basics** for any word you don't know.
""")

if sig is None or lead_set == "-":
    st.info("Choose an example or upload a file in the left panel to begin.")
    st.stop()

model = models[lead_set]
sig_used = sig if snr == "clean" else add_wearable_noise(np.nan_to_num(sig), float(snr), seed)
feats = features_for(sig_used, model.leads)
probs = model.predict_proba(feats)
flagged = [c for c in C.CLASSES if probs[c] >= model.thresholds[c]]

hr_cols = [c for c in feats.columns if c.endswith("__hr")]
heart_rate = float(np.nanmedian(feats[hr_cols].values)) if hr_cols else float("nan")
abnormal_flags = [c for c in flagged if c != "NORM"]
result_txt, result_col = (("Review", RED) if abnormal_flags else ("Normal", GREEN) if "NORM" in flagged
                          else ("Unclear", AMBER))
beat_s = 60 / heart_rate if heart_rate == heart_rate and heart_rate > 0 else 1
quality = {"clean": "Clean", 24: "Light noise", 18: "Moderate", 12: "Strong", 6: "Very noisy"}[snr]
st.markdown(f'''<div class="vitals">
  <div class="vital"><div class="k">Heart rate</div><div class="v" style="color:{GREEN}">
     <span class="beat" style="--beat:{beat_s:.2f}s">♥</span> {heart_rate:.0f}<span class="u">bpm</span></div></div>
  <div class="vital"><div class="k">Device</div><div class="v" style="color:{CYAN}">
     {len(model.leads)}<span class="u">{"lead" if len(model.leads) == 1 else "leads"}</span></div></div>
  <div class="vital"><div class="k">Signal</div><div class="v" style="color:{AMBER}">{quality}</div></div>
  <div class="vital"><div class="k">Screening result</div><div class="v" style="color:{result_col}">{result_txt}</div></div>
</div>''', unsafe_allow_html=True)

tab_check, tab_compare, tab_results, tab_learn = st.tabs(
    ["Check an ECG", "Compare devices", "Study results", "Learn the basics"])

# ==================================================================== tab: check
with tab_check:
    abnormal = [c for c in flagged if c != "NORM"]
    if abnormal:
        headline = "Worth a closer look: " + ", ".join(f"{SHORT[c].lower()} ({probs[c]:.0%})" for c in abnormal)
        sub = "The model sees patterns linked to " + " and ".join(LONG[c] for c in abnormal) + "."
    elif "NORM" in flagged:
        headline = f"Looks normal ({probs['NORM']:.0%} chance)"
        sub = "No condition crossed its flag line."
    else:
        top = max(C.CLASSES, key=probs.get)
        headline = "Nothing stands out clearly"
        sub = f"No category crossed its flag line. Highest: {LONG[top]} at {probs[top]:.0%}."
    if truth:
        hits = set(flagged) & set(truth)
        if set(truth) <= set(flagged):
            match_big, match_sub = "Matches", "The model flagged everything the cardiologist found."
        elif hits:
            match_big, match_sub = "Partly matches", "Missed: " + ", ".join(SHORT[c] for c in set(truth) - hits) + "."
        else:
            match_big, match_sub = "Does not match", "The model missed: " + ", ".join(SHORT[c] for c in truth) + "."
        side = (f'<div class="side"><div class="lbl">Cardiologist\'s diagnosis</div>'
                f'<div class="big">{", ".join(LONG[c] for c in truth)}</div>'
                f'<div class="sub"><b>{match_big}.</b> {match_sub}</div></div>')
    else:
        side = ('<div class="side"><div class="lbl">Cardiologist\'s diagnosis</div><div class="big">Not available'
                '</div><div class="sub">Uploaded recordings have no reference diagnosis.</div></div>')
    noise_txt = "clean signal" if snr == "clean" else f"{snr} dB movement noise"
    accent = RED if abnormal else GREEN if "NORM" in flagged else AMBER
    st.markdown(f'<div class="verdict"><div class="main" style="--accent:{accent}"><div class="lbl">{DEVICE_NAME[lead_set]}, {noise_txt}</div>'
                f'<div class="big">{headline}</div><div class="sub">{sub}</div></div>{side}</div>',
                unsafe_allow_html=True)

    # ---------- ECG viewer
    st.subheader("The recording")
    c1, c2 = st.columns([3, 2])
    with c1:
        window = st.slider("Zoom: which seconds to show", 0.0, 10.0, (0.0, 10.0), step=0.5,
                           help="Drag the two handles closer together to zoom in on a few heartbeats. You can also "
                                "drag across the chart, or scroll on it, to zoom.")
    with c2:
        show_all = st.radio("Leads to show", ["Only this device's leads", "All recorded leads"], horizontal=True,
                            help="Grey leads are recorded in the file but are not visible to the chosen device.")
    leads_shown = [l for l in C.ALL_LEADS if l in available] if show_all == "All recorded leads" else model.leads
    t0, t1 = window if window[1] - window[0] >= 0.5 else (window[0], window[0] + 0.5)
    st.plotly_chart(ecg_paper(sig_used, leads_shown, t0, t1, model.leads), width="stretch", config=PLOT_CONFIG)
    explain_box("Each row is one <b>lead</b>, a view of the heart's electrical signal from one direction. "
                "Every repeating pattern is one heartbeat: a small bump (<b>P wave</b>), a tall sharp spike "
                "(<b>QRS</b>) and a rounded bump (<b>T wave</b>). The grid follows standard ECG paper: each big "
                "square is 0.2 seconds wide and 0.5 mV tall. Hover over the line to read exact values.")

    # ---------- probabilities + reasons
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("How likely is each condition?")
        st.plotly_chart(probability_chart(probs, model.thresholds, truth), width="stretch", config=PLOT_CONFIG)
        explain_box("Bars show the model's <b>chance</b> for each condition (hover a bar for what it means). "
                    "The dotted line is that condition's <b>flag line</b>: a coloured bar crossed it and is flagged, "
                    "a grey bar did not. The chances are calibrated, so 30% means about 30 in 100 similar ECGs "
                    "really had that condition. More than one condition can be flagged at once.")
        with st.expander("What do these conditions mean?"):
            for c in C.CLASSES:
                st.markdown(f'<span class="chip" style="background:{CLASS_COLORS[c]}">{SHORT[c]}</span> '
                            f'{MEANING[c]}', unsafe_allow_html=True)

    with right:
        st.subheader("Why did it say this?")
        default_cls = abnormal[0] if abnormal else (flagged[0] if flagged else max(C.CLASSES, key=probs.get))
        cls = st.selectbox("Explain the result for", C.CLASSES, index=C.CLASSES.index(default_cls),
                           format_func=lambda c: f"{LONG[c]}: {probs[c]:.0%}",
                           help="See which measurements pushed this condition's chance up or down.")
        ex = model.explain(feats, cls, top=8)
        view = st.radio("Show as", ["Sentences", "Chart"], horizontal=True, label_visibility="collapsed")
        if view == "Chart":
            st.plotly_chart(reasons_chart(ex, cls), width="stretch", config=PLOT_CONFIG)
        else:
            html = []
            for _, r in ex.head(6).iterrows():
                name, unit = describe_feature(r.feature)
                strength = "strongly" if abs(r.shap) > 0.5 else "moderately" if abs(r.shap) > 0.2 else "slightly"
                direction = "more" if r.shap > 0 else "less"
                color = CRIMSON if r.shap > 0 else TEAL
                html.append(f'<div class="reason"><div><div class="what">{name}</div>'
                            f'<div class="val">Measured: {fmt_value(r.value, unit)}</div></div>'
                            f'<div class="eff" style="color:{color}">{"▲" if r.shap > 0 else "▼"} {r.shap:+.2f}</div>'
                            f'<div class="why">{strength.capitalize()} made {SHORT[cls].lower()} {direction} likely.'
                            f'</div></div>')
            st.markdown("".join(html), unsafe_allow_html=True)
        explain_box("This uses <b>SHAP</b>, which splits the decision into the push each measurement gave. "
                    f"<span style='color:{RED}'><b style='color:{RED}'>Red / ▲</b></span> pushed towards the condition, "
                    f"<span style='color:{GREEN}'><b style='color:{GREEN}'>green / ▼</b></span> pushed away. Bigger numbers mean a stronger "
                    "push. Look for the medical clues: Q dips and the ST segment for heart attacks, spike width for "
                    "conduction delays, spike height for thickened muscle.")

# ==================================================================== tab: compare
with tab_compare:
    st.subheader("The same heartbeat, seen by four devices")
    st.markdown('<p class="lede">The research question of this project on a single recording: what does each '
                'device conclude, and what would a smartwatch miss?</p>', unsafe_allow_html=True)
    devices = [ls for ls in C.LEAD_SET_ORDER if ls in models and set(C.LEAD_SETS[ls]) <= set(available)]
    table = {}
    for ls in devices:
        m = models[ls]
        p = m.predict_proba(features_for(sig_used, m.leads))
        table[ls] = {"p": p, "flag": [c for c in C.CLASSES if p[c] >= m.thresholds[c]]}

    fig = go.Figure()
    for ls in devices[::-1]:                      # reversed so the hospital ECG is drawn on top
        fig.add_trace(go.Bar(
            name=DEVICE_NAME[ls], y=[LONG[c] for c in C.CLASSES[::-1]],
            x=[table[ls]["p"][c] for c in C.CLASSES[::-1]], orientation="h", marker_color=DEVICE_COLORS[ls],
            text=[f"{table[ls]['p'][c]:.0%}" for c in C.CLASSES[::-1]], textposition="outside", cliponaxis=False,
            textfont=dict(size=11),
            hovertemplate=f"<b>{DEVICE_NAME[ls]}</b><br>%{{y}}: %{{x:.0%}}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", gridcolor=LINE, dtick=0.25, title_text="Calibrated chance")
    st.plotly_chart(base_layout(fig, 520, barmode="group", bargap=0.25, bargroupgap=0.05,
                                legend=dict(orientation="h", y=1.06, x=0, traceorder="reversed")),
                    width="stretch", config=PLOT_CONFIG)

    rows = []
    for ls in devices:
        row = {"Device": DEVICE_NAME[ls]}
        for c in C.CLASSES:
            row[SHORT[c]] = f"{'● ' if c in table[ls]['flag'] else ''}{table[ls]['p'][c]:.0%}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("Device"), width="stretch")
    st.caption("● = flagged (crossed that condition's flag line).")

    if "12-lead" in table and "1-lead" in table:
        missed = set(table["12-lead"]["flag"]) - set(table["1-lead"]["flag"]) - {"NORM"}
        extra = set(table["1-lead"]["flag"]) - set(table["12-lead"]["flag"]) - {"NORM"}
        if missed:
            msg = f"The smartwatch would <b>miss {', '.join(LONG[c] for c in missed)}</b>, which the hospital ECG flags."
        elif extra:
            msg = f"The smartwatch flags <b>{', '.join(LONG[c] for c in extra)}</b>, which the hospital ECG does not."
        else:
            msg = "For this recording the smartwatch flags the <b>same conditions</b> as the hospital ECG."
        st.markdown(f'<div class="explain">{msg}'
                    + (f" Cardiologist's diagnosis: {', '.join(LONG[c] for c in truth)}." if truth else "")
                    + "</div>", unsafe_allow_html=True)

# ==================================================================== tab: results
with tab_results:
    if summary is None or metrics is None:
        st.info("Run the notebook to create the study results (results/summary.json and metrics.csv).")
    else:
        best = summary["best_model"]
        ds = summary["dataset"]
        clean = metrics[metrics.noise == "clean"]
        st.subheader("What the study found")
        st.markdown(f'<p class="lede">Results on {ds["split_sizes"]["test"]:,} test ECGs that no model saw during '
                    f'training. Best model: {best}.</p>', unsafe_allow_html=True)
        cols = st.columns(4)
        stats = [(f"{ds['n_records_used']:,}", "hospital ECGs analysed"),
                 (f"{summary['macro_auroc'][best]['12-lead']:.3f}", "AUROC with 12 leads (1.0 = perfect)"),
                 (f"{summary['retention_pct'][best]['6-lead']:.0f}%", "of that kept with 6 leads"),
                 (f"{summary['retention_pct'][best]['1-lead']:.0f}%", "of that kept by a smartwatch")]
        for col, (n, l) in zip(cols, stats):
            col.markdown(f'<div class="stat"><div class="n">{n}</div><div class="l">{l}</div></div>',
                         unsafe_allow_html=True)
        st.write("")

        # --- 1. AUROC by device and model (dot plot with CIs)
        st.markdown("#### Accuracy by device and model")
        fig = go.Figure()
        for mname in C.MODEL_ORDER:
            d = clean[clean.model == mname].set_index("lead_set").reindex(C.LEAD_SET_ORDER)
            fig.add_trace(go.Bar(
                x=[DEVICE_NAME[ls] for ls in C.LEAD_SET_ORDER], y=d.macro_auroc, name=mname,
                marker_color=MODEL_COLORS[mname],
                customdata=np.column_stack([d.auroc_ci_lo, d.auroc_ci_hi]),
                text=[f"{v:.3f}" for v in d.macro_auroc], textposition="outside", cliponaxis=False, textangle=0,
                textfont=dict(color=INK, size=12),
                hovertemplate=f"<b>{mname}</b><br>%{{x}}<br>AUROC %{{y:.3f}}<br>"
                              "95% confidence interval %{customdata[0]:.3f} to %{customdata[1]:.3f}<extra></extra>"))
        fig.update_yaxes(title_text="AUROC (higher is better)", gridcolor=LINE, range=[0.75, 0.95])
        st.plotly_chart(base_layout(fig, 420, barmode="group", bargap=0.25, bargroupgap=0.06), width="stretch",
                        config=PLOT_CONFIG)
        explain_box("<b>AUROC</b> measures how well a model separates ECGs with a condition from those without "
                    "(0.5 = guessing, 1.0 = perfect), averaged over the five conditions. Hover a bar to see "
                    "its 95% confidence interval. Note the axis starts at 0.75 so the differences are visible.")

        # --- 2. Retention heatmap
        st.markdown("#### How much accuracy each device keeps")
        ret = clean.pivot(index="model", columns="lead_set", values="retention_pct").reindex(
            index=C.MODEL_ORDER, columns=C.LEAD_SET_ORDER)
        fig = go.Figure(go.Heatmap(
            z=ret.values, x=[DEVICE_NAME[c] for c in ret.columns], y=list(ret.index), zmin=85, zmax=100,
            colorscale=HEAT,
            text=[[f"{v:.1f}%" for v in row] for row in ret.values], texttemplate="%{text}",
            textfont=dict(size=15, color="#FFFFFF", family=DISPLAY), colorbar=dict(title="% kept", ticksuffix="%"),
            hovertemplate="<b>%{y}</b>, %{x}<br>keeps %{z:.1f}% of its 12-lead AUROC<extra></extra>"))
        st.plotly_chart(base_layout(fig, 300), width="stretch", config=PLOT_CONFIG)
        explain_box("Each cell is the reduced device's AUROC as a percentage of the same model's 12-lead AUROC. "
                    "100% means nothing was lost.")

        # --- 3. Per-condition
        st.markdown(f"#### Which conditions are hardest to catch with fewer leads ({best})")
        pcs = summary["per_class_auroc"][best]
        z = [[pcs[ls][c] for ls in C.LEAD_SET_ORDER] for c in C.CLASSES]
        fig = go.Figure(go.Heatmap(
            z=z, x=[DEVICE_NAME[ls] for ls in C.LEAD_SET_ORDER], y=[LONG[c] for c in C.CLASSES],
            zmin=0.75, zmax=0.95, colorscale=HEAT,
            text=[[f"{v:.3f}" for v in row] for row in z], texttemplate="%{text}", textfont=dict(size=15, color="#FFFFFF", family=DISPLAY),
            colorbar=dict(title="AUROC"),
            hovertemplate="<b>%{y}</b><br>%{x}: AUROC %{z:.3f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(base_layout(fig, 360), width="stretch", config=PLOT_CONFIG)
        worst = min(C.CLASSES, key=lambda c: pcs["1-lead"][c] / pcs["12-lead"][c])
        explain_box(f"<b>{LONG[worst]}</b> drops the most when only lead I is left "
                    f"({pcs['12-lead'][worst]:.2f} → {pcs['1-lead'][worst]:.2f}). Its signs appear in chest or lower "
                    "leads that a smartwatch cannot see.")

        # --- 4. Noise
        st.markdown(f"#### Effect of movement noise ({best})")
        nz = summary["noise_auroc"][best]
        levels = summary.get("noise_levels", ["clean", "24", "12", "6"])
        lab = {"clean": "None", "24": "Light (24 dB)", "12": "Strong (12 dB)", "6": "Very strong (6 dB)"}
        z = [[nz[ls][n] for n in levels] for ls in C.LEAD_SET_ORDER]
        fig = go.Figure(go.Heatmap(
            z=z, x=[lab.get(n, n) for n in levels], y=[DEVICE_NAME[ls] for ls in C.LEAD_SET_ORDER],
            colorscale=HEAT, zmin=0.78, zmax=0.94,
            text=[[f"{v:.3f}" for v in row] for row in z], texttemplate="%{text}", textfont=dict(size=15, color="#FFFFFF", family=DISPLAY),
            colorbar=dict(title="AUROC"),
            hovertemplate="<b>%{y}</b><br>noise: %{x}<br>AUROC %{z:.3f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title_text="Movement noise added to the test ECGs", side="bottom")
        st.plotly_chart(base_layout(fig, 330, margin=dict(l=10, r=20, t=20, b=50)), width="stretch",
                        config=PLOT_CONFIG)
        drop = {ls: nz[ls][levels[0]] - nz[ls][levels[-1]] for ls in C.LEAD_SET_ORDER}
        explain_box("Models were trained on clean hospital ECGs and then tested on the same ECGs with simulated "
                    "wearable noise. Read each row left to right: the smaller the change, the better the device "
                    f"copes. With very strong noise the hospital ECG loses {drop['12-lead']:.3f} and the smartwatch "
                    f"{drop['1-lead']:.3f}.")

        # --- 5. Calibration + tuning
        cA, cB = st.columns(2, gap="large")
        with cA:
            st.markdown("#### Are the probabilities trustworthy?")
            e = summary["ece"]
            fig = go.Figure()
            for kind, colr, name in [("uncalibrated", "#3B5960", "Before calibration"),
                                     ("calibrated", TEAL, "After calibration")]:
                vals = [e[m]["12-lead"][kind] for m in C.MODEL_ORDER]
                fig.add_trace(go.Bar(x=C.MODEL_ORDER, y=vals, name=name, marker_color=colr,
                                     text=[f"{v:.3f}" for v in vals], textposition="outside", cliponaxis=False,
                                     hovertemplate=f"{name}<br>%{{x}}: error %{{y:.3f}}<extra></extra>"))
            fig.update_yaxes(title_text="Calibration error (lower is better)", gridcolor=LINE)
            st.plotly_chart(base_layout(fig, 360, barmode="group"), width="stretch", config=PLOT_CONFIG)
            explain_box("Calibration error is the average gap between the stated chance and what really happened. "
                        "After calibration, a 30% chance really means about 30%.")
        with cB:
            if "tuning_val_auroc" in summary:
                st.markdown("#### Did tuning the settings help?")
                tv = summary["tuning_val_auroc"]
                fig = go.Figure()
                for kind, colr, name in [("default", "#3B5960", "Default settings"), ("tuned", GREEN, "Tuned")]:
                    vals = [tv[m][kind] for m in C.MODEL_ORDER]
                    fig.add_trace(go.Bar(x=C.MODEL_ORDER, y=vals, name=name, marker_color=colr,
                                         text=[f"{v:.3f}" for v in vals], textposition="outside", cliponaxis=False))
                lo = min(min(v.values()) for v in tv.values())
                fig.update_yaxes(title_text="AUROC on validation data", gridcolor=LINE, range=[lo - 0.02, 0.96])
                st.plotly_chart(base_layout(fig, 360, barmode="group"), width="stretch", config=PLOT_CONFIG)
                explain_box("Each model's settings were searched on validation data. Small gains mean the default "
                            "settings were already close to the best.")

        with st.expander("Full results table"):
            show = clean[["model", "lead_set", "macro_auroc", "auroc_ci_lo", "auroc_ci_hi", "macro_auprc",
                          "macro_f1", "macro_sens", "macro_spec", "macro_ece", "retention_pct"]].copy()
            show["lead_set"] = show["lead_set"].map(DEVICE_NAME)
            show.columns = ["Model", "Device", "AUROC", "CI low", "CI high", "AUPRC", "F1", "Sensitivity",
                            "Specificity", "Calibration error", "Retention %"]
            st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns[2:-1]} | {"Retention %": "{:.1f}"}),
                         width="stretch", hide_index=True)

# ==================================================================== tab: learn
with tab_learn:
    st.subheader("Learn the basics")
    a, b = st.columns([1.1, 1], gap="large")
    with a:
        st.markdown("#### One heartbeat, explained")
        tt = np.linspace(-0.3, 0.6, 600)
        g = lambda mu, sd, amp: amp * np.exp(-0.5 * ((tt - mu) / sd) ** 2)
        beat = g(-0.16, 0.025, 0.15) + g(-0.025, 0.008, -0.12) + g(0, 0.011, 1.1) + g(0.03, 0.01, -0.25) \
            + g(0.28, 0.05, 0.3)
        fig = go.Figure([go.Scatter(x=tt, y=beat, mode="lines", line=dict(color="rgba(62,232,154,.25)", width=9),
                                    hoverinfo="skip"),
                         go.Scatter(x=tt, y=beat, mode="lines", line=dict(color=GREEN, width=2.6), hoverinfo="skip")])
        notes = [(-0.16, 0.17, "P wave", "upper chambers contract", -30, -55),
                 (0.0, 1.12, "QRS (main spike)", "main chambers contract", 70, -10),
                 (-0.025, -0.13, "Q dip", "", -40, 35),
                 (0.11, 0.02, "ST segment", "should be flat", 10, 55),
                 (0.28, 0.32, "T wave", "main chambers recover", 40, -50)]
        for x, y, title, desc, ax, ay in notes:
            fig.add_annotation(x=x, y=y, text=f"<b>{title}</b>" + (f"<br>{desc}" if desc else ""), showarrow=True,
                               arrowhead=0, arrowcolor=MUTED, ax=ax, ay=ay, align="left",
                               font=dict(size=12, color=INK), bgcolor="rgba(3,20,26,.92)", bordercolor=LINE)
        fig.add_vrect(x0=0.05, x1=0.17, fillcolor=AMBER, opacity=0.12, line_width=0)
        grid = dict(showgrid=True, gridcolor=GRID_MAJOR, minor=dict(showgrid=True, gridcolor=GRID_MINOR),
                    zeroline=False)
        fig.update_xaxes(dtick=0.2, minor_dtick=0.04, title_text="Time around the main spike (seconds)", **grid)
        fig.update_yaxes(dtick=0.5, minor_dtick=0.1, title_text="Voltage (mV)", range=[-0.45, 1.45], **grid)
        base_layout(fig, 380, plot_bgcolor=PAPER, showlegend=False)
        fig.update_xaxes(gridcolor=GRID_MAJOR, minor=dict(gridcolor=GRID_MINOR))
        fig.update_yaxes(gridcolor=GRID_MAJOR, minor=dict(gridcolor=GRID_MINOR))
        st.plotly_chart(fig, width="stretch",
                        config=PLOT_CONFIG)
        st.markdown("""
Each heartbeat starts with a small electrical signal that spreads through the heart and makes it squeeze.
Electrodes on the skin pick it up, and the ECG draws it over time. Doctors read the **shape, size and timing** of the waves:

- a **wide main spike** suggests the signal is travelling slowly (conduction delay);
- an **unusually tall spike** can mean thickened muscle;
- a **raised or lowered ST segment** or an **upside-down T wave** can mean the muscle is short of blood;
- **deep Q dips** can be the mark of an old heart attack.
""")
    with b:
        st.markdown("#### Why the number of leads matters")
        st.markdown("""
A **lead** is the heart's signal seen from one direction. A hospital ECG uses 10 electrodes to get **12 views**,
like 12 cameras placed around the heart. A smartwatch has electrodes only at your wrist and fingertip, so it
records **one view** (lead I, across the arms).

One view is enough to see the heart rate, the rhythm and many shape changes. But damage on a part of the heart
that the single camera faces side-on can be invisible to it. That is why, in this study, heart-attack signs were
the hardest condition for a smartwatch to catch.
""")
        st.markdown("#### How the model works")
        st.markdown("""
1. The ECG is cleaned and every heartbeat is found.
2. **48 measurements per lead** are taken: heart rate, rhythm regularity, wave heights, spike width, ST level, and the average beat shape.
3. A **LightGBM** model, trained on about 17,000 hospital ECGs, turns those measurements into a chance for each condition.
4. The chances are **calibrated** and compared with each condition's **flag line**.
5. **SHAP** shows which measurements mattered most.
""")
    st.markdown("#### Glossary")
    q = st.text_input("Search a term", placeholder="e.g. ST segment, lead, AUROC",
                      help="Type any word you met on this page.")
    items = [(t, d) for t, d in GLOSSARY.items() if not q or q.lower() in t.lower() or q.lower() in d.lower()]
    if not items:
        st.caption("No term matches that search. Try a shorter word.")
    g1, g2 = st.columns(2, gap="large")
    for i, (t, d) in enumerate(items):
        (g1 if i % 2 == 0 else g2).markdown(f'<div class="term"><div class="t">{t}</div><div class="d">{d}</div></div>',
                                            unsafe_allow_html=True)
    st.caption("Data: Wagner P. et al., PTB-XL, a large publicly available electrocardiography dataset. "
               "Scientific Data 7, 154 (2020). PhysioNet, CC BY 4.0.")
