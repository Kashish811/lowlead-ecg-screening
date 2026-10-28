# 🫀 Low-Lead ECG Screening

**Evaluating diagnostic accuracy retention with wearable-style reduced-lead signals using machine learning**

AUCML (TE7483) End-Semester Case Study Project · Symbiosis Institute of Technology, Pune

> ⚠️ **Screening-support prototype, not a diagnostic tool.** Outputs are research estimates for education only.

---

## The question

Smartwatches record **1 lead**, handheld devices **6**, patches **1-3**, while clinicians use **12**.
How much of the 12-lead diagnostic accuracy survives when the ECG is reduced to what a wearable can record,
and what happens when wearable noise is added?

This repo answers that on **PTB-XL** (21,799 clinical ECGs, 18,869 patients) for five diagnostic superclasses:

| Code | Superclass |
|------|------------|
| NORM | Normal ECG |
| MI   | Myocardial infarction |
| STTC | ST/T change |
| CD   | Conduction disturbance |
| HYP  | Hypertrophy |

## What is compared

| Axis | Levels |
|------|--------|
| Lead configuration | 12-lead · 6-lead (limb) · 3-lead (I, II, V2) · 1-lead (lead I, smartwatch) |
| Model | Logistic Regression · Random Forest · LightGBM (one-vs-rest, multi-label) |
| Signal quality | clean · 24 dB · 12 dB · 6 dB SNR (baseline wander + EMG + motion artefacts) |
| Metrics | macro AUROC (95% bootstrap CI) · AUPRC · F1 · sensitivity/specificity · Brier · ECE · retention % · paired bootstrap test vs 12-lead |
| Explainability | TreeSHAP, global (per lead, per feature group) and per ECG |

Every model is trained on folds 1-8, **Platt-calibrated** and thresholded on fold 9, and tested once on fold 10
(official patient-wise PTB-XL split, so no patient leaks between splits).

## Quick start

**All the code of the study is in one notebook: [`LowLead_ECG_Screening.ipynb`](LowLead_ECG_Screening.ipynb).**
It runs from top to bottom with comments and explanations at every step.

```bash
git clone https://github.com/<your-username>/lowlead-ecg-screening.git
cd lowlead-ecg-screening
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements-train.txt
```

1. Open `LowLead_ECG_Screening.ipynb` in VS Code (or Jupyter) and select the `.venv` Python as the kernel.
2. In section 14, put your name, PRN and guide.
3. Click **Run All**. The notebook downloads PTB-XL (~1.7 GB), trains and evaluates every model, saves the
   figures and app models, and writes the report and slides into `reports/`.
4. Start the app: `streamlit run streamlit_app.py`

Typical runtime on a 4-8 core laptop: 30-90 min for the first run. Features are cached in `data/features/`, so
re-runs take a few minutes. Set `FAST_MODE = True` in the settings cell for a quicker check with smaller models.

If the PhysioNet download is slow, download the zip manually from <https://physionet.org/content/ptb-xl/1.0.3/>,
extract it into `data/ptbxl/` and run the notebook again; it finds the files automatically.

**Quick 3-minute test without downloading:** run `python scripts/make_synthetic_ptbxl.py --n 2000`, then set
`USE_SYNTHETIC_DATA = True` in the settings cell. Synthetic results are clearly labelled in the app, report and
slides. **Do not submit them.**

## Outputs

| Path | Content |
|------|---------|
| `results/metrics.csv` | every model × lead set × noise level, all metrics |
| `results/summary.json` | headline numbers used by the report and slides |
| `results/paired_bootstrap_vs_12lead.csv` | significance of each reduced-lead loss |
| `results/shap_*.csv` | SHAP importance tables |
| `results/figures/*.png` | 14 figures (pipeline, data, results, calibration, SHAP) |
| `models/*.json` | calibrated LightGBM models for the app (portable JSON, no pickles) |
| `app_samples/` | held-out test ECGs + example CSVs for the app |
| `reports/*.docx, *.pptx` | report and presentation generated from your results |

## Repository structure

```
lowlead-ecg-screening/
├── LowLead_ECG_Screening.ipynb   # ⭐ the whole study, step by step (main code)
├── streamlit_app.py              # demo app (Streamlit Cloud entry point)
├── src/                          # small helpers the app needs (the notebook checks they match)
│   ├── config.py                 #   lead sets, classes, folders
│   ├── features.py               #   same feature code as notebook section 6
│   ├── noise.py                  #   same noise code as notebook section 7
│   └── app_model.py              #   loads the JSON models saved in notebook section 13
├── scripts/
│   ├── make_report.py            # Word report from results/summary.json
│   ├── make_slides.py            # PowerPoint from results/summary.json
│   ├── narrative.py              # shared, data-driven findings text
│   └── make_synthetic_ptbxl.py   # fake PTB-XL-format data for a quick test
├── tests/test_core.py
├── docs/                         # deployment guide, viva prep, rubric mapping
├── requirements.txt              # app (Streamlit Cloud)
├── requirements-train.txt        # notebook + report + slides
└── packages.txt                  # system package for LightGBM on Streamlit Cloud
```

## Deploying

See [`docs/GITHUB_AND_STREAMLIT_GUIDE.md`](docs/GITHUB_AND_STREAMLIT_GUIDE.md) for step-by-step GitHub and
Streamlit Community Cloud instructions.

## Data and citation

PTB-XL is © its authors and distributed by PhysioNet under CC BY 4.0. Example ECGs in `app_samples/` are from
PTB-XL test fold 10.

> Wagner, P., Strodthoff, N., Bousseljot, R.-D. et al. *PTB-XL, a large publicly available electrocardiography
> dataset.* Scientific Data 7, 154 (2020). https://doi.org/10.1038/s41597-020-0495-6

> Goldberger, A. et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation 101(23), e215-e220 (2000).

## Limitations

Reduced-lead inputs are subsets of clinical recordings, not real wearable recordings; noise is simulated;
PTB-XL is single-centre (Germany, 1989-1996). See the report, section 8.

Code: MIT License.
