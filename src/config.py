"""Shared settings for the Streamlit app and the report/slide scripts (the notebook has its own settings cell).

If you change a setting here (e.g. the lead sets), change it in the notebook's settings cell too.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PTBXL_DIR = DATA_DIR / "ptbxl"            # PTB-XL is extracted here
FEATURE_DIR = DATA_DIR / "features"        # cached feature tables (parquet)
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT / "models"                # trained models used by the app
SAMPLE_DIR = ROOT / "app_samples"          # example ECGs bundled with the app
REPORT_DIR = ROOT / "reports"

PTBXL_VERSION = "1.0.3"
PTBXL_ZIP_URL = (
    "https://physionet.org/static/published-projects/ptb-xl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
)

# --------------------------------------------------------------------------
# Signal settings
# --------------------------------------------------------------------------
FS = 100                 # we use the 100 Hz version of PTB-XL (10 s = 1000 samples)
SIGNAL_SECONDS = 10
BANDPASS = (0.5, 40.0)   # Hz, standard diagnostic-ish ECG band

# PTB-XL stores leads in this order
ALL_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]

# Lead configurations compared in the study (the core research question)
LEAD_SETS = {
    "12-lead": ALL_LEADS,
    # Limb leads only: what 6-lead handheld devices (e.g. two-electrode
    # + left-leg contact) produce. Leads III, aVR, aVL, aVF are derived from I & II.
    "6-lead": ["I", "II", "III", "AVR", "AVL", "AVF"],
    # Two limb leads + one precordial lead, a common reduced-lead patch layout
    "3-lead": ["I", "II", "V2"],
    # Single lead I: what a smartwatch records (wrist-to-finger)
    "1-lead": ["I"],
}
LEAD_SET_ORDER = list(LEAD_SETS.keys())

# --------------------------------------------------------------------------
# Labels: the five PTB-XL diagnostic superclasses
# --------------------------------------------------------------------------
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_NAMES = {
    "NORM": "Normal ECG",
    "MI": "Myocardial infarction",
    "STTC": "ST/T change",
    "CD": "Conduction disturbance",
    "HYP": "Hypertrophy",
}

# Official PTB-XL stratified folds: 1-8 train, 9 validation, 10 test
TRAIN_FOLDS = list(range(1, 9))
VAL_FOLDS = [9]
TEST_FOLDS = [10]

# --------------------------------------------------------------------------
# Wearable-noise simulation (applied to the test set only)
# --------------------------------------------------------------------------
# Signal-to-noise ratios in dB; "clean" means no added noise.
NOISE_LEVELS = ["clean", 24, 12, 6]

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
MODEL_ORDER = ["LogReg", "RandomForest", "LightGBM"]
CALIBRATION = "sigmoid"   # Platt scaling on the validation fold ("isotonic" also supported)
RANDOM_STATE = 42
N_BOOTSTRAP = 500         # bootstrap resamples for confidence intervals

# Model shipped in the Streamlit app (one per lead set)
APP_MODEL = "LightGBM"


def ensure_dirs() -> None:
    for d in [DATA_DIR, FEATURE_DIR, RESULTS_DIR, FIG_DIR, MODEL_DIR, SAMPLE_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
