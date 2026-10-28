# From laptop to GitHub to a live Streamlit app

Follow the steps in order. Commands are for a terminal (Windows: PowerShell or Git Bash; macOS/Linux: Terminal),
run from inside the `lowlead-ecg-screening` folder.

---

## Step 0 · Run the study on your laptop first

The app on Streamlit Cloud only *loads* models; it does not train them. So train locally and commit the results.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements-train.txt
```

Open `LowLead_ECG_Screening.ipynb` in VS Code, choose the `.venv` Python as the kernel (top right), fill in your
name/PRN in section 14, and click **Run All** (~1.7 GB download + 30-90 min on the first run). Then:

```bash
python -m pytest -q                      # should print "9 passed"
streamlit run streamlit_app.py           # opens http://localhost:8501
```

Check in the app that the red "Smoke-test models" banner is **gone**. If it is still there, you are looking at the
synthetic sample models: run the notebook on real PTB-XL (`USE_SYNTHETIC_DATA = False`).

Optional: take a screenshot of the app, save it as `reports/app_screenshot.png`, and re-run the last cells of the
notebook (section 14) so it is added to the report and slides. You can also regenerate them from a terminal:

```bash
python scripts/make_report.py --author "Your Name" --prn 2024XXXXXX --guide "Dr. ..."
python scripts/make_slides.py  --author "Your Name" --prn 2024XXXXXX --guide "Dr. ..."
```

---

## Step 1 · Put the project on GitHub

### 1a. One-time setup
1. Create a free account at <https://github.com> (use your college email if you like).
2. Install Git: <https://git-scm.com/downloads>. Check with `git --version`.
3. Tell Git who you are:
   ```bash
   git config --global user.name  "Your Name"
   git config --global user.email "you@example.com"
   ```

### 1b. Create an empty repository on GitHub
1. Click **+** (top right) → **New repository**.
2. Name: `lowlead-ecg-screening`. Visibility: **Public** (Streamlit's free tier deploys public repos easily).
3. **Do not** tick "Add a README", ".gitignore" or "license" (the project already has them).
4. Click **Create repository** and keep the page open; it shows your repo URL.

### 1c. Push the code
```bash
git init
git add .
git status                 # make sure data/ is NOT listed (it is in .gitignore)
git commit -m "Low-lead ECG screening: pipeline, results, Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/lowlead-ecg-screening.git
git push -u origin main
```

* A browser window asks you to sign in to GitHub the first time (Git Credential Manager). If the terminal instead
  asks for a password, paste a **Personal Access Token** (GitHub → Settings → Developer settings → Personal access
  tokens → Tokens (classic) → Generate, tick `repo`), not your account password.
* Make sure these folders are pushed, because the app needs them: `models/`, `results/`, `app_samples/`.
  Check on the GitHub page after pushing.
* GitHub rejects files over 100 MB. The model files are normally 5-20 MB each. If one is bigger, re-run with
  `FAST_MODE = True` or lower `n_estimators` in the notebook's `make_model` function.

**Prefer clicking?** GitHub Desktop (<https://desktop.github.com>) → File → Add local repository → choose the folder →
Publish repository.

### 1d. Updating later
```bash
git add .
git commit -m "Describe what changed"
git push
```

---

## Step 2 · Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and **Continue with GitHub**; authorise Streamlit to see your repositories.
2. Click **Create app** → **Deploy a public app from GitHub** (wording may differ slightly).
3. Fill in:
   * **Repository:** `<your-username>/lowlead-ecg-screening`
   * **Branch:** `main`
   * **Main file path:** `streamlit_app.py`
   * **App URL:** choose something like `lowlead-ecg-<yourname>` (optional)
4. Open **Advanced settings** → **Python version: 3.11 or 3.12**. Leave secrets empty.
5. Click **Deploy**. The first build installs `requirements.txt` and `packages.txt` and takes 2-5 minutes.
6. Your app is live at `https://<your-app-url>.streamlit.app`. Every `git push` redeploys it automatically.

### Troubleshooting

| Symptom in the Streamlit log | Fix |
|---|---|
| `No trained models found in models/` in the app | You did not commit `models/`. Run the notebook, `git add models results app_samples`, commit, push. |
| `OSError: libgomp.so.1` | `packages.txt` with `libgomp1` must be in the repo root (it is included). Reboot the app. |
| `ModuleNotFoundError: src` | Main file path must be `streamlit_app.py` at the repo root, not inside a folder. |
| App sleeps after inactivity | Normal on the free tier. Open the URL and click "Yes, get this app back up". |
| Red "Smoke-test models" banner online | Synthetic models were pushed. Run the notebook on real PTB-XL, commit, push. |

---

## Step 3 · Put the links everywhere

1. Add the app URL and repo URL to the top of `README.md`, commit and push.
2. Put both links on the title or implementation slide and in the report's Implementation section.
3. Before the viva, open the app once so it is awake, and keep the local version ready
   (`streamlit run streamlit_app.py`) in case the college network blocks it.
