# Viva / project-defence preparation

The rubric awards the top band for "deep individual understanding of the project, dataset, model, methodology,
results, limitations and future scope". Below are the questions examiners most often ask about a project like
this, with short answers you should be able to say in your own words. Fill the bracketed numbers from
`results/summary.json` after your run.

## Problem and motivation
**Q1. What problem does your project solve?**
It measures how much ECG diagnostic accuracy is lost when the 12-lead ECG is reduced to the 6, 3 or 1 lead a
wearable can record, with and without realistic noise, for five diagnostic superclasses.

**Q2. Why is this useful in the real world?**
Wearables can screen continuously and cheaply. A device maker needs to know which conditions it can responsibly
flag, and a clinic needs calibrated risk estimates to decide whom to refer.

**Q3. How did you refine the problem from Phase-I?**
From generic "ECG classification" to a measurable question: accuracy retention versus lead count and noise, with
calibration and explainability as deployment requirements.

## Dataset
**Q4. Describe PTB-XL.** 21,799 ten-second 12-lead ECGs from 18,869 patients, 500 Hz and 100 Hz versions,
labelled by cardiologists with SCP-ECG statements, PhysioNet, CC BY 4.0. I used the 100 Hz version.

**Q5. How did you build the labels?** Map every SCP code with `diagnostic = 1` to its `diagnostic_class`; an ECG
gets every superclass its codes map to (multi-hot). Records with no superclass are dropped.

**Q6. Why is it multi-label and how do you handle that?** One ECG can show MI and CD together. I train one binary
classifier per class (one-vs-rest) and evaluate each class separately, then macro-average.

**Q7. How did you split the data and why?** Official `strat_fold`: 1-8 train, 9 validation, 10 test. Folds are
patient-wise (no patient in two splits, so no identity leakage) and label-stratified.

**Q8. Class imbalance?** HYP is the rarest class. I use class-balanced weights (LR, RF), report AUROC and AUPRC
(robust to imbalance), macro-average so rare classes count equally, and tune each class threshold on validation.

**Q9. Missing values?** Signals are complete; height/weight are mostly missing so they are not used. Feature-level
gaps (no detectable beats) are median-imputed inside the pipeline, fitted on training data only.

## Features and preprocessing
**Q10. Why 0.5-40 Hz band-pass?** Below 0.5 Hz is baseline wander from breathing; above 40 Hz is mostly muscle and
powerline noise. QRS and ST-T content lie inside the band. Zero-phase filtering (filtfilt) avoids shifting waves.

**Q11. How do you detect R peaks?** Squared first derivative, smoothed over 100 ms, adaptive threshold at 30% of
its 99th percentile, 300 ms refractory period, then refine to the largest absolute deflection. Polarity-independent,
so it works in aVR and V1 where QRS is negative.

**Q12. Why a median beat?** Averaging aligned beats suppresses noise and one-off ectopic beats, giving stable
P/QRS/ST/T measurements.

**Q13. Why hand-crafted features rather than a CNN?** CPU-friendly, fast, and above all interpretable: SHAP on named
clinical features (ST level, Q depth) can be checked against cardiology. A CNN is listed as future work.

**Q14. How do you simulate a smartwatch?** By using only lead I's feature columns. Because every column is named
`LEAD__feature`, reduced-lead models see exactly their own leads and share the same extraction.

**Q15. Why are aVR, aVL, aVF and III "free" in a 6-lead device?** They are linear combinations of I and II
(Einthoven / Goldberger), so a 3-electrode handheld device can display all six limb leads.

## Models, calibration, evaluation
**Q16. Why these three models?** Logistic Regression = linear, transparent baseline; Random Forest = robust bagged
non-linear ensemble; LightGBM = state-of-the-art boosting for tabular data with native missing values and fast
TreeSHAP.

**Q17. What is calibration and why Platt scaling?** A model is calibrated if, among ECGs given probability 0.3,
about 30% truly have the label. Tree ensembles are often over- or under-confident. Platt scaling fits a logistic
regression on the logit of the raw score using the validation fold. It has only two parameters per class, so it is
stable with a ~2,000-ECG validation fold (isotonic is also supported in `config.py`).

**Q18. What is ECE?** Expected Calibration Error: bin predictions into 10 bins, take the weighted average gap
between mean predicted probability and observed frequency. Lower is better.

**Q19. Why AUROC as the main metric?** It is threshold-independent and insensitive to prevalence, so it compares
lead sets fairly. AUPRC, F1, sensitivity and specificity complete the picture at the operating point.

**Q20. How did you choose thresholds?** For each class, the threshold maximising F1 on the validation fold. In a
real screening programme you would choose a high-sensitivity threshold instead.

**Q21. How do you know a loss is significant?** Paired bootstrap: resample test patients, compute the AUROC
difference between the reduced-lead and 12-lead model on the same resample, 500 times; if the 95% CI of the
difference excludes zero, the loss is significant.

**Q22. Retention, how is it defined?** Reduced-lead macro AUROC ÷ the same model's 12-lead macro AUROC × 100.

## Results (fill in from your run)
**Q23. Headline result?** "[best model] reached [AUROC 12] with 12 leads; 6 / 3 / 1 leads retained
[x] / [y] / [z] %."

**Q24. Which class suffers most and why?** Usually MI: its signs are localised (anterior → V1-V4, inferior →
II, III, aVF), and lead I cannot see them. Conduction disturbance and hypertrophy change QRS width/voltage globally,
so they survive in any lead.

**Q25. Why might 3 leads (I, II, V2) beat 6 limb leads?** The 6 limb leads all lie in the frontal plane; V2 adds a
horizontal-plane (precordial) view, which carries anterior MI and hypertrophy information.

**Q26. What did SHAP tell you?** Which leads and feature groups drive predictions; check that ST level and Q depth
matter for MI, QRS width for CD, voltage for HYP. That agreement with physiology increases trust.

**Q27. What does noise do?** AUROC drops as SNR falls; single-lead models have no redundancy to average noise away,
so they degrade more.

## Limitations, ethics, future
**Q28. Biggest limitation?** Reduced leads are taken from clinical recordings with ideal electrodes; real
wearables differ. Also single-centre, older data.

**Q29. Is this a diagnostic tool?** No. A screening-support prototype: it flags for review, it does not diagnose.
That is why probabilities are calibrated and every page has a disclaimer.

**Q30. What would you do next?** CNN on raw signals, noise-augmented training, validation on real smartwatch data,
lead reconstruction, rhythm classes like AF.

## Demo script (2 minutes)
1. Sidebar → example MI ECG → 12-lead: MI flagged, show SHAP reasons.
2. Switch to 1-lead: show MI probability fall; open "Lead-reduction explorer" for all four side by side.
3. Add 6 dB noise: show the trace and probabilities change.
4. "Study results" tab: point at the retention heatmap.
