# How this project meets each ESE rubric component (30 marks)

| Component (marks) | "Excellent" asks for | Where it is in this project |
|---|---|---|
| **Problem refinement & use-case relevance (3)** | refined from Phase-I; objective, scope, domain, users/stakeholders, real-world relevance | Report §1.1-1.4 (incl. scope/stakeholder table); slides 2-3 |
| **Dataset, preprocessing & features (4)** | source, size, attributes, target, data types, missing values, preprocessing, feature selection/engineering, validation strategy | Report §3 (attribute table, target, class table, missing values, split table) and §4; slides 4-6; notebook sections 3-8 |
| **Methodology & ML pipeline (5)** | data prep, model selection, training strategy, evaluation method, justification | Report §5 (pipeline figure, model-justification table, calibration, metrics, design); slide 7; notebook sections 9-10 |
| **Implementation / execution (5)** | complete, aligned with methodology, clear and reproducible code/workflow/outputs | One commented notebook that runs top to bottom, built-in consistency checks, pytest tests, cached features, fixed seeds, portable models, live Streamlit app; Report §6; slide 8 |
| **Results, metrics & comparison (5)** | accurate, clearly presented, suitable metrics (accuracy, precision, recall, F1, confusion matrix, ROC-AUC ...), comparison | AUROC with CIs, AUPRC, F1, sensitivity/specificity, confusion matrices, ROC curves, Brier/ECE, paired bootstrap significance, 3 models × 4 lead sets × 4 noise levels; Report §7; slides 9-15 |
| **Report quality (3)** | professional; problem statement, methodology, dataset, results, conclusion, future scope, references | `reports/AUCML_LowLead_ECG_Report.docx`, generated from your results (title page, abstract, 10 sections, numbered figures/tables, IEEE-style references, reproduction appendix) |
| **Communication & presentation (2)** | clear, structured, time-managed, effective visuals | `reports/AUCML_LowLead_ECG_Presentation.pptx`: 18 slides (≈ 12-15 min) with native charts and speaker notes; 2-minute live-demo script in `VIVA_PREP.md` |
| **Viva / project defence (3)** | deep understanding of project, dataset, model, methodology, results, limitations, future scope | `docs/VIVA_PREP.md`: 30 likely questions with answers and a demo script |

### Before you present
- [ ] Ran the notebook on real PTB-XL (`USE_SYNTHETIC_DATA = False`) (no red "synthetic" banner anywhere)
- [ ] Regenerated report and slides with your name, PRN and guide
- [ ] Added `reports/app_screenshot.png`, then regenerated
- [ ] Pushed to GitHub, app deployed, links added to README / report / slides
- [ ] Read the report's results section and can explain every number
- [ ] Rehearsed the talk with a timer and the live demo
