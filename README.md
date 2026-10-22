# Spam Detection Filter using Machine Learning

An end-to-end ML project that classifies SMS messages (and emails) as
**spam** or **ham**, with the evaluation done the honest way: dumb
baselines first, an untouched test set, a tuned decision threshold,
cross-domain tests (SMS ↔ email), disguised-spam robustness tests, a
hand-written set of modern Indian scam messages, error analysis, and
word-level explanations for every prediction.

**Headline result:** XGBoost — test **F1 0.953** at the tuned threshold
0.831 (precision 0.979, recall 0.929), against a majority baseline that
scores 87% accuracy while catching zero spam.

## Quick start (Windows)

```bat
cd spam-detection-ml
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

The trained model is already saved in `models/`, so the app starts
instantly at http://localhost:8501. To retrain everything from scratch:

```bat
python train.py
```

(Python 3.10+ recommended; developed and tested on Python 3.12. Data is
cached in `data/processed/`, so after the first `train.py` run the whole
project works offline.)

## What the app does

- **Classifier tab** — paste any SMS/email: spam-probability gauge,
  SPAM / NOT SPAM verdict, the message re-shown with **spam-pushing words
  highlighted red and ham-pushing words green**, a **threshold slider**
  with live precision/recall readouts, one-click examples (including
  Indian KYC/lottery/delivery scams), and a **batch mode** (upload a CSV,
  download classified results).
- **Model Insights tab** — leaderboard vs baselines, confusion matrix,
  ROC & PR curves, threshold analysis, cross-domain results, robustness
  results, all 30 Indian test messages with verdicts, top-word chart,
  a LIME explanation, and the model's actual mistakes with commentary.
- **Data Explorer tab** — class balance, message-length distributions
  and top-word charts for SMS and email, with sample messages.

## Results (test set — never seen during training or tuning)

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Baseline: majority (all ham) | 0.874 | 0.000 | 0.000 | 0.000 | 0.126 |
| Baseline: keyword rules | 0.930 | 0.879 | 0.520 | 0.654 | 0.622 |
| Naive Bayes | 0.987 | 0.940 | 0.959 | 0.950 | 0.989 |
| Logistic Regression | 0.987 | 0.949 | 0.949 | 0.949 | 0.986 |
| Linear SVM | 0.988 | 0.968 | 0.939 | 0.953 | 0.987 |
| Random Forest | 0.982 | 0.988 | 0.867 | 0.924 | 0.983 |
| **XGBoost (winner, by CV)** | **0.988** | **0.979** | **0.929** | **0.953** | **0.989** |

Model rows at the tuned threshold 0.831 for XGBoost; others at 0.5.
Full numbers: `reports/metrics.json`.

**Beyond the leaderboard** (this is where the project earns its marks):

- **Threshold tuning:** chosen on the *validation* split — best F1 among
  thresholds with precision ≥ 0.95, because a false positive (a real
  message filed as spam) costs more than a missed spam.
- **Cross-domain:** train on SMS → test on email: F1 **0.63**; train on
  email → test on SMS: F1 **0.38** (vs ~0.95–0.99 same-domain). Different
  era, vocabulary and message length — models only know their own world.
- **Obfuscated spam recall** (at the tuned threshold): clean spam 92.9%,
  leetspeak (`fr33 m0ney`) 85.7%, spaced letters 83.7%, dotted letters
  (`f.r.e.e`) 58.2% — character n-grams help, but heavy disguise works.
- **Indian test set** (30 hand-written modern messages): accuracy 0.667.
  Honest failure: 2020s scam scripts (KYC, UPI, courier fees) barely exist
  in the 2000s-era training data. The production fix is fresh local data.

## How it works

1. **Data** — UCI SMS Spam Collection (5,572 → **5,171** after removing
   403 exact duplicates *before* splitting, so no message can appear in
   both train and test) + SpamAssassin public email corpus (3,437
   messages) for the cross-domain test + 30 hand-written Indian messages.
2. **Cleaning** — lowercase; URLs → `urltoken`, emails → `emailtoken`,
   phone numbers → `phonetoken`, digits → `numtoken`. Tokens, not
   deletion: the *shape* of a message is itself a spam signal.
3. **Features** — word TF-IDF (1–2 grams) + character TF-IDF (3–5 grams,
   catches disguises) + handcrafted features (length, uppercase ratio,
   digit ratio, URL count, special characters, currency presence). All
   inside sklearn Pipelines fitted on training data only — no leakage.
4. **Models** — Stratified 3-Fold CV tuning on the training split,
   optimising **PR-AUC** (not accuracy). Class imbalance handled with
   class weights / `scale_pos_weight` inside the pipelines.
5. **Selection & threshold** — winner by CV PR-AUC (XGBoost, 0.979),
   threshold on validation, winner refit on train+validation, single
   final evaluation on the test set.
6. **Explainability** — global word evidence (model weights for linear
   models; log-odds for the tree winner), per-message highlighting, and
   a LIME explanation figure.

## Project structure

```
spam-detection-ml/
├── app/app.py                  # Streamlit demo (3 tabs)
├── data/processed/             # cached CSVs (offline-ready)
├── models/                     # spam_pipeline.joblib + metadata.json
├── notebooks/spam_detection.ipynb
├── reports/metrics.json        # every number in this README
├── reports/figures/            # confusion, ROC/PR, threshold, cross-domain,
│                               # top words, LIME plots
├── src/
│   ├── data.py                 # downloads, email parsing, dedupe, Indian set
│   ├── features.py             # cleaning + feature pipelines
│   ├── models.py               # model specs, grids, baselines
│   ├── evaluate.py             # metrics, threshold choice, figures
│   └── explain.py              # word weights, contributions, LIME
├── train.py                    # one command retrains + re-evaluates everything
└── requirements.txt            # pinned versions
```

## Design Decisions

- **SpamAssassin as the second dataset** (public, no login, downloadable
  tarballs) instead of Enron — Enron is huge and mostly ham; SpamAssassin
  gives balanced, labelled spam+ham for a fair cross-domain test.
- **Duplicates removed before splitting** — identical messages in train
  and test inflate scores; 403 were removed from SMS, 261 from email.
- **Numbers/URLs replaced by tokens, never deleted** — they are spam
  signals; the raw values would be useless one-off vocabulary.
- **Character n-grams included** — word features alone are blind to
  `fr33`, `c@$h` style disguises; char grams see the fragments.
- **Tree models get SVD-compressed TF-IDF (150 dense features)** — trees
  do not work well on 70k sparse dimensions; linear models keep the full
  sparse space where they excel.
- **LinearSVC is calibrated** (Platt scaling) — SVMs output distances,
  not probabilities; the app gauge, PR curves and threshold tuning all
  need real probabilities.
- **Tuning optimises PR-AUC, winner chosen by CV** — accuracy rewards
  lazy models on imbalanced data; choosing the winner on the test set
  would be cheating, so the test set is used exactly once, at the end.
- **Class weights instead of SMOTE/resampling** — synthetic TF-IDF
  'messages' don't resemble real text; weights fix the imbalance inside
  each CV fold with no fabricated data.
- **Threshold tuned for precision ≥ 0.95 first** — losing a genuine
  message to the spam folder is worse for a user than seeing one spam.
- **Winner refit on train+validation** after the threshold is chosen —
  standard practice: more data for the deployed model, threshold kept
  from the untouched-validation estimate.
- **DistilBERT skipped deliberately** — it needs a ~2 GB PyTorch
  download and slow CPU fine-tuning, while classical models already
  reach F1 ≈ 0.95; the honest cost/benefit says skip for a lab project.
- **App highlighting uses model weights when available, else word
  log-odds** — the winner (XGBoost) has no per-word weights, so the app
  falls back to training-data log-odds, which the Insights tab states
  openly as the source.

## Screenshots

Run the app (Quick start above) — the Classifier tab's gauge, verdict
banner and word highlighting are the demo centrepiece. All evaluation
plots are in `reports/figures/` (confusion matrix, ROC/PR curves,
threshold analysis, cross-domain chart, top words, LIME example).
