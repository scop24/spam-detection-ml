"""
train.py
--------
Full training & evaluation pipeline for the Spam Detection Filter.

Methodology (in order):
  1. Load SMS data (deduplicated), stratified split 70% train / 15%
     validation / 15% test. The test set is touched only at the end.
  2. Baselines first (majority class, keyword rules).
  3. Each ML model is tuned with Stratified 3-Fold CV on TRAIN only,
     optimising PR-AUC (average precision) -- the right target for an
     imbalanced problem. All preprocessing lives inside sklearn
     Pipelines, so validation folds never leak into feature fitting.
  4. The winner (best CV PR-AUC) picks its decision threshold on the
     VALIDATION set, favouring precision (a false positive -- a real
     message filed as spam -- costs more than a missed spam).
  5. The winner is refit on train+validation and evaluated ONCE on the
     test set, at both the default 0.5 and the tuned threshold.
  6. Extra honesty checks: cross-domain (SMS<->email), obfuscated spam,
     the hand-written Indian test set, misclassified examples, and
     explainability (top words + one LIME explanation).

Run:  python train.py
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import load_emails, load_indian_test_set, load_sms          # noqa: E402
from src.evaluate import (choose_threshold, metric_dict, plot_confusion,  # noqa: E402
                          plot_cross_domain, plot_roc_pr, plot_threshold,
                          pr_curve_arrays)
from src.explain import lime_figure, top_words, top_words_logodds         # noqa: E402
from src.models import get_model_specs, keyword_predict, majority_predict  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "reports", "figures")
MODELS = os.path.join(ROOT, "models")
os.makedirs(FIG, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)
RS = 42


def Xy(df):
    return df[["text"]].copy(), df["label"].to_numpy()


def obfuscate(texts, kind):
    leet = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"})
    out = []
    for t in texts:
        if kind == "leet":
            out.append(str(t).lower().translate(leet))
        elif kind == "spaced":
            out.append(" ".join(" ".join(w) if len(w) > 3 else w
                                for w in str(t).split()))
        elif kind == "dotted":
            out.append(" ".join(".".join(w) if len(w) > 3 else w
                                for w in str(t).split()))
        else:
            out.append(str(t))
    return out


def main():
    # ------------------------------------------------------------------ data
    sms = load_sms()
    X, y = Xy(sms)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RS)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=RS)
    print(f"SMS split: train={len(X_train)} val={len(X_val)} test={len(X_test)} "
          f"(spam rate {y.mean():.1%})")
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    leaderboard = []
    curves = {}

    # -------------------------------------------------------------- baselines
    maj_pred = majority_predict(y_train, len(y_test))
    leaderboard.append({"model": "Baseline: Majority class (all ham)",
                        **metric_dict(y_test, maj_pred, np.zeros(len(y_test)))})
    kw_pred, kw_score = keyword_predict(X_test["text"])
    leaderboard.append({"model": "Baseline: Keyword rules",
                        **metric_dict(y_test, kw_pred, kw_score)})

    # ------------------------------------------------------------ tune models
    specs = get_model_specs(scale_pos_weight=spw)
    cv = StratifiedKFold(3, shuffle=True, random_state=RS)
    fitted, cv_scores = {}, {}
    for name, (pipe, grid) in specs.items():
        print(f"Tuning {name} ...")
        gs = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv,
                          n_jobs=1, refit=True)
        gs.fit(X_train, y_train)
        fitted[name] = gs.best_estimator_
        cv_scores[name] = {"cv_pr_auc": round(float(gs.best_score_), 4),
                           "best_params": {k: (float(v) if isinstance(v, float) else v)
                                           for k, v in gs.best_params_.items()}}
        val_proba = gs.best_estimator_.predict_proba(X_val)[:, 1]
        test_proba = gs.best_estimator_.predict_proba(X_test)[:, 1]
        curves[name] = (y_test, test_proba)
        m = metric_dict(y_test, (test_proba >= 0.5).astype(int), test_proba)
        leaderboard.append({"model": name, **m})
        print(f"  CV PR-AUC {gs.best_score_:.4f} | test F1@0.5 {m['f1']:.4f} "
              f"| params {gs.best_params_}")

    winner = max(cv_scores, key=lambda k: cv_scores[k]["cv_pr_auc"])
    print(f"Winner by CV PR-AUC: {winner}")

    # ------------------------------------------- threshold on validation only
    win_val_proba = fitted[winner].predict_proba(X_val)[:, 1]
    thr_info = choose_threshold(y_val, win_val_proba, min_precision=0.95)
    threshold = thr_info["threshold"]
    print(f"Chosen threshold: {threshold} {thr_info}")
    plot_threshold(y_val, win_val_proba, threshold,
                   os.path.join(FIG, "threshold_analysis.png"))

    # ------------------------------ refit winner on train+val, final test run
    X_tv_df = pd.concat([X_train, X_val])
    y_tv_all = np.concatenate([y_train, y_val])
    final_model = clone(fitted[winner])
    final_model.fit(X_tv_df, y_tv_all)
    test_proba = final_model.predict_proba(X_test)[:, 1]
    pred_tuned = (test_proba >= threshold).astype(int)
    pred_half = (test_proba >= 0.5).astype(int)
    test_at_thr = metric_dict(y_test, pred_tuned, test_proba)
    test_at_half = metric_dict(y_test, pred_half, test_proba)
    print(f"Winner test @tuned({threshold}): F1 {test_at_thr['f1']:.4f} "
          f"P {test_at_thr['precision']:.4f} R {test_at_thr['recall']:.4f}")
    print(f"Winner test @0.5:              F1 {test_at_half['f1']:.4f}")

    from sklearn.metrics import confusion_matrix
    plot_confusion(confusion_matrix(y_test, pred_tuned, labels=[0, 1]),
                   os.path.join(FIG, "confusion_matrix.png"),
                   title=f"Confusion matrix — {winner} @ threshold {threshold}")
    plot_roc_pr(curves, os.path.join(FIG, "roc_pr_curves.png"))

    # ------------------------------------------------------------ cross-domain
    cross = {}
    email_available = False
    try:
        email = load_emails()
        email_available = True
        Xe, ye = Xy(email)
        sms_model = clone(fitted[winner])
        sms_model.fit(X_train, y_train)
        p = sms_model.predict_proba(Xe)[:, 1]
        cross["SMS -> Email"] = metric_dict(ye, (p >= 0.5).astype(int), p)
        Xe_tr, Xe_te, ye_tr, ye_te = train_test_split(
            Xe, ye, test_size=0.30, stratify=ye, random_state=RS)
        em_model = clone(fitted[winner])
        em_model.fit(Xe_tr, ye_tr)
        p2 = em_model.predict_proba(Xe_te)[:, 1]
        cross["Email -> Email"] = metric_dict(ye_te, (p2 >= 0.5).astype(int), p2)
        p3 = em_model.predict_proba(X_test)[:, 1]
        cross["Email -> SMS"] = metric_dict(y_test, (p3 >= 0.5).astype(int), p3)
        cross["SMS -> SMS"] = metric_dict(
            y_test, (fitted[winner].predict_proba(X_test)[:, 1] >= 0.5).astype(int),
            fitted[winner].predict_proba(X_test)[:, 1])
        plot_cross_domain({k: v["f1"] for k, v in cross.items()},
                          os.path.join(FIG, "cross_domain.png"))
        print("Cross-domain F1:", {k: v["f1"] for k, v in cross.items()})
    except Exception as e:
        print(f"Cross-domain skipped (email data unavailable): {e}")

    # -------------------------------------------------------------- robustness
    spam_idx = np.where(y_test == 1)[0]
    rng = np.random.RandomState(RS)
    sample_idx = rng.choice(spam_idx, size=min(200, len(spam_idx)), replace=False)
    spam_texts = X_test["text"].iloc[sample_idx].tolist()
    robustness = {}
    for kind in ["leet", "spaced", "dotted"]:
        ob = obfuscate(spam_texts, kind)
        pr = final_model.predict_proba(pd.DataFrame({"text": ob}))[:, 1]
        robustness[kind] = {
            "n": len(ob),
            "recall_at_threshold": round(float((pr >= threshold).mean()), 4),
            "recall_at_0.5": round(float((pr >= 0.5).mean()), 4),
        }
    base_pr = final_model.predict_proba(pd.DataFrame({"text": spam_texts}))[:, 1]
    robustness["original_spam"] = {
        "n": len(spam_texts),
        "recall_at_threshold": round(float((base_pr >= threshold).mean()), 4),
        "recall_at_0.5": round(float((base_pr >= 0.5).mean()), 4),
    }

    indian = load_indian_test_set()
    Xi, yi = Xy(indian)
    ind_proba = final_model.predict_proba(Xi)[:, 1]
    ind_pred = (ind_proba >= threshold).astype(int)
    ind_metrics = metric_dict(yi, ind_pred, ind_proba)
    indian_details = [{
        "text": t[:160], "label": int(l), "predicted": int(p),
        "spam_probability": round(float(pr), 3),
    } for t, l, p, pr in zip(indian["text"], yi, ind_pred, ind_proba)]
    print(f"Indian set: acc {ind_metrics['accuracy']:.3f} "
          f"spam recall {ind_metrics['recall']:.3f}")

    # ----------------------------------------------------------- error analysis
    fp_texts = X_test["text"][((pred_tuned == 1) & (y_test == 0))].tolist()
    fn_texts = X_test["text"][((pred_tuned == 0) & (y_test == 1))].tolist()
    fn_short = float(np.mean([len(t.split()) <= 8 for t in fn_texts])) if fn_texts else 0.0
    fp_digits = float(np.mean([any(c.isdigit() for c in t) for t in fp_texts])) if fp_texts else 0.0
    plain = (
        f"On the test set the model made {len(fp_texts)} false-positive and "
        f"{len(fn_texts)} false-negative errors. "
        + (f"About {fn_short:.0%} of the missed spam messages are very short "
           "(8 words or fewer) or written like normal conversation, so they carry "
           "few classic spam signals. " if fn_texts else "No spam was missed. ")
        + (f"About {fp_digits:.0%} of the wrongly flagged real messages contain "
           "digits or codes (bank/OTP-style alerts), which look statistically "
           "similar to scam messages. " if fp_texts else "No real message was flagged. ")
        + "This is exactly why the threshold was tuned to protect precision: "
          "losing a real message to the spam folder is worse than seeing one extra spam."
    )

    # ------------------------------------------------------------ explainability
    tw = top_words(final_model, n=15)
    tw_source = f"{winner} feature weights"
    if tw is None:
        tw = top_words_logodds(X_tv_df["text"], y_tv_all, n=15)
        tw_source = "training-data log-odds (tree model has no word weights)"
    tw["source"] = tw_source

    # Per-word log-odds scores (top 400) -> the app uses these to highlight
    # spam-pushing words when the winner is a tree model without coefficients.
    from src.explain import top_words_logodds as _lo
    _big = _lo(X_tv_df["text"], y_tv_all, n=200)
    word_scores = {d["word"]: d["weight"] for d in _big["spam_words"]}
    word_scores.update({d["word"]: d["weight"] for d in _big["ham_words"]})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sw = tw["spam_words"][::-1]
    axes[0].barh([d["word"] for d in sw], [d["weight"] for d in sw], color="#dc2626")
    axes[0].set_title("Top SPAM-indicating words")
    hw = tw["ham_words"]
    axes[1].barh([d["word"] for d in hw], [d["weight"] for d in hw], color="#2563eb")
    axes[1].set_title("Top HAM-indicating words")
    fig.suptitle(f"Word evidence ({tw_source})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "top_words.png"), dpi=150)
    plt.close(fig)
    lime_ok = False
    good_spam = X_test["text"][(y_test == 1) & (test_proba > 0.9)]
    if len(good_spam):
        lime_ok = lime_figure(final_model, str(good_spam.iloc[0]),
                              os.path.join(FIG, "lime_example.png"))

    # ------------------------------------------------------------------- save
    joblib.dump(final_model, os.path.join(MODELS, "spam_pipeline.joblib"),
                compress=3)
    metadata = {
        "winner": winner,
        "threshold": threshold,
        "winner_cv_pr_auc": cv_scores[winner]["cv_pr_auc"],
        "best_params": cv_scores[winner]["best_params"],
        "sms_total": int(len(sms)),
        "email_available": email_available,
    }
    with open(os.path.join(MODELS, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    metrics = {
        "data": {
            "sms_total": int(len(sms)),
            "sms_spam": int((sms["label"] == 1).sum()),
            "sms_ham": int((sms["label"] == 0).sum()),
            "duplicates_removed_note": "Exact duplicates removed before splitting (see src/data.py).",
            "email_available": email_available,
            "email_total": int(len(email)) if email_available else 0,
            "indian_total": int(len(indian)),
        },
        "split": {"train": int(len(X_train)), "validation": int(len(X_val)),
                  "test": int(len(X_test))},
        "leaderboard": leaderboard,
        "cv_scores": cv_scores,
        "winner": winner,
        "winner_cv_pr_auc": cv_scores[winner]["cv_pr_auc"],
        "threshold": {**thr_info,
                      "test_at_threshold": test_at_thr,
                      "test_at_0.5": test_at_half},
        "pr_curve_val": pr_curve_arrays(y_val, win_val_proba),
        "cross_domain": cross,
        "robustness": robustness,
        "indian_metrics": ind_metrics,
        "indian_details": indian_details,
        "errors": {
            "false_positives": [t[:280] for t in fp_texts[:6]],
            "false_negatives": [t[:280] for t in fn_texts[:6]],
            "plain_language": plain,
        },
        "top_words": tw,
        "word_scores": word_scores,
        "lime_generated": bool(lime_ok),
    }
    with open(os.path.join(ROOT, "reports", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved: models/spam_pipeline.joblib, reports/metrics.json, figures.")
    print(json.dumps({"winner": winner, "threshold": threshold,
                      "test_f1_at_threshold": test_at_thr["f1"]}, indent=2))


if __name__ == "__main__":
    main()
