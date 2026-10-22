"""
src/evaluate.py
---------------
Metric computation, threshold selection and report figures.

Why not accuracy alone? Only ~13% of SMS messages are spam, so a lazy
model that calls EVERYTHING ham scores ~87% accuracy while catching zero
spam. Precision / recall / F1 / PR-AUC measure what we actually care
about: of the messages we flag, how many were really spam (precision),
and of all real spam, how much did we catch (recall).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             precision_recall_curve, recall_score,
                             roc_auc_score, roc_curve)

SPAM = 1


def metric_dict(y_true, y_pred, y_score=None):
    """All headline metrics. y_score: spam probability (or ranking score)."""
    out = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_true, y_score)), 4)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return out


def choose_threshold(y_true, y_proba, min_precision=0.95):
    """Pick the decision threshold on VALIDATION data.

    A false positive (a real message sent to the spam folder) costs the
    user more than a missed spam, so among thresholds whose precision is
    at least `min_precision`, we take the one with the best F1. If no
    threshold reaches that precision, fall back to the plain best-F1 one.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    ok = prec[:-1] >= min_precision
    if ok.any():
        idx = int(np.argmax(np.where(ok, f1[:-1], -1)))
    else:
        idx = int(np.argmax(f1[:-1]))
    return {
        "threshold": round(float(thr[idx]), 3),
        "val_precision": round(float(prec[idx]), 4),
        "val_recall": round(float(rec[idx]), 4),
        "val_f1": round(float(f1[idx]), 4),
        "min_precision_required": min_precision,
    }


def pr_curve_arrays(y_true, y_proba, max_points=60):
    """Downsampled precision/recall vs threshold (for the app slider)."""
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    prec, rec, thr = prec[:-1], rec[:-1], thr
    if len(thr) > max_points:
        idx = np.linspace(0, len(thr) - 1, max_points).astype(int)
        prec, rec, thr = prec[idx], rec[idx], thr[idx]
    return {"thresholds": [round(float(x), 3) for x in thr],
            "precisions": [round(float(x), 4) for x in prec],
            "recalls": [round(float(x), 4) for x in rec]}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_confusion(cm, path, title="Confusion matrix (test set)"):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted ham", "Predicted spam"])
    ax.set_yticks([0, 1], ["Actual ham", "Actual spam"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13, fontweight="bold")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_pr(curves, path):
    """curves: {model_name: (y_true, y_score)}"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, y_score):.3f})")
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        axes[1].plot(rec, prec,
                     label=f"{name} (PR-AUC {average_precision_score(y_true, y_score):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate (recall)")
    axes[0].set_title("ROC curves (test set)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curves (test set)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold(y_true, y_proba, chosen, path):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(thr, prec[:-1], label="Precision")
    ax.plot(thr, rec[:-1], label="Recall")
    ax.axvline(chosen, color="red", ls="--",
               label=f"Chosen threshold = {chosen}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision vs recall on the validation set")
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cross_domain(results, path):
    """results: {'SMS -> SMS': f1, 'SMS -> Email': f1, 'Email -> Email': ..., }"""
    labels = list(results.keys())
    values = [results[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(labels, values, color=["#2563eb", "#f59e0b", "#2563eb", "#f59e0b"])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Same-domain vs cross-domain F1 (models trained on one, tested on the other)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
