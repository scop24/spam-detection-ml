"""
src/models.py
-------------
Model pipelines + tuning grids, and the two non-ML baselines every model
must beat: (a) majority class, (b) a keyword rule filter.

Class imbalance (~13% spam in SMS) is handled with class weights
(balanced) for LR/SVM/RF and scale_pos_weight for XGBoost -- applied
inside the pipelines, so CV folds stay honest.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.features import build_linear_features, build_tree_features

RANDOM_STATE = 42


def get_model_specs(scale_pos_weight=7.0):
    """name -> (pipeline, param_grid). Grids are small and tuned with
    Stratified K-Fold CV optimising PR-AUC (average precision)."""
    specs = {}

    specs["Naive Bayes"] = (
        Pipeline([("features", build_linear_features(scaler="maxabs")),
                  ("clf", MultinomialNB())]),
        {"clf__alpha": [0.1, 0.5, 1.0]},
    )

    specs["Logistic Regression"] = (
        Pipeline([("features", build_linear_features(scaler="standard")),
                  ("clf", LogisticRegression(max_iter=1000,
                                             class_weight="balanced",
                                             random_state=RANDOM_STATE))]),
        {"clf__C": [0.1, 1.0, 10.0]},
    )

    # LinearSVC has no predict_proba; calibrate it (Platt scaling) so the
    # app can show probabilities and we can draw PR/ROC curves.
    specs["Linear SVM"] = (
        Pipeline([("features", build_linear_features(scaler="standard")),
                  ("clf", CalibratedClassifierCV(
                      LinearSVC(class_weight="balanced",
                                random_state=RANDOM_STATE), cv=3))]),
        {"clf__estimator__C": [0.1, 1.0]},
    )

    specs["Random Forest"] = (
        Pipeline([("features", build_tree_features()),
                  ("clf", RandomForestClassifier(
                      n_estimators=300, class_weight="balanced_subsample",
                      n_jobs=-1, random_state=RANDOM_STATE))]),
        {"clf__max_depth": [None, 30]},
    )

    specs["XGBoost"] = (
        Pipeline([("features", build_tree_features()),
                  ("clf", XGBClassifier(
                      n_estimators=300, learning_rate=0.1, subsample=0.9,
                      scale_pos_weight=scale_pos_weight,
                      eval_metric="logloss", n_jobs=-1,
                      random_state=RANDOM_STATE))]),
        {"clf__max_depth": [4, 6]},
    )
    return specs


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def majority_predict(y_train, n):
    """Always predict the majority class (ham) -- the accuracy trap."""
    majority = int(np.bincount(y_train).argmax())
    return np.full(n, majority)


SPAM_KEYWORDS = [
    "free", "win", "winner", "won", "prize", "cash", "claim", "urgent",
    "offer", "click", "call now", "congratulations", "lottery", "bonus",
    "credit", "loan", "earn", "money", "discount", "limited", "guaranteed",
    "selected", "reward", "gift", "buy now", "subscribe", "txt", "text",
]


def keyword_predict(texts, min_hits=2):
    """Spam if at least `min_hits` spam keywords appear. scores = hit count
    (used as a pseudo-probability for ranking metrics)."""
    preds, scores = [], []
    for t in texts:
        low = f" {str(t).lower()} "
        hits = sum(1 for k in SPAM_KEYWORDS if k in low)
        scores.append(hits)
        preds.append(1 if hits >= min_hits else 0)
    return np.asarray(preds), np.asarray(scores, dtype=float)
