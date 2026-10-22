"""
src/explain.py
--------------
Explainability helpers.

For linear models (Naive Bayes, Logistic Regression, Linear SVM) every
feature has a weight: positive = pushes towards SPAM, negative = pushes
towards HAM. We use those weights for (a) the global top spam/ham word
lists and (b) highlighting words in a single message. For tree models
(which have no word weights) the top-word lists fall back to a
data-driven log-odds calculation, and the app uses LIME-style local
explanations instead (see lime_figure).
"""

import numpy as np
import pandas as pd


def _get_coef(pipeline):
    """Coefficient vector aligned with the pipeline's feature step, or None."""
    clf = pipeline.named_steps["clf"]
    try:
        if hasattr(clf, "coef_"):                       # Logistic Regression
            return np.asarray(clf.coef_[0], dtype=float)
        if hasattr(clf, "feature_log_prob_"):           # Multinomial NB
            return np.asarray(clf.feature_log_prob_[1] - clf.feature_log_prob_[0],
                              dtype=float)
        if hasattr(clf, "calibrated_classifiers_"):     # Calibrated LinearSVC
            coefs = [cc.estimator.coef_[0] for cc in clf.calibrated_classifiers_
                     if hasattr(cc.estimator, "coef_")]
            if coefs:
                return np.mean(coefs, axis=0)
    except Exception:
        pass
    return None


def _feature_names(pipeline):
    feats = pipeline.named_steps["features"]
    try:
        return list(feats.get_feature_names_out())
    except Exception:
        return None


def top_words(pipeline, n=15):
    """Top spam-indicating and ham-indicating WORD features (word block only)."""
    coef = _get_coef(pipeline)
    names = _feature_names(pipeline)
    if coef is None or names is None:
        return None
    pairs = [(name.split("__", 1)[-1], float(w))
             for name, w in zip(names, coef) if name.startswith("word__")]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return {
        "spam_words": [{"word": w, "weight": round(v, 3)} for w, v in pairs[:n]],
        "ham_words": [{"word": w, "weight": round(v, 3)} for w, v in pairs[-n:]][::-1],
    }


def top_words_logodds(texts, labels, n=15):
    """Model-agnostic fallback: words with the highest spam-vs-ham log-odds
    in the training data (with smoothing)."""
    from sklearn.feature_extraction.text import CountVectorizer
    from src.features import clean_text
    cleaned = [clean_text(t) for t in texts]
    vec = CountVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000)
    X = vec.fit_transform(cleaned)
    y = np.asarray(labels)
    spam_counts = np.asarray(X[y == 1].sum(axis=0)).ravel() + 1.0
    ham_counts = np.asarray(X[y == 0].sum(axis=0)).ravel() + 1.0
    logodds = np.log(spam_counts / spam_counts.sum()) - np.log(ham_counts / ham_counts.sum())
    names = vec.get_feature_names_out()
    order = np.argsort(logodds)
    return {
        "spam_words": [{"word": names[i], "weight": round(float(logodds[i]), 3)}
                       for i in order[-n:][::-1]],
        "ham_words": [{"word": names[i], "weight": round(float(logodds[i]), 3)}
                      for i in order[:n]],
    }


def word_contributions(pipeline, text):
    """{word_feature: contribution} for features present in this one message.
    Contribution = feature value x model weight (linear models only)."""
    coef = _get_coef(pipeline)
    names = _feature_names(pipeline)
    if coef is None or names is None:
        return {}
    X = pipeline.named_steps["features"].transform(pd.DataFrame({"text": [text]}))
    if hasattr(X, "toarray"):
        X = X.toarray()
    row = np.asarray(X)[0]
    out = {}
    for name, value, w in zip(names, row, coef):
        if value != 0 and name.startswith("word__"):
            out[name.split("__", 1)[-1]] = float(value * w)
    return out


def lime_figure(pipeline, text, path):
    """Save a LIME explanation figure for one message. Returns True/False."""
    try:
        from lime.lime_text import LimeTextExplainer
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def predict_proba(texts):
            return pipeline.predict_proba(pd.DataFrame({"text": list(texts)}))

        explainer = LimeTextExplainer(class_names=["ham", "spam"])
        exp = explainer.explain_instance(text, predict_proba, num_features=10)
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        return False
