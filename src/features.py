"""
src/features.py
---------------
Text cleaning + feature engineering.

Cleaning philosophy: spam signals hide in the *shape* of a message, so we
do NOT blindly delete numbers/URLs -- we replace them with tokens. A
message full of "urltoken" and "phonetoken" is itself a spam signal the
model can learn, while the actual digits (which never repeat) would be
useless vocabulary.

Features = three blocks, combined in ONE sklearn Pipeline so the
vectorizers/scalers are fitted on training data only (no leakage):
  1. word TF-IDF (1-2 grams)
  2. character TF-IDF (3-5 grams) -- catches obfuscation like "fr33"
  3. handcrafted numeric features from the RAW text
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{6,}\d)")
_NUM_RE = re.compile(r"\d+")
_NONWORD_RE = re.compile(r"[^a-z0-9\s!?\u20B9$£€%]")   # keep meaning-bearing symbols
_WS_RE = re.compile(r"\s+")


def clean_text(text):
    """Lowercase + tokenise URLs/emails/phones/numbers (in that order)."""
    t = str(text).lower()
    t = _URL_RE.sub(" urltoken ", t)
    t = _EMAIL_RE.sub(" emailtoken ", t)
    t = _PHONE_RE.sub(" phonetoken ", t)
    t = _NUM_RE.sub(" numtoken ", t)
    t = _NONWORD_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


HANDCRAFTED_NAMES = [
    "char_length", "word_count", "uppercase_ratio", "digit_ratio",
    "url_count", "exclaim_count", "special_char_count", "has_currency",
]


def handcrafted_matrix(texts):
    rows = []
    for text in texts:
        t = str(text)
        n = max(len(t), 1)
        letters = [c for c in t if c.isalpha()]
        upper = sum(1 for c in letters if c.isupper())
        digits = sum(1 for c in t if c.isdigit())
        urls = len(_URL_RE.findall(t))
        special = sum(1 for c in t if not c.isalnum() and not c.isspace())
        rows.append([
            len(t),
            len(t.split()),
            upper / max(len(letters), 1),
            digits / n,
            urls,
            t.count("!"),
            special,
            1.0 if any(sym in t for sym in ("\u20B9", "$", "£", "€", "rs", "Rs", "RS")) else 0.0,
        ])
    return np.asarray(rows, dtype=float)


class CleanText(BaseEstimator, TransformerMixin):
    """DataFrame with a 'text' column -> cleaned text Series."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = X["text"] if isinstance(X, pd.DataFrame) else pd.Series(X)
        return texts.astype(str).map(clean_text)


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """DataFrame with a 'text' column -> handcrafted numeric matrix (raw text)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = X["text"] if isinstance(X, pd.DataFrame) else pd.Series(X)
        return handcrafted_matrix(texts.astype(str).tolist())

    def get_feature_names_out(self, input_features=None):
        return np.asarray(HANDCRAFTED_NAMES)


def build_linear_features(scaler="standard"):
    """Word + char TF-IDF + handcrafted. For NB use scaler='maxabs'
    (keeps every feature >= 0, which MultinomialNB requires)."""
    hand_scaler = MaxAbsScaler() if scaler == "maxabs" else StandardScaler(with_mean=False)
    return FeatureUnion([
        ("word", Pipeline([
            ("clean", CleanText()),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=40000,
                                      sublinear_tf=True, min_df=2)),
        ])),
        ("char", Pipeline([
            ("clean", CleanText()),
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                      max_features=30000, sublinear_tf=True,
                                      min_df=2)),
        ])),
        ("hand", Pipeline([
            ("extract", HandcraftedFeatures()),
            ("scale", hand_scaler),
        ])),
    ])


def build_tree_features(n_components=150):
    """Compact dense features for tree models: word TF-IDF compressed with
    SVD + handcrafted features. Trees do not suit 70k sparse dimensions."""
    return FeatureUnion([
        ("word_svd", Pipeline([
            ("clean", CleanText()),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 1), max_features=20000,
                                      sublinear_tf=True, min_df=2)),
            ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
        ])),
        ("hand", Pipeline([
            ("extract", HandcraftedFeatures()),
            ("scale", StandardScaler()),
        ])),
    ])
