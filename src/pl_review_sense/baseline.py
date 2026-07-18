"""TF-IDF + logistic-regression sentiment baseline. Pure scikit-learn; persistence via joblib.

The vectorizer lives inside the Pipeline, so it is fit on the training fold only — no leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config


def build_pipeline() -> Pipeline:
    """TF-IDF (word n-grams) -> multinomial logistic regression. Nothing is fit here."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=config.TFIDF_NGRAM_RANGE,
                    max_features=config.TFIDF_MAX_FEATURES,
                    min_df=config.TFIDF_MIN_DF,
                    sublinear_tf=config.TFIDF_SUBLINEAR_TF,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=config.LOGREG_C,
                    max_iter=config.LOGREG_MAX_ITER,
                    class_weight=config.CLASS_WEIGHT,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )


def train(texts: Sequence[str], labels: Sequence[int]) -> Pipeline:
    pipe = build_pipeline()
    pipe.fit(list(texts), list(labels))
    return pipe


def predict(pipe: Pipeline, texts: Sequence[str]) -> List[int]:
    return [int(p) for p in pipe.predict(list(texts))]


def predict_proba(pipe: Pipeline, texts: Sequence[str]) -> List[List[float]]:
    return [[float(x) for x in row] for row in pipe.predict_proba(list(texts))]


def save(pipe: Pipeline, path: Path = config.BASELINE_MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    return path


def load(path: Path = config.BASELINE_MODEL_PATH) -> Pipeline:
    return joblib.load(path)
