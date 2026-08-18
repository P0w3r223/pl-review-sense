"""TF-IDF + logistic-regression sentiment baseline. Pure scikit-learn; persistence via joblib.

The vectorizer lives inside the Pipeline, so it is fit on the training fold only — no leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config

# The pipeline's step names, named once. `interpret` reaches into the fitted pipeline by these
# keys, and a rename here would otherwise fail there at runtime rather than at import.
VECTORIZER_STEP = "tfidf"
CLASSIFIER_STEP = "clf"


def build_pipeline() -> Pipeline:
    """TF-IDF (word n-grams) -> multinomial logistic regression. Nothing is fit here."""
    return Pipeline(
        [
            (
                VECTORIZER_STEP,
                TfidfVectorizer(
                    ngram_range=config.TFIDF_NGRAM_RANGE,
                    max_features=config.TFIDF_MAX_FEATURES,
                    min_df=config.TFIDF_MIN_DF,
                    sublinear_tf=config.TFIDF_SUBLINEAR_TF,
                ),
            ),
            (
                CLASSIFIER_STEP,
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


def save(pipe: Pipeline, path: Optional[Path] = None) -> Path:
    """Persist the fitted pipeline. The destination is resolved when called, not at import.

    A default of ``config.BASELINE_MODEL_PATH`` in the signature is evaluated once, when this
    module is first imported, and binds the real path into the function forever: redirecting
    ``config`` afterwards has no effect, and anything calling ``save(pipe)`` writes over the
    trained model in ``models/`` whatever it was configured to do. A test doing exactly that
    is what surfaced this.
    """
    path = path or config.BASELINE_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    return path


def load(path: Optional[Path] = None) -> Pipeline:
    """Load the fitted pipeline, resolving the default at call time — see ``save``."""
    return joblib.load(path or config.BASELINE_MODEL_PATH)
