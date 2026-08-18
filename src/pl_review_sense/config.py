"""Central configuration: dataset, label scheme, paths, model settings. No I/O — only constants."""

from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"

PUBLISH_DIR = PROJECT_ROOT / "docs"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_tfidf_logreg.joblib"
BASELINE_METRICS_PATH = METRICS_DIR / "baseline.json"
HERBERT_METRICS_PATH = METRICS_DIR / "herbert.json"
BASELINE_PREDICTIONS_PATH = PREDICTIONS_DIR / "baseline_test.json"
HERBERT_PREDICTIONS_PATH = PREDICTIONS_DIR / "herbert_test.json"

# One file per question the page asks, so a panel whose evidence has not been produced yet
# is missing on its own rather than blocking every other panel.
MANIFEST_PATH = METRICS_DIR / "manifest.json"
SIGNIFICANCE_PATH = METRICS_DIR / "significance.json"
SEGMENTS_PATH = METRICS_DIR / "segments.json"
LEARNING_CURVE_PATH = METRICS_DIR / "learning_curve.json"
CHALLENGE_PATH = METRICS_DIR / "challenge.json"
DEFERRAL_PATH = METRICS_DIR / "deferral.json"
INTERPRETABILITY_PATH = METRICS_DIR / "interpretability.json"
COST_PATH = METRICS_DIR / "cost.json"

# --- Dataset -----------------------------------------------------------------
# PolEmo 2.0 (CLARIN-PL). Loaded through a dataset script -> requires datasets < 3.
# Original ClassLabel names: 'zero' (neutral), 'minus' (negative), 'plus' (positive),
# 'amb' (ambiguous). We keep three classes and drop 'amb'.
DATASET = "clarin-pl/polemo2-official"
DATASET_CONFIG = "all_text"  # full reviews across all four domains
TRUST_REMOTE_CODE = True
# Pin to a known-good commit so trust_remote_code executes an audited script, not whatever
# happens to be the upstream HEAD at download time.
DATASET_REVISION = "802e35d2b12bae84bb07911d841e8f046dc2fcef"
TEXT_COLUMN = "text"
TARGET_COLUMN = "target"
DROP_LABEL_NAME = "amb"

# Our ordinal 3-class scheme.
LABEL_NAMES = ("negative", "neutral", "positive")
# Map the PolEmo ClassLabel *name* to our integer label ('amb' is excluded upstream).
POLEMO_NAME_TO_LABEL = {"minus": 0, "zero": 1, "plus": 2}

# --- Baseline (TF-IDF + logistic regression) ---------------------------------
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 50_000
TFIDF_MIN_DF = 2
TFIDF_SUBLINEAR_TF = True
LOGREG_C = 1.0
LOGREG_MAX_ITER = 1000
# Classes are imbalanced -> weight them; and macro-F1 is the headline metric, not accuracy.
CLASS_WEIGHT = "balanced"

# --- HerBERT fine-tuning -----------------------------------------------------
HERBERT_MODEL = "allegro/herbert-base-cased"
HERBERT_MAX_LEN = 256
HERBERT_EPOCHS = 4
HERBERT_BATCH_SIZE = 16
HERBERT_LR = 2e-5
HERBERT_WEIGHT_DECAY = 0.01

# CPU smoke run — proves the training code path works end-to-end; NOT representative.
SMOKE_SUBSET = 200
SMOKE_EPOCHS = 1
SMOKE_BATCH_SIZE = 8
SMOKE_MAX_LEN = 128

# --- Uncertainty around a score ----------------------------------------------
# A single macro-F1 on 684 test rows is a point estimate; the interval is what says whether
# two models are actually apart. Percentile bootstrap over the test rows.
BOOTSTRAP_RESAMPLES = 2_000
CONFIDENCE_LEVEL = 0.95
# Where the paired test stops calling a difference chance. Conventional, and stated once so the
# page and any later analysis cannot disagree about what "significant" meant.
SIGNIFICANCE_LEVEL = 0.05

# --- Learning curve ----------------------------------------------------------
# Absolute training sizes, because "how many labelled reviews do I need" is the question a
# reader brings to this chart — a fraction of a corpus they do not have answers nothing.
LEARNING_CURVE_SIZES = (150, 300, 600, 1_200, 2_400, 4_800)
# Several draws per size: one subsample measures that draw's luck, not the size.
LEARNING_CURVE_SEEDS = (0, 1, 2, 3, 4)

# --- Calibration and deferral ------------------------------------------------
CALIBRATION_BINS = 10
# How far the average confidence may sit from the average accuracy before the page calls the
# model over- or under-confident rather than roughly honest.
CALIBRATION_TOLERANCE = 0.02
# Shares of the test set handed onward, lowest confidence first. 0.0 is kept deliberately:
# it is the no-deferral reference every other row is read against.
DEFERRAL_RATES = (0.0, 0.05, 0.10, 0.20, 0.30)

# --- Length segments ---------------------------------------------------------
# Word counts. The page claims the corpus score does not carry over to short reviews; these
# buckets ask the corpus itself, where the sentences are nobody's choice but PolEmo's.
LENGTH_BUCKET_EDGES = (25, 50, 100, 200)
# Below this many test rows a segment is drawn muted and read as a direction, not a score.
# Fifty rather than thirty: a three-class macro-F1 over thirty reviews moves with which class
# the rows happened to be. The 25–49-word bucket is the demonstration — 90% accuracy and 0.62
# macro-F1 in the same thirty rows, because one class contributed a handful of them.
MIN_SEGMENT_N = 50

# --- Challenge set -----------------------------------------------------------
# Below this many cases a phenomenon is reported as a count, never as a percentage: eight
# sentences cannot support a rate, and rounding them into one invites exactly that reading.
MIN_PHENOMENON_N = 20

# --- Interpretability --------------------------------------------------------
TOP_TERMS_PER_CLASS = 10

# --- Misc --------------------------------------------------------------------
RANDOM_STATE = 42
TOP_N_ERRORS = 15
