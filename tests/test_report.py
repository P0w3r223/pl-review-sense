from pl_review_sense.report import generate_report

_BASELINE = {
    "model": "tfidf+logreg",
    "accuracy": 0.94,
    "macro_f1": 0.944,
    "per_class": [
        {"label": "negative", "precision": 0.94, "recall": 0.95, "f1": 0.946, "support": 339},
        {"label": "neutral", "precision": 0.99, "recall": 0.95, "f1": 0.970, "support": 118},
        {"label": "positive", "precision": 0.92, "recall": 0.92, "f1": 0.916, "support": 227},
    ],
    "confusion": [[323, 0, 16], [3, 112, 3], [18, 1, 208]],
    "labels": ["negative", "neutral", "positive"],
}


def test_report_renders_with_baseline_and_pending_herbert(tmp_path):
    out = generate_report(
        output_path=tmp_path / "index.html", baseline_metrics=_BASELINE, herbert_metrics=None
    )
    html = out.read_text(encoding="utf-8")
    assert "pl-review-sense" in html
    assert "0.944" in html  # baseline macro-F1
    assert "pending GPU run" in html  # HerBERT not run yet
    assert "data:image/png;base64," in html  # confusion matrix embedded
    assert "CC BY-NC-SA" in html  # dataset attribution


def test_report_shows_herbert_when_representative(tmp_path):
    herbert = {
        "model": "herbert",
        "representative": True,
        "accuracy": 0.95,
        "macro_f1": 0.952,
        "per_class": _BASELINE["per_class"],
        "confusion": _BASELINE["confusion"],
        "labels": _BASELINE["labels"],
    }
    out = generate_report(
        output_path=tmp_path / "i.html", baseline_metrics=_BASELINE, herbert_metrics=herbert
    )
    html = out.read_text(encoding="utf-8")
    assert "0.952" in html
    assert "pending GPU run" not in html


def test_report_handles_missing_baseline(tmp_path):
    out = generate_report(
        output_path=tmp_path / "i.html", baseline_metrics=None, herbert_metrics=None
    )
    html = out.read_text(encoding="utf-8")
    assert "baseline_train" in html  # placeholder tells you to train first
