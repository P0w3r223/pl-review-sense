"""Build a standalone HTML report into docs/ for GitHub Pages.

Reads committed metrics JSON (numbers only — no PolEmo text is embedded). The HerBERT column
shows "pending GPU run" until a representative ``reports/metrics/herbert.json`` exists. Error
analysis uses aggregate stats plus *illustrative* sentences written here (not dataset excerpts).
"""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import date
from pathlib import Path
from typing import Optional

from . import config

# Illustrative phenomena (our own sentences, NOT from PolEmo) that challenge lexical models.
_ILLUSTRATIVE_ERRORS = [
    ("Nie jest to zły produkt.", "double negation flips the sentiment a bag-of-words misses"),
    ("Świetnie, kolejna rzecz, która zepsuła się po tygodniu.", "sarcasm: positive words, negative meaning"),
    ("Miałem nie polecać, ale ostatecznie jestem zadowolony.", "contrastive clause reverses the leading cue"),
]


# Sentinel so an explicit ``None`` ("no metrics") is distinct from "not provided" (load from disk).
_UNSET = object()


def _load_json(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _confusion_png(confusion, labels) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(confusion)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fmt(metrics: Optional[dict], key: str) -> str:
    if metrics and metrics.get("representative", True):
        return f"{metrics[key]:.3f}"
    return "<em>pending GPU run</em>"


def _per_class_rows(metrics: dict) -> str:
    # Everything interpolated into HTML is escaped. Inputs are trusted today (numbers + config
    # label names), but the report must never embed dataset text, and escaping keeps a future
    # data change from turning into stored XSS on GitHub Pages.
    rows = ""
    for cls in metrics["per_class"]:
        label = html.escape(str(cls["label"]))
        rows += (
            f"<tr><td>{label}</td><td>{cls['precision']:.3f}</td>"
            f"<td>{cls['recall']:.3f}</td><td>{cls['f1']:.3f}</td><td>{cls['support']}</td></tr>"
        )
    return rows


def generate_report(
    output_path: Optional[Path] = None,
    baseline_metrics=_UNSET,
    herbert_metrics=_UNSET,
) -> Path:
    if baseline_metrics is _UNSET:
        baseline_metrics = _load_json(config.BASELINE_METRICS_PATH)
    if herbert_metrics is _UNSET:
        herbert_metrics = _load_json(config.HERBERT_METRICS_PATH)
    output_path = output_path or (config.PROJECT_ROOT / "docs" / "index.html")

    labels = list(config.LABEL_NAMES)

    if baseline_metrics:
        confusion_img = (
            f'<img alt="baseline confusion matrix" '
            f'src="data:image/png;base64,{_confusion_png(baseline_metrics["confusion"], labels)}">'
        )
        per_class = _per_class_rows(baseline_metrics)
    else:
        confusion_img = "<p><em>Run <code>python -m pl_review_sense.baseline_train</code> to populate metrics.</em></p>"
        per_class = '<tr><td colspan="5"><em>no metrics yet</em></td></tr>'

    illustrative = "".join(
        f"<li><code>{html.escape(text)}</code> — {html.escape(why)}</li>"
        for text, why in _ILLUSTRATIVE_ERRORS
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pl-review-sense — TF-IDF vs HerBERT</title>
<style>
  body {{ font: 16px/1.55 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         max-width: 820px; margin: 0 auto; padding: 32px 20px 56px; color: #1c2430; }}
  h1 {{ margin-bottom: 2px; }} .sub {{ color: #667085; margin-top: 0; }}
  .card {{ border: 1px solid #e3e7ee; border-radius: 12px; padding: 18px 20px; margin: 18px 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid #eef1f6; }}
  th:first-child, td:first-child {{ text-align: left; }}
  code {{ background: #eef1f6; padding: 1px 5px; border-radius: 4px; font-size: 0.9em; }}
  img {{ max-width: 100%; }} footer {{ color: #667085; font-size: 0.85rem; margin-top: 24px; }}
  a {{ color: #2563eb; }}
</style>
</head>
<body>
<h1>pl-review-sense</h1>
<p class="sub">Polish review sentiment (3-class) — classic TF-IDF vs HerBERT fine-tuning, on PolEmo 2.0.</p>

<div class="card">
  <h2>Model comparison (test set)</h2>
  <table>
    <tr><th>model</th><th>accuracy</th><th>macro-F1</th></tr>
    <tr><td>TF-IDF + logistic regression</td><td>{_fmt(baseline_metrics, "accuracy")}</td><td>{_fmt(baseline_metrics, "macro_f1")}</td></tr>
    <tr><td>HerBERT (fine-tuned)</td><td>{_fmt(herbert_metrics, "accuracy")}</td><td>{_fmt(herbert_metrics, "macro_f1")}</td></tr>
  </table>
  <p class="sub"><strong>Macro-F1</strong> is the headline metric because the classes are imbalanced.</p>
</div>

<div class="card">
  <h2>Baseline — per class &amp; confusion</h2>
  <table>
    <tr><th>class</th><th>precision</th><th>recall</th><th>F1</th><th>support</th></tr>
    {per_class}
  </table>
  <p>{confusion_img}</p>
</div>

<div class="card">
  <h2>Where models struggle (illustrative)</h2>
  <p class="sub">Own examples of phenomena a bag-of-words model handles poorly — not PolEmo excerpts:</p>
  <ul>{illustrative}</ul>
</div>

<div class="card">
  <h2>Methodology &amp; honest compute cost</h2>
  <ul>
    <li>Three classes: PolEmo's <code>ambiguous</code> dropped; <code>minus/zero/plus</code> → negative/neutral/positive.</li>
    <li>No leakage: TF-IDF fit on train only (inside a Pipeline); untouched test split for metrics.</li>
    <li><strong>Baseline</strong>: trains in seconds on a CPU — cheap, strong, interpretable.</li>
    <li><strong>HerBERT</strong>: fine-tuned on a free Colab GPU (a few minutes/epoch); heavier for a marginal gain.</li>
  </ul>
</div>

<footer>
  Data: <a href="https://clarin-pl.eu/dspace/handle/11321/710">PolEmo 2.0</a> (CLARIN-PL, CC BY-NC-SA 4.0) ·
  <a href="https://github.com/P0w3r223/pl-review-sense">source on GitHub</a> ·
  generated {date.today().isoformat()}
</footer>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    print("wrote", generate_report())
