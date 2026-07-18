"""HerBERT fine-tuning for three-class Polish sentiment.

Heavy deps (torch/transformers) are imported lazily, so importing this module — and the rest
of the package — works without them; CI never installs them.

Real training runs on GPU (Colab: ``notebooks/herbert_colab.ipynb``). The local ``--smoke``
mode fine-tunes on a tiny subset for one epoch **only to prove the code path runs**; its
numbers are not representative and are written to a separate ``herbert_smoke.json`` — never to
the headline ``herbert.json``.

Run:
    python -m pl_review_sense.herbert --smoke     # local CPU code-path check
    python -m pl_review_sense.herbert             # full run (needs a GPU)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from . import config, evaluate
from .data import Dataset, Split, load_polemo


@dataclass(frozen=True)
class TrainArgs:
    epochs: int
    batch_size: int
    max_len: int
    subset: Optional[int]  # cap the training set (smoke); None = full
    output_dir: str


def smoke_args() -> TrainArgs:
    return TrainArgs(
        epochs=config.SMOKE_EPOCHS,
        batch_size=config.SMOKE_BATCH_SIZE,
        max_len=config.SMOKE_MAX_LEN,
        subset=config.SMOKE_SUBSET,
        output_dir=str(config.MODELS_DIR / "herbert_smoke"),
    )


def full_args() -> TrainArgs:
    return TrainArgs(
        epochs=config.HERBERT_EPOCHS,
        batch_size=config.HERBERT_BATCH_SIZE,
        max_len=config.HERBERT_MAX_LEN,
        subset=None,
        output_dir=str(config.MODELS_DIR / "herbert"),
    )


def _require_transformers():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise SystemExit(
            "HerBERT needs the transformer extra: pip install -e .[transformer]"
        ) from exc


def fine_tune(data: Dataset, args: TrainArgs) -> Tuple[evaluate.EvalResult, List[int]]:
    """Fine-tune HerBERT and evaluate on the test split. Returns (metrics, predictions)."""
    _require_transformers()
    import numpy as np
    import torch
    from torch.utils.data import Dataset as TorchDataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    train_split = data.train
    if args.subset is not None:
        train_split = Split(train_split.texts[: args.subset], train_split.labels[: args.subset])

    tokenizer = AutoTokenizer.from_pretrained(config.HERBERT_MODEL)

    def encode(split: Split):
        enc = tokenizer(split.texts, truncation=True, padding=True, max_length=args.max_len)

        class _Encoded(TorchDataset):
            def __len__(self) -> int:
                return len(split.labels)

            def __getitem__(self, i):
                item = {k: torch.tensor(v[i]) for k, v in enc.items()}
                item["labels"] = torch.tensor(split.labels[i])
                return item

        return _Encoded()

    model = AutoModelForSequenceClassification.from_pretrained(
        config.HERBERT_MODEL, num_labels=len(config.LABEL_NAMES)
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=config.HERBERT_LR,
        weight_decay=config.HERBERT_WEIGHT_DECAY,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=config.RANDOM_STATE,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=encode(train_split))
    trainer.train()

    predictions = trainer.predict(encode(data.test))
    y_pred = np.argmax(predictions.predictions, axis=1).astype(int).tolist()
    return evaluate.evaluate(data.test.labels, y_pred), y_pred


def write_metrics(result: evaluate.EvalResult, run: str) -> None:
    """Write HerBERT metrics. A representative GPU run -> herbert.json; smoke -> herbert_smoke.json."""
    representative = run == "gpu"
    payload = {
        "model": "herbert",
        "run": run,
        "representative": representative,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "per_class": [asdict(c) for c in result.per_class],
        "confusion": result.confusion,
        "labels": list(config.LABEL_NAMES),
    }
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.HERBERT_METRICS_PATH if representative else config.METRICS_DIR / "herbert_smoke.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pl_review_sense.herbert", description="Fine-tune HerBERT.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny CPU run to validate the training code (non-representative numbers)",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    train_args = smoke_args() if args.smoke else full_args()
    run = "cpu-smoke" if args.smoke else "gpu"

    data = load_polemo()
    result, _ = fine_tune(data, train_args)
    print(f"[{run}] accuracy={result.accuracy:.4f}  macro_f1={result.macro_f1:.4f}")
    if args.smoke:
        print("NOTE: smoke numbers are NOT representative — run notebooks/herbert_colab.ipynb on a GPU.")
    write_metrics(result, "cpu-smoke" if args.smoke else "gpu")


if __name__ == "__main__":
    main()
