"""Load PolEmo 2.0 and reduce it to a three-class sentiment task.

Only ``load_polemo`` performs I/O (the dataset download); label filtering/mapping are pure
functions so they are unit-tested without touching the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from . import config


@dataclass(frozen=True)
class Split:
    texts: List[str]
    labels: List[int]  # 0 = negative, 1 = neutral, 2 = positive

    def __len__(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class Dataset:
    train: Split
    validation: Split
    test: Split


def label_name(index: int) -> str:
    """Human-readable name for a label index."""
    return config.LABEL_NAMES[index]


def map_target_name(name: str) -> Optional[int]:
    """Map a PolEmo class name to our 3-class label, or None to drop it (ambiguous/unknown)."""
    if name == config.DROP_LABEL_NAME:
        return None
    return config.POLEMO_NAME_TO_LABEL.get(name)


def to_split(texts: Sequence[str], target_names: Sequence[str]) -> Split:
    """Pure: pair texts with mapped labels, dropping ambiguous/unknown rows."""
    keep_texts: List[str] = []
    keep_labels: List[int] = []
    for text, name in zip(texts, target_names):
        label = map_target_name(name)
        if label is None:
            continue
        keep_texts.append(text)
        keep_labels.append(label)
    return Split(keep_texts, keep_labels)


def load_polemo() -> Dataset:
    """Download PolEmo (cached by ``datasets``) and build the three-class splits. Only I/O here."""
    from datasets import load_dataset

    raw = load_dataset(
        config.DATASET,
        config.DATASET_CONFIG,
        revision=config.DATASET_REVISION,
        trust_remote_code=config.TRUST_REMOTE_CODE,
    )
    splits = {}
    for name in ("train", "validation", "test"):
        part = raw[name]
        int2str = part.features[config.TARGET_COLUMN].int2str
        target_names = [int2str(t) for t in part[config.TARGET_COLUMN]]
        splits[name] = to_split(part[config.TEXT_COLUMN], target_names)

    dataset = Dataset(splits["train"], splits["validation"], splits["test"])
    # Fail loudly at the data boundary if the mapping produced nothing or the wrong labels
    # (e.g. upstream renamed the classes), instead of a confusing model error much later.
    expected = set(range(len(config.LABEL_NAMES)))
    if len(dataset.train) == 0 or set(dataset.train.labels) != expected:
        raise ValueError(
            f"PolEmo label mapping yielded {sorted(set(dataset.train.labels))}; "
            f"expected {sorted(expected)} — check the dataset revision/label names."
        )
    return dataset
