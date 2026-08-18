"""The challenge set: what a sentence has to contain for a bag of words to slip on it.

The report used to *assert* that negation and sarcasm are where a lexical model fails, and
illustrate the claim with three sentences nobody had scored. This module turns the assertion
into a measurement.

Every case is written here, by us, in Polish. None of it comes from PolEmo: the corpus is
CC BY-NC-SA and is not redistributed by this repository, and a probe made of training-domain
excerpts would in any case be measuring memorisation rather than the phenomenon.

Two design decisions carry the whole thing:

**Surface-matched pairs.** A sarcasm probe made only of ironic praise measures the model's
prior for "negative", not its grasp of irony — a classifier that answers "negative" to every
sentence scores a perfect 100% on it. Each ironic case therefore has a genuine counterpart
in the same enthusiastic register, down to minimal pairs ("bateria wytrzymała całe trzy
godziny" against "całe trzy dni"), so the cell can only be won by telling them apart.

**Derived variants, not hand-written ones.** Robustness to missing diacritics and to typos is
measured by transforming the same 80 cases mechanically. A hand-written "typo set" would
differ from the base set in content as well as in spelling, and the comparison would no
longer isolate the spelling.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Sequence

from . import config

# Phenomenon identifiers. Strings rather than an enum because they travel through JSON to the
# page, and one spelling for the whole journey is worth more than the type.
PLAIN = "plain"
NEGATION = "negation"
SARCASM = "sarcasm"
CONTRAST = "contrast"
NO_DIACRITICS = "no-diacritics"
TYPOS = "typos"

# What each cell of the probe is for, in the page's own words.
PHENOMENON_NOTES: Dict[str, str] = {
    PLAIN: "unambiguous sentiment, no negation, irony or pivot — the control cell",
    NEGATION: "sentiment carried by a negation cue, in both directions",
    SARCASM: "ironic praise against genuine praise in the same register",
    CONTRAST: "a pivot (ale / jednak / mimo) after which the sentiment lands",
    NO_DIACRITICS: "every base case with Polish diacritics stripped",
    TYPOS: "every base case with one adjacent-character swap in its longest word",
}


@dataclass(frozen=True)
class Case:
    """One probe sentence and the label a Polish reader would give it."""

    text: str
    label: int  # 0 negative, 1 neutral, 2 positive
    phenomenon: str


@dataclass(frozen=True)
class PhenomenonScore:
    phenomenon: str
    correct: int
    total: int
    note: str

    @property
    def share(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def thin(self) -> bool:
        """Too few cases to read the share as a rate rather than as a count."""
        return self.total < config.MIN_PHENOMENON_N


def strip_diacritics(text: str) -> str:
    """Remove Polish diacritics the way a hurried reviewer does: ą→a, ł→l, ż→z.

    Decomposition handles every accented letter except ``ł``/``Ł``, which carries its stroke
    inside the code point rather than as a combining mark and therefore survives NFD intact.
    """
    without_stroke = text.replace("ł", "l").replace("Ł", "L")
    decomposed = unicodedata.normalize("NFD", without_stroke)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def introduce_typo(text: str) -> str:
    """Swap two adjacent characters in the longest alphabetic word.

    Deterministic on purpose: a randomly corrupted probe would score differently on every
    run, and a page rebuilt from committed numbers has to reproduce them exactly. The
    longest word is chosen because it is the one carrying the sentiment often enough for
    the swap to matter — corrupting "i" would measure nothing.
    """
    words = text.split(" ")
    candidates = [
        index for index, word in enumerate(words) if len(_letters(word)) >= 4
    ]
    if not candidates:
        return text
    target = max(candidates, key=lambda index: (len(_letters(words[index])), -index))
    word = words[target]
    letters = _letters(word)
    position = _swappable_position(letters)
    if position is None:
        return text
    start = word.index(letters)
    swapped = (
        letters[: position - 1]
        + letters[position]
        + letters[position - 1]
        + letters[position + 1 :]
    )
    words[target] = word[:start] + swapped + word[start + len(letters) :]
    return " ".join(words)


def _swappable_position(letters: str) -> int | None:
    """Index of the second character in the pair to swap, nearest the middle of the word.

    Swapping two identical characters is not a typo — it leaves the word untouched and would
    silently drop that sentence from the variant set. Positions are therefore tried outward
    from the middle until a pair actually differs.
    """
    middle = len(letters) // 2
    for offset in range(len(letters)):
        for position in (middle + offset, middle - offset):
            if 1 <= position < len(letters) and letters[position - 1] != letters[position]:
                return position
    return None


def _letters(word: str) -> str:
    """The alphabetic core of a token, without the punctuation stuck to its ends."""
    return word.strip(".,!?;:\"'()—–-")


def derive_variants(cases: Sequence[Case]) -> List[Case]:
    """The base cases re-spelled two ways, keeping their labels.

    A stripped or mistyped sentence means exactly what it meant before — that is the point
    of the comparison — so the gold label travels with the text unchanged.
    """
    variants: List[Case] = []
    for case in cases:
        stripped = strip_diacritics(case.text)
        if stripped != case.text:
            variants.append(Case(text=stripped, label=case.label, phenomenon=NO_DIACRITICS))
        mistyped = introduce_typo(case.text)
        if mistyped != case.text:
            variants.append(Case(text=mistyped, label=case.label, phenomenon=TYPOS))
    return variants


def score(cases: Sequence[Case], predictions: Sequence[int]) -> List[PhenomenonScore]:
    """Correct answers per phenomenon, as a count over its own n.

    Reported as ``k of n`` rather than as a percentage: a cell holds twenty sentences, and a
    percentage invites the reader to compare it with the 684-row test figures on the same
    page as if the two were the same kind of number.
    """
    if len(cases) != len(predictions):
        raise ValueError(f"{len(cases)} cases against {len(predictions)} predictions")

    totals: Dict[str, int] = {}
    hits: Dict[str, int] = {}
    for case, predicted in zip(cases, predictions):
        totals[case.phenomenon] = totals.get(case.phenomenon, 0) + 1
        hits[case.phenomenon] = hits.get(case.phenomenon, 0) + int(predicted == case.label)

    order = [PLAIN, NEGATION, SARCASM, CONTRAST, NO_DIACRITICS, TYPOS]
    ranked = sorted(totals, key=lambda name: order.index(name) if name in order else len(order))
    return [
        PhenomenonScore(
            phenomenon=name,
            correct=hits[name],
            total=totals[name],
            note=PHENOMENON_NOTES.get(name, ""),
        )
        for name in ranked
    ]
