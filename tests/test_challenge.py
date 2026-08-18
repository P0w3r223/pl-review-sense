"""The challenge set and its derived variants.

The corpus itself is tested as much as the code around it: a probe cell that has drifted out
of balance measures the model's prior for one class instead of the phenomenon it is named
after, and nothing downstream would notice.
"""

from __future__ import annotations

import pytest

from pl_review_sense import challenge, challenge_set, config

BASE_CELLS = (challenge.PLAIN, challenge.NEGATION, challenge.SARCASM, challenge.CONTRAST)


def test_every_case_carries_a_known_label_and_cell():
    for case in challenge_set.CASES:
        assert case.phenomenon in BASE_CELLS
        assert case.label in range(len(config.LABEL_NAMES))
        assert case.text.strip() == case.text


def test_no_case_is_written_twice():
    texts = [case.text for case in challenge_set.CASES]
    assert len(set(texts)) == len(texts)


@pytest.mark.parametrize("cell", BASE_CELLS)
def test_each_cell_is_large_enough_to_report_as_a_rate(cell):
    cases = [case for case in challenge_set.CASES if case.phenomenon == cell]
    assert len(cases) >= config.MIN_PHENOMENON_N


@pytest.mark.parametrize("cell", BASE_CELLS)
def test_no_single_answer_can_win_a_cell(cell):
    """A cell of one label would be scored perfectly by a model that always answers it."""
    labels = [case.label for case in challenge_set.CASES if case.phenomenon == cell]
    majority = max(labels.count(label) for label in set(labels))
    assert majority <= len(labels) * 0.6


def test_strip_diacritics_covers_every_polish_letter():
    assert challenge.strip_diacritics("ąćęłńóśźż ĄĆĘŁŃÓŚŹŻ") == "acelnoszz ACELNOSZZ"


def test_strip_diacritics_leaves_plain_text_alone():
    assert challenge.strip_diacritics("Hotel bez windy.") == "Hotel bez windy."


def test_introduce_typo_swaps_two_letters_of_the_longest_word():
    original = "Ten produkt jest ok"
    mistyped = challenge.introduce_typo(original)

    assert mistyped != original
    assert len(mistyped) == len(original)
    assert sorted(mistyped) == sorted(original), "a swap moves letters, it does not add them"


def test_introduce_typo_is_deterministic():
    text = challenge_set.CASES[0].text
    assert challenge.introduce_typo(text) == challenge.introduce_typo(text)


def test_introduce_typo_leaves_text_without_a_long_word_alone():
    assert challenge.introduce_typo("a to i on") == "a to i on"


def test_derived_variants_keep_the_gold_label():
    cases = list(challenge_set.CASES)
    variants = challenge.derive_variants(cases)

    assert {variant.phenomenon for variant in variants} == {
        challenge.NO_DIACRITICS,
        challenge.TYPOS,
    }
    for variant in variants:
        assert variant.label in range(len(config.LABEL_NAMES))
    # Every case yields a typo variant; only those with diacritics yield a stripped one.
    typos = [v for v in variants if v.phenomenon == challenge.TYPOS]
    assert len(typos) == len(cases)


def test_score_counts_per_cell_and_flags_thin_ones():
    cases = [
        challenge.Case("a", 0, challenge.PLAIN),
        challenge.Case("b", 1, challenge.PLAIN),
        challenge.Case("c", 2, challenge.SARCASM),
    ]

    scores = challenge.score(cases, [0, 2, 2])

    by_cell = {item.phenomenon: item for item in scores}
    assert (by_cell[challenge.PLAIN].correct, by_cell[challenge.PLAIN].total) == (1, 2)
    assert by_cell[challenge.SARCASM].correct == 1
    assert by_cell[challenge.PLAIN].thin, "two cases cannot support a rate"
    assert by_cell[challenge.PLAIN].share == pytest.approx(0.5)


def test_score_refuses_predictions_that_do_not_line_up():
    with pytest.raises(ValueError):
        challenge.score(list(challenge_set.CASES), [0])
