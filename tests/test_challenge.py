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


def test_introduce_typo_keeps_the_punctuation_stuck_to_the_word_it_corrupts():
    mistyped = challenge.introduce_typo("Nie polecam, sprzęt zepsuł się szybko.")

    assert mistyped.endswith(".")
    assert "," in mistyped
    assert sorted(mistyped) == sorted("Nie polecam, sprzęt zepsuł się szybko.")


def test_introduce_typo_corrupts_the_longest_word_not_the_first_one():
    original = "ok bardzo dobre urządzenie"
    mistyped = challenge.introduce_typo(original)

    changed = [
        (before, after)
        for before, after in zip(original.split(" "), mistyped.split(" "))
        if before != after
    ]
    assert len(changed) == 1
    assert changed[0][0] == "urządzenie"


def test_a_word_of_one_repeated_letter_is_left_alone_rather_than_silently_unchanged():
    """Swapping two identical letters is not a typo; the case is dropped from the variant set."""
    assert challenge.introduce_typo("aaaa bb") == "aaaa bb"
    assert challenge.derive_variants([challenge.Case("aaaa bb", 0, challenge.PLAIN)]) == []


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


def test_score_lists_the_hand_written_cells_before_the_derived_ones():
    """The order the page draws its bars in: the control cell first, the re-spellings last."""
    cases = [
        challenge.Case("a", 0, challenge.TYPOS),
        challenge.Case("b", 0, challenge.SARCASM),
        challenge.Case("c", 0, challenge.PLAIN),
        challenge.Case("d", 0, challenge.NO_DIACRITICS),
        challenge.Case("e", 0, challenge.NEGATION),
        challenge.Case("f", 0, challenge.CONTRAST),
    ]

    scores = challenge.score(cases, [0] * len(cases))

    assert [item.phenomenon for item in scores] == [
        challenge.PLAIN,
        challenge.NEGATION,
        challenge.SARCASM,
        challenge.CONTRAST,
        challenge.NO_DIACRITICS,
        challenge.TYPOS,
    ]


def test_score_carries_the_note_explaining_what_each_cell_is_for():
    scores = challenge.score([challenge.Case("a", 0, challenge.SARCASM)], [0])
    assert scores[0].note == challenge.PHENOMENON_NOTES[challenge.SARCASM]


def test_score_refuses_predictions_that_do_not_line_up():
    with pytest.raises(ValueError):
        challenge.score(list(challenge_set.CASES), [0])
