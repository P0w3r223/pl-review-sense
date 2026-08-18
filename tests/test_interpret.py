"""Extracting the model's own reasons from its coefficients."""

from __future__ import annotations

import pytest

from pl_review_sense import baseline, interpret


def test_top_terms_ranks_by_weight_within_each_class():
    ranked = interpret.top_terms(
        feature_names=["nie polecam", "polecam", "hotel"],
        coefficients=[[3.0, -1.0, 0.5], [-2.0, -2.0, 1.5], [-1.0, 4.0, -0.5]],
        top_n=2,
        labels=("negative", "neutral", "positive"),
    )

    assert [term.term for term in ranked["negative"]] == ["nie polecam", "hotel"]
    assert [term.term for term in ranked["positive"]] == ["polecam"]
    assert ranked["negative"][0].weight == pytest.approx(3.0)


def test_top_terms_drops_negative_weights_rather_than_padding_the_list():
    ranked = interpret.top_terms(
        feature_names=["a", "b"], coefficients=[[-1.0, -2.0]], top_n=2, labels=("negative",)
    )
    assert ranked["negative"] == []


def test_top_terms_breaks_ties_by_term_so_the_page_does_not_move_between_runs():
    first = interpret.top_terms(
        feature_names=["zeta", "alfa"], coefficients=[[1.0, 1.0]], top_n=2, labels=("negative",)
    )
    assert [term.term for term in first["negative"]] == ["alfa", "zeta"]


def test_top_terms_refuses_shapes_that_do_not_line_up():
    with pytest.raises(ValueError):
        interpret.top_terms(["a", "b"], [[1.0]], labels=("negative",))
    with pytest.raises(ValueError):
        interpret.top_terms(["a"], [[1.0], [2.0]], labels=("negative",))


def test_from_pipeline_reads_a_fitted_baseline():
    texts = [
        "fatalny hotel, nie polecam nikomu",
        "okropna obsługa, nie polecam",
        "hotel ma parking i recepcję",
        "hotel ma windę i recepcję",
        "świetny hotel, gorąco polecam",
        "polecam, obsługa wyjątkowa",
    ]
    labels = [0, 0, 1, 1, 2, 2]

    ranked = interpret.from_pipeline(baseline.train(texts, labels), top_n=5)

    assert set(ranked) == {"negative", "neutral", "positive"}
    assert all(len(terms) <= 5 for terms in ranked.values())
    assert any(term.term == "nie polecam" for term in ranked["negative"])
