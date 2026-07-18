from pl_review_sense import evaluate


def test_perfect_prediction_scores_one():
    y = [0, 1, 2, 0, 1, 2]
    result = evaluate.evaluate(y, y)
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.confusion == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_confusion_matrix_orientation_is_true_by_pred():
    # One true-negative(0) predicted as positive(2): row 0, col 2 must be 1.
    y_true = [0, 1, 2]
    y_pred = [2, 1, 2]
    result = evaluate.evaluate(y_true, y_pred)
    assert result.confusion[0][2] == 1
    assert result.accuracy == 2 / 3


def test_macro_f1_ignores_class_imbalance():
    # Predicting the majority class always -> high accuracy but low macro-F1.
    y_true = [0, 0, 0, 0, 1, 2]
    y_pred = [0, 0, 0, 0, 0, 0]
    result = evaluate.evaluate(y_true, y_pred)
    assert result.accuracy == 4 / 6
    assert result.macro_f1 < result.accuracy


def test_has_negation_detects_polish_cues():
    assert evaluate.has_negation("To nie jest dobry produkt")
    assert evaluate.has_negation("bez sensu, brak jakości")
    assert not evaluate.has_negation("Świetny produkt, polecam")


def test_misclassified_sorts_by_length_and_flags_negation():
    texts = ["nie polecam tego bardzo słabego produktu naprawdę", "ok", "dobry"]
    y_true = [2, 0, 2]
    y_pred = [0, 1, 2]  # first two wrong, third correct
    errors = evaluate.misclassified(texts, y_true, y_pred, limit=10)
    assert len(errors) == 2
    assert errors[0].length >= errors[1].length  # longest first
    assert errors[0].negation is True
    assert errors[0].true_label == "positive" and errors[0].pred_label == "negative"


def test_negation_error_share():
    errors = evaluate.misclassified(
        ["nie dobre wcale", "spoko produkt fajny"], [2, 0], [0, 2], limit=10
    )
    assert 0.0 <= evaluate.negation_error_share(errors) <= 1.0
