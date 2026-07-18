from pl_review_sense import baseline


# Tiny synthetic corpus with tokens repeated across docs so min_df=2 keeps a vocabulary.
_TEXTS = [
    "fatalny produkt zły zły",
    "zły okropny fatalny produkt",
    "produkt w porządku neutralny opis",
    "neutralny opis w porządku produkt",
    "świetny produkt polecam świetny",
    "polecam świetny produkt rewelacja",
]
_LABELS = [0, 0, 1, 1, 2, 2]


def test_train_and_predict_shapes():
    pipe = baseline.train(_TEXTS, _LABELS)
    preds = baseline.predict(pipe, _TEXTS)
    assert len(preds) == len(_TEXTS)
    assert set(preds) <= {0, 1, 2}


def test_learns_training_signal():
    # On this separable toy set the pipeline should fit its own training data well.
    pipe = baseline.train(_TEXTS, _LABELS)
    preds = baseline.predict(pipe, _TEXTS)
    correct = sum(a == b for a, b in zip(preds, _LABELS))
    assert correct >= 5  # at least 5/6


def test_predict_proba_rows_sum_to_one():
    pipe = baseline.train(_TEXTS, _LABELS)
    proba = baseline.predict_proba(pipe, ["świetny produkt"])
    assert len(proba[0]) == 3
    assert abs(sum(proba[0]) - 1.0) < 1e-6


def test_save_and_load_roundtrip(tmp_path):
    pipe = baseline.train(_TEXTS, _LABELS)
    path = tmp_path / "model.joblib"
    baseline.save(pipe, path)
    assert path.exists()
    reloaded = baseline.load(path)
    assert baseline.predict(reloaded, _TEXTS) == baseline.predict(pipe, _TEXTS)
