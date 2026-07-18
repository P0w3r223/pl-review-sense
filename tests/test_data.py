from pl_review_sense.data import Split, label_name, map_target_name, to_split


def test_map_target_name_maps_the_three_kept_classes():
    assert map_target_name("minus") == 0  # negative
    assert map_target_name("zero") == 1  # neutral
    assert map_target_name("plus") == 2  # positive


def test_map_target_name_drops_ambiguous_and_unknown():
    assert map_target_name("amb") is None
    assert map_target_name("something-else") is None


def test_to_split_drops_ambiguous_and_pairs_labels():
    texts = ["bad", "meh", "great", "who knows"]
    names = ["minus", "zero", "plus", "amb"]
    split = to_split(texts, names)
    assert split.texts == ["bad", "meh", "great"]
    assert split.labels == [0, 1, 2]
    assert len(split) == 3


def test_split_len_matches_labels():
    assert len(Split(["a", "b"], [0, 2])) == 2


def test_label_name_is_ordinal():
    assert label_name(0) == "negative"
    assert label_name(1) == "neutral"
    assert label_name(2) == "positive"
