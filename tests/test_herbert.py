import pl_review_sense.herbert as herbert
from pl_review_sense import config


def test_module_imports_without_torch():
    # The heavy deps must be lazy: importing the module must not require torch.
    assert hasattr(herbert, "fine_tune")


def test_smoke_args_are_small_and_single_epoch():
    args = herbert.smoke_args()
    assert args.subset is not None and args.subset <= 500
    assert args.epochs == 1


def test_full_args_follow_config():
    args = herbert.full_args()
    assert args.subset is None
    assert args.epochs == config.HERBERT_EPOCHS
    assert args.max_len == config.HERBERT_MAX_LEN


def test_parser_smoke_flag():
    assert herbert.build_parser().parse_args(["--smoke"]).smoke is True
    assert herbert.build_parser().parse_args([]).smoke is False


def test_write_metrics_representative_becomes_headline(tmp_path, monkeypatch):
    import json

    from pl_review_sense import evaluate

    monkeypatch.setattr(config, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(config, "HERBERT_METRICS_PATH", tmp_path / "herbert.json")
    result = evaluate.evaluate([0, 1, 2], [0, 1, 2])

    herbert.write_metrics(result, "full", representative=True)
    assert (tmp_path / "herbert.json").exists()
    assert not (tmp_path / "herbert_smoke.json").exists()
    assert json.loads((tmp_path / "herbert.json").read_text(encoding="utf-8"))["representative"] is True


def test_write_metrics_non_representative_goes_to_sidecar(tmp_path, monkeypatch):
    from pl_review_sense import evaluate

    monkeypatch.setattr(config, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(config, "HERBERT_METRICS_PATH", tmp_path / "herbert.json")
    result = evaluate.evaluate([0, 1, 2], [0, 1, 2])

    herbert.write_metrics(result, "smoke", representative=False)
    assert (tmp_path / "herbert_smoke.json").exists()
    assert not (tmp_path / "herbert.json").exists()
