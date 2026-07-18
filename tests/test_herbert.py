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
