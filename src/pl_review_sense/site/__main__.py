"""``python -m pl_review_sense.site`` — rebuild the page from the committed metrics.

``--out`` exists for the CI drift job, which builds into a scratch directory and diffs the
result against the committed ``docs/index.html``. Without it the page would have to be
overwritten in place to be checked, and a failing check would leave the working tree dirty.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .build import build


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="pl_review_sense.site", description="Build the published page."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory to write index.html into (default: docs/)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="directory to read the committed metrics from (default: reports/metrics/)",
    )
    args = parser.parse_args(argv)
    print("wrote", build(out_dir=args.out, metrics_dir=args.metrics))


if __name__ == "__main__":
    main()
