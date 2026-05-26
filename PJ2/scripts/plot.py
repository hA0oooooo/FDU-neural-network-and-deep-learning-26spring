from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.plot import (
    plot_confusion_from_test_json,
    plot_train_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    curves = subparsers.add_parser("curves")
    curves.add_argument("experiments", nargs="+")
    curves.add_argument("--output", required=True)

    confusion = subparsers.add_parser("confusion")
    confusion.add_argument("experiment")

    return parser.parse_args()


def experiment_dir(experiment_id: str) -> Path:
    task = "bn" if experiment_id.startswith(("bn_", "vgga_")) else "classification"
    return ROOT / "outputs" / task / experiment_id


def main() -> None:
    args = parse_args()

    if args.command == "curves":
        train_csvs = {
            experiment_id: experiment_dir(experiment_id) / "train.csv"
            for experiment_id in args.experiments
        }
        plot_train_comparison(train_csvs, ROOT / "outputs" / args.output)
        return

    exp_dir = experiment_dir(args.experiment)
    plot_confusion_from_test_json(exp_dir / "test.json", exp_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
