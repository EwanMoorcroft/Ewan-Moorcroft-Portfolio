"""Command-line interface for neural chunking experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_chunking.training import TrainingConfig, evaluate_pipeline, train_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="fit and select with validation data")
    train_parser.add_argument("data", type=Path, help="two-column token and BIO label file")
    train_parser.add_argument("--output", type=Path, default=Path("runs/default"))
    train_parser.add_argument(
        "--architecture",
        choices=("bilstm", "transformer"),
        default="bilstm",
    )
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--seed", type=int, default=534)

    evaluate_parser = commands.add_parser("evaluate", help="evaluate one selected checkpoint")
    evaluate_parser.add_argument("data", type=Path, help="same data file used during training")
    evaluate_parser.add_argument("checkpoint", type=Path)
    evaluate_parser.add_argument("--output", type=Path, default=Path("runs/test-results.json"))
    return parser


def main() -> int:
    """Run the selected training or evaluation workflow."""
    args = build_parser().parse_args()
    if args.command == "train":
        result = train_pipeline(
            args.data,
            args.output,
            TrainingConfig(architecture=args.architecture, epochs=args.epochs, seed=args.seed),
        )
        print(f"Best validation span F1: {result['best_validation_span_f1']:.4f}")
        return 0
    result = evaluate_pipeline(args.data, args.checkpoint, args.output)
    print(f"Test span F1: {result['test']['span_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
