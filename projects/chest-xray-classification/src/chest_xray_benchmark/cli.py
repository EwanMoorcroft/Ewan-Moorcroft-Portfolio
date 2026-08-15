"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .manifest import (
    group_duplicates,
    read_manifest,
    scan_dataset,
    verification_report,
    write_manifest,
)
from .spec import DatasetSpec
from .splitting import assign_groups, audit_partition_map, write_json, write_splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cxr-benchmark",
        description="Leakage-aware chest X-ray classification benchmark.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify", help="Inventory images and form duplicate groups.")
    verify.add_argument("--data-root", type=Path, required=True)
    verify.add_argument("--spec", type=Path, required=True)
    verify.add_argument("--manifest-out", type=Path, required=True)
    verify.add_argument("--report-out", type=Path, required=True)
    verify.add_argument("--near-hamming", type=int, default=2)
    verify.add_argument("--strict", action="store_true")

    split = commands.add_parser("split", help="Create deterministic group-level partitions.")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--summary-out", type=Path, required=True)
    split.add_argument("--seed", type=int, default=534)
    split.add_argument("--train-fraction", type=float, default=0.70)
    split.add_argument("--validation-fraction", type=float, default=0.15)
    split.add_argument("--test-fraction", type=float, default=0.15)

    train = commands.add_parser(
        "train", help="Fit with train data and select with validation data."
    )
    train.add_argument("--data-root", type=Path, required=True)
    train.add_argument("--splits", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)

    evaluate = commands.add_parser("evaluate", help="Evaluate a selected checkpoint on test data.")
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--splits", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--run-metadata", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def _verify(args: argparse.Namespace) -> int:
    spec = DatasetSpec.load(args.spec)
    records, errors, unexpected = scan_dataset(args.data_root, spec)
    grouped = group_duplicates(records, args.near_hamming)
    report = verification_report(grouped, spec, errors, unexpected, args.near_hamming)
    write_manifest(grouped, args.manifest_out)
    write_json(report, args.report_out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if args.strict and not report["identity_matches_spec"] else 0


def _split(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    fractions = {
        "train": args.train_fraction,
        "validation": args.validation_fraction,
        "test": args.test_fraction,
    }
    partition_map = assign_groups(records, args.seed, fractions)
    summary = audit_partition_map(records, partition_map, args.seed, fractions)
    write_splits(records, partition_map, args.output)
    write_json(summary, args.summary_out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["split_ready"] else 2


def _train(args: argparse.Namespace) -> int:
    from .training import train_model

    metadata = train_model(args.data_root, args.splits, args.config, args.output_dir)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_checkpoint

    payload = evaluate_checkpoint(
        args.data_root,
        args.splits,
        args.checkpoint,
        args.run_metadata,
        args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "verify": _verify,
        "split": _split,
        "train": _train,
        "evaluate": _evaluate,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
