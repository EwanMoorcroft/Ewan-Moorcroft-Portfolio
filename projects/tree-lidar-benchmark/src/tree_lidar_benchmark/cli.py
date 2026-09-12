"""Command-line interface for offline result inspection and verification."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tree_lidar_benchmark.evaluator import evaluate_aligned_labels
from tree_lidar_benchmark.verification import load_overall_rows, verify_project

DISPLAY_NAMES = {
    "forestformer3d": "ForestFormer3D",
    "forainet": "ForAINet",
    "segmentanytree": "SegmentAnyTree",
    "tls2trees": "TLS2trees",
    "treelearn": "TreeLearn",
    "treex": "TreeX",
}


def _summary(root: Path | None) -> list[dict[str, object]]:
    rows = load_overall_rows(root)
    indexed = {(row["method"], row["variant"]): row for row in rows}
    summary: list[dict[str, object]] = []
    for method in sorted(DISPLAY_NAMES, key=lambda value: DISPLAY_NAMES[value].lower()):
        published = float(indexed[(method, "published_default")]["micro_f1"])
        selected = float(indexed[(method, "development_tuned")]["micro_f1"])
        summary.append(
            {
                "method": DISPLAY_NAMES[method],
                "published_default_micro_f1": published,
                "development_selected_micro_f1": selected,
                "difference": selected - published,
            }
        )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tree-lidar-benchmark",
        description="Inspect and verify retained tree LiDAR benchmark evidence.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="Run all integrity checks")
    verify.add_argument("--root", type=Path, help="Project root; inferred by default")
    verify.add_argument("--json", action="store_true", help="Emit JSON")
    summary = subparsers.add_parser("summary", help="Show paired micro F1 values")
    summary.add_argument("--root", type=Path, help="Project root; inferred by default")
    summary.add_argument("--json", action="store_true", help="Emit JSON")
    evaluate = subparsers.add_parser("evaluate", help="Score small source-row-aligned label arrays")
    evaluate.add_argument(
        "--labels", required=True, type=Path, help="JSON with four aligned arrays"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected command and return its process status."""

    args = _build_parser().parse_args(argv)
    if args.command == "evaluate":
        try:
            payload = json.loads(
                args.labels.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
            )
            fields = {
                "predicted_tree_id",
                "reference_tree_id",
                "classification",
                "source_row_index",
            }
            if not isinstance(payload, dict) or set(payload) != fields:
                raise ValueError("labels must contain exactly the four aligned arrays")
            if any(not isinstance(value, list) for value in payload.values()):
                raise ValueError("each label field must be an array")
            result = evaluate_aligned_labels(**payload)
        except (OSError, ValueError, TypeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "verify":
        payload = verify_project(args.root).to_dict()
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                "verified "
                f"{payload['protocol']}: "
                f"{payload['per_plot_rows']} plot rows, "
                f"{payload['aggregate_values_checked']} aggregate values, "
                f"{payload['retained_hashes_checked']} retained hashes"
            )
        return 0

    rows = _summary(args.root)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(f"{'Method':<18} {'Published':>10} {'Selected':>10} {'Difference':>11}")
        for row in rows:
            print(
                f"{row['method']:<18} "
                f"{row['published_default_micro_f1']:>10.3f} "
                f"{row['development_selected_micro_f1']:>10.3f} "
                f"{row['difference']:>+11.3f}"
            )
    return 0


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate label field: {key}")
        payload[key] = value
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
