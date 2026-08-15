"""Command-line entry point for validation, evaluation, and safe artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import ProtocolConfig
from .data import load_gameweeks, validate_sequence
from .errors import ForecastingError
from .evaluation import evaluate_models
from .features import build_training_frame
from .models import fit_ridge
from .report import write_evaluation_html
from .synthetic import write_synthetic_gameweeks


def _protocol(path: str | None) -> ProtocolConfig:
    return ProtocolConfig.from_json(path) if path else ProtocolConfig()


def _snapshots(directory: str, protocol: ProtocolConfig):
    return load_gameweeks(
        directory,
        minimum_adjacent_coverage=protocol.minimum_adjacent_player_coverage,
    )


def _write_frame(frame, path: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fpl-forecast",
        description="Leak-safe next-gameweek forecasting from completed FPL live data.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate completed gameweeks")
    validate.add_argument("--gameweek-dir", required=True)
    validate.add_argument("--config")

    build = commands.add_parser("build", help="build a leak-checked training table")
    build.add_argument("--gameweek-dir", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--config")

    evaluate = commands.add_parser(
        "evaluate", help="run rolling-origin baseline and ridge comparisons"
    )
    evaluate.add_argument("--gameweek-dir", required=True)
    evaluate.add_argument("--config")
    evaluate.add_argument("--report-json")
    evaluate.add_argument("--predictions-csv")
    evaluate.add_argument("--report-html")

    train = commands.add_parser(
        "train", help="fit ridge on all completed targets and save safe JSON"
    )
    train.add_argument("--gameweek-dir", required=True)
    train.add_argument("--artifact", required=True)
    train.add_argument("--config")

    synthetic = commands.add_parser("synthetic", help="write deterministic local gameweek fixtures")
    synthetic.add_argument("--output-dir", required=True)
    synthetic.add_argument("--gameweeks", type=int, default=10)
    synthetic.add_argument("--players", type=int, default=24)
    synthetic.add_argument("--seed", type=int, default=42)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run a parsed command and return its concise summary."""

    if args.command == "synthetic":
        paths = write_synthetic_gameweeks(
            args.output_dir,
            gameweeks=args.gameweeks,
            players=args.players,
            seed=args.seed,
        )
        return {
            "command": "synthetic",
            "gameweeks": len(paths),
            "players": args.players,
            "seed": args.seed,
        }

    protocol = _protocol(args.config)
    snapshots = _snapshots(args.gameweek_dir, protocol)
    if args.command == "validate":
        summary = validate_sequence(
            snapshots,
            minimum_adjacent_coverage=protocol.minimum_adjacent_player_coverage,
        )
        return {"command": "validate", **summary.to_dict()}

    frame = build_training_frame(
        snapshots,
        minimum_adjacent_coverage=protocol.minimum_adjacent_player_coverage,
    )
    if args.command == "build":
        _write_frame(frame, args.output)
        return {
            "command": "build",
            "rows": len(frame),
            "players": int(frame["player_id"].nunique()),
            "target_gameweek_start": int(frame["target_gw"].min()),
            "target_gameweek_end": int(frame["target_gw"].max()),
        }

    if args.command == "evaluate":
        result = evaluate_models(frame, protocol)
        if args.report_json:
            result.write_json(args.report_json)
        if args.predictions_csv:
            result.write_predictions(args.predictions_csv)
        if args.report_html:
            write_evaluation_html(result.report, args.report_html)
        return {"command": "evaluate", **result.report["data"]}

    if args.command == "train":
        model = fit_ridge(frame, alpha=protocol.ridge_alpha)
        model.save(args.artifact)
        return {
            "command": "train",
            "artifact_format": "fpl-ridge-json-v1",
            "training_rows": model.training_rows,
            "trained_through_target_gw": model.trained_through_target_gw,
        }
    raise ForecastingError(f"Unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except (ForecastingError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
