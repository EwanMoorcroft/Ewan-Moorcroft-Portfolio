"""Command-line entry point for validation, evaluation, and safe artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .batch_store import store_generated_batch
from .config import ProtocolConfig
from .contracts import canonical_json_bytes, sha256_bytes, validate_season_id
from .data import load_gameweeks, validate_sequence
from .errors import DataContractError, ForecastingError
from .evaluation import evaluate_models
from .features import build_forecast_frame, build_training_frame
from .historical import import_historical_gameweeks
from .models import ARTIFACT_FORMAT, FittedRidge, fit_ridge
from .operational import (
    COMPLETION_STATUS,
    operational_manifest,
    ranked_prediction_payload,
    snapshot_input_audit,
    validate_deployment_model,
    validate_expected_as_of,
)
from .report import write_evaluation_html
from .synthetic import write_synthetic_gameweeks


def _protocol(path: str | None) -> ProtocolConfig:
    return ProtocolConfig.from_json(path) if path else ProtocolConfig()


def _snapshots(
    directory: str,
    protocol: ProtocolConfig,
    *,
    expected_season_id: str | None = None,
):
    return load_gameweeks(
        directory,
        minimum_adjacent_coverage=protocol.minimum_adjacent_player_coverage,
        expected_season_id=expected_season_id,
    )


def _write_frame(frame, path: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fpl-forecast",
        description="Chronological next-gameweek forecasting from completed FPL live data.",
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
    train.add_argument("--season", required=True)
    train.add_argument("--config")

    predict = commands.add_parser(
        "predict", help="rank next-gameweek predictions from completed snapshots"
    )
    predict.add_argument("--gameweek-dir", required=True)
    predict.add_argument("--artifact", required=True)
    predict.add_argument("--season", required=True)
    predict.add_argument("--expected-as-of-gw", required=True, type=int)
    predict.add_argument(
        "--completion-status",
        required=True,
        choices=[COMPLETION_STATUS],
        help="caller declaration that every supplied snapshot is completed",
    )
    predict.add_argument("--predictions", required=True)
    predict.add_argument("--manifest", required=True)
    predict.add_argument("--config")

    store_batch = commands.add_parser(
        "store-batch",
        help="verify generated forecast evidence and persist one immutable batch",
    )
    store_batch.add_argument("--database", required=True)
    store_batch.add_argument("--gameweek-dir", required=True)
    store_batch.add_argument("--artifact", required=True)
    store_batch.add_argument("--predictions", required=True)
    store_batch.add_argument("--manifest", required=True)

    synthetic = commands.add_parser("synthetic", help="write deterministic local gameweek fixtures")
    synthetic.add_argument("--output-dir", required=True)
    synthetic.add_argument("--gameweeks", type=int, default=10)
    synthetic.add_argument("--players", type=int, default=24)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--season", required=True)

    historical = commands.add_parser(
        "import-vaastav",
        help="convert a pinned historical CSV interval into completed gameweek JSON",
    )
    historical.add_argument("--source-dir", required=True)
    historical.add_argument("--output-dir", required=True)
    historical.add_argument("--manifest", required=True)
    historical.add_argument("--season", required=True)
    historical.add_argument("--source-revision", required=True)
    historical.add_argument("--gameweek-start", required=True, type=int)
    historical.add_argument("--gameweek-end", required=True, type=int)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run a parsed command and return its concise summary."""

    if args.command == "synthetic":
        paths = write_synthetic_gameweeks(
            args.output_dir,
            gameweeks=args.gameweeks,
            players=args.players,
            seed=args.seed,
            season_id=args.season,
        )
        return {
            "command": "synthetic",
            "gameweeks": len(paths),
            "players": args.players,
            "seed": args.seed,
            "season_id": validate_season_id(args.season),
        }

    if args.command == "import-vaastav":
        result = import_historical_gameweeks(
            args.source_dir,
            args.output_dir,
            season=args.season,
            source_revision=args.source_revision,
            gameweek_start=args.gameweek_start,
            gameweek_end=args.gameweek_end,
        )
        manifest = Path(args.manifest)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(result.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "command": "import-vaastav",
            "gameweeks": len(result.paths),
            "gameweek_start": args.gameweek_start,
            "gameweek_end": args.gameweek_end,
            "source_files_sha256": result.manifest["source_files_sha256"],
        }

    if args.command == "store-batch":
        result = store_generated_batch(
            database_path=args.database,
            gameweek_dir=args.gameweek_dir,
            artifact_path=args.artifact,
            predictions_path=args.predictions,
            manifest_path=args.manifest,
        )
        return {
            "command": "store-batch",
            "run_id": result.run_id,
            "idempotency_key": result.idempotency_key,
            "predictions": result.prediction_count,
            "replayed": result.replayed,
        }

    protocol = _protocol(args.config)
    expected_season_id = args.season if args.command in {"train", "predict"} else None
    snapshots = _snapshots(
        args.gameweek_dir,
        protocol,
        expected_season_id=expected_season_id,
    )
    if args.command == "validate":
        summary = validate_sequence(
            snapshots,
            minimum_adjacent_coverage=protocol.minimum_adjacent_player_coverage,
        )
        return {"command": "validate", **summary.to_dict()}

    if args.command == "predict":
        season_id = validate_season_id(args.season)
        as_of_gw = validate_expected_as_of(snapshots, args.expected_as_of_gw)
        frame = build_forecast_frame(
            snapshots,
            minimum_adjacent_coverage=protocol.minimum_adjacent_player_coverage,
        )

        artifact_path = Path(args.artifact)
        artifact_bytes = artifact_path.read_bytes()
        model = FittedRidge.from_json_bytes(artifact_bytes)
        validate_deployment_model(
            model,
            season_id=season_id,
            config_sha256=protocol.sha256(),
            as_of_gw=as_of_gw,
        )
        prediction_payload = ranked_prediction_payload(
            frame,
            model,
            season_id=season_id,
            completion_status=args.completion_status,
        )
        prediction_bytes = canonical_json_bytes(prediction_payload)
        input_files, input_sha256 = snapshot_input_audit(snapshots)
        prediction_path = Path(args.predictions)
        manifest_path = Path(args.manifest)

        protected = {
            artifact_path.resolve(),
            *(snapshot.source.resolve() for snapshot in snapshots),
        }
        if args.config:
            protected.add(Path(args.config).resolve())
        if prediction_path.resolve() == manifest_path.resolve():
            raise DataContractError("Predictions and manifest paths must be different")
        if prediction_path.resolve() in protected or manifest_path.resolve() in protected:
            raise DataContractError("Output paths cannot overwrite an artifact, config, or input")

        manifest_payload = operational_manifest(
            season_id=season_id,
            as_of_gw=as_of_gw,
            completion_status=args.completion_status,
            protocol=protocol,
            model=model,
            artifact_path=artifact_path,
            artifact_sha256=sha256_bytes(artifact_bytes),
            input_files=input_files,
            input_sha256=input_sha256,
            predictions_path=prediction_path,
            predictions_sha256=sha256_bytes(prediction_bytes),
            prediction_rows=len(frame),
        )
        manifest_bytes = canonical_json_bytes(manifest_payload)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_bytes(prediction_bytes)
        manifest_path.write_bytes(manifest_bytes)
        return {
            "command": "predict",
            "season_id": season_id,
            "as_of_gw": as_of_gw,
            "target_gw": as_of_gw + 1,
            "predictions": len(frame),
            "caller_declared_completion_status": args.completion_status,
            "manifest_format": manifest_payload["manifest_format"],
        }

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
        artifact_path = Path(args.artifact)
        protected = {snapshot.source.resolve() for snapshot in snapshots}
        if args.config:
            protected.add(Path(args.config).resolve())
        if artifact_path.resolve() in protected:
            raise DataContractError("Artifact path cannot overwrite a config or input snapshot")
        model = fit_ridge(
            frame,
            alpha=protocol.ridge_alpha,
            season_id=args.season,
            protocol_config_sha256=protocol.sha256(),
        )
        model.save(artifact_path)
        return {
            "command": "train",
            "artifact_format": ARTIFACT_FORMAT,
            "season_id": model.season_id,
            "feature_schema_sha256": model.feature_schema_sha256,
            "protocol_config_sha256": model.protocol_config_sha256,
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
