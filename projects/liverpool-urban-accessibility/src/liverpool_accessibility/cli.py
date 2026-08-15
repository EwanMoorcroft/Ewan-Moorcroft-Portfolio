"""Command-line entry point for preparation, analysis, reporting, and verification."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from .analysis import prepare_evidence, write_prepared
from .contracts import DataContractError
from .evidence import verify_evidence, verify_source_manifest, write_results
from .fixtures import fixture_source_manifest, write_fixture


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="liverpool-access",
        description="Reproducible spatial analysis of Liverpool Census workplace flows.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="validate and derive compact analysis inputs")
    prepare.add_argument("--flow-csv", required=True)
    prepare.add_argument("--boundaries", required=True)
    prepare.add_argument("--centroids", required=True)
    prepare.add_argument("--source-manifest", required=True)
    prepare.add_argument("--output-dir", required=True)

    analyse = commands.add_parser("analyse", help="fit the fixed model and spatial statistic")
    analyse.add_argument("--evidence-dir", required=True)

    report = commands.add_parser("report", help="render figures from retained evidence")
    report.add_argument("--evidence-dir", required=True)
    report.add_argument("--output-dir", required=True)

    verify = commands.add_parser("verify", help="verify retained hashes and numerical claims")
    verify.add_argument("--evidence-dir", required=True)

    fixture = commands.add_parser("fixture", help="run the full pipeline on fictional grid data")
    fixture.add_argument("--output-dir", required=True)
    fixture.add_argument("--size", type=int, default=5)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run one parsed command and return a concise JSON-ready summary."""
    if args.command == "prepare":
        destination = Path(args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        source_manifest = Path(args.source_manifest)
        payload = json.loads(source_manifest.read_text(encoding="utf-8"))
        verify_source_manifest(
            payload,
            {
                "flow_csv": args.flow_csv,
                "boundaries": args.boundaries,
                "centroids": args.centroids,
            },
        )
        prepared = prepare_evidence(args.flow_csv, args.boundaries, args.centroids)
        paths = write_prepared(prepared, destination)
        retained_manifest = destination / "source-manifest.json"
        if source_manifest.resolve() != retained_manifest.resolve():
            shutil.copyfile(source_manifest, retained_manifest)
        return {
            "command": "prepare",
            "areas": len(prepared.metrics),
            "spatial_edges": len(prepared.edges),
            "files": [path.name for path in paths],
        }
    if args.command == "analyse":
        paths = write_results(args.evidence_dir)
        return {"command": "analyse", "files": [path.name for path in paths]}
    if args.command == "report":
        from .reporting import write_figures

        paths = write_figures(args.evidence_dir, args.output_dir)
        return {"command": "report", "figures": [path.name for path in paths]}
    if args.command == "verify":
        return {"command": "verify", **verify_evidence(args.evidence_dir)}
    if args.command == "fixture":
        from .reporting import write_figures

        output = Path(args.output_dir)
        source = output / "source"
        evidence = output / "evidence"
        figures = output / "figures"
        flow_path, boundary_path, centroid_path = write_fixture(source, size=args.size)
        source_manifest = source / "source-manifest.json"
        _write_json(
            source_manifest,
            fixture_source_manifest(flow_path, boundary_path, centroid_path),
        )
        verify_source_manifest(
            json.loads(source_manifest.read_text(encoding="utf-8")),
            {
                "flow_csv": flow_path,
                "boundaries": boundary_path,
                "centroids": centroid_path,
            },
        )
        prepared = prepare_evidence(flow_path, boundary_path, centroid_path)
        write_prepared(prepared, evidence)
        shutil.copyfile(source_manifest, evidence / "source-manifest.json")
        write_results(evidence)
        write_figures(evidence, figures)
        return {
            "command": "fixture",
            "areas": len(prepared.metrics),
            "spatial_edges": len(prepared.edges),
            **verify_evidence(evidence),
        }
    raise DataContractError(f"unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except (DataContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
