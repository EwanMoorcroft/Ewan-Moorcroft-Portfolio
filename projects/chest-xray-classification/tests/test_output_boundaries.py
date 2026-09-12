"""Evaluation must preserve source files and publish a coherent artifact pair."""

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from chest_xray_benchmark import evaluation
from chest_xray_benchmark.image_data import LABEL_ORDER
from chest_xray_benchmark.splitting import SPLIT_FIELDS


@pytest.fixture
def evaluation_files(tmp_path, monkeypatch):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([0.4, 0.3, 0.3]).log())

        def forward(self, images):
            return self.logits.expand(len(images), -1)

    monkeypatch.setattr(evaluation, "build_resnet18", lambda *args, **kwargs: TinyModel())
    rows = []
    for index, (split, label) in enumerate(
        (s, label) for s in ("train", "validation", "test") for label in LABEL_ORDER
    ):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (8, 8), color=(index, 0, 0)).save(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        rows.append(
            {
                "relative_path": path.name,
                "label": label,
                "byte_size": len(raw),
                "sha256": digest,
                "difference_hash": "0",
                "width": 8,
                "height": 8,
                "color_mode": "RGB",
                "has_exif": 0,
                "exact_group_id": "sha256-" + digest,
                "split": split,
            }
        )
    splits = tmp_path / "splits.csv"
    with splits.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SPLIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(TinyModel().state_dict(), checkpoint)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "label_order": list(LABEL_ORDER),
                "settings": {"image_size": 64, "device": "cpu"},
                "split_file_sha256": hashlib.sha256(splits.read_bytes()).hexdigest(),
                "checkpoint_file_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        )
    )
    return {
        "root": tmp_path,
        "splits": splits,
        "checkpoint": checkpoint,
        "metadata": metadata,
        "image": tmp_path / "image-0.png",
        "report": tmp_path / "evaluation.json",
    }


def run_evaluation(files):
    return evaluation.evaluate_checkpoint(
        files["root"], files["splits"], files["checkpoint"], files["metadata"], files["report"]
    )


@pytest.mark.parametrize("source_key", ["splits", "checkpoint", "metadata", "image"])
@pytest.mark.parametrize("output_kind", ["report", "predictions"])
@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_report_and_sidecar_cannot_alias_any_input(
    evaluation_files, source_key, output_kind, alias_kind
):
    files = evaluation_files
    source = files[source_key]
    original = source.read_bytes()
    target = (
        files["report"] if output_kind == "report" else files["root"] / "evaluation-predictions.csv"
    )
    if alias_kind == "symlink":
        target.symlink_to(source)
    else:
        os.link(source, target)
    with pytest.raises(ValueError, match="overwrite"):
        run_evaluation(files)
    assert source.read_bytes() == original


def test_generated_prediction_name_cannot_overwrite_split_csv(evaluation_files):
    files = evaluation_files
    raw = files["splits"].read_bytes()
    files["splits"] = files["root"] / "evaluation-predictions.csv"
    files["splits"].write_bytes(raw)
    with pytest.raises(ValueError, match="overwrite"):
        run_evaluation(files)
    assert files["splits"].read_bytes() == raw


@pytest.mark.parametrize("existing", [True, False])
def test_second_replace_failure_restores_or_removes_prediction_pair(
    evaluation_files, monkeypatch, existing
):
    files = evaluation_files
    report = files["report"]
    sidecar = files["root"] / "evaluation-predictions.csv"
    if existing:
        report.write_bytes(b"old report")
        sidecar.write_bytes(b"old predictions")
    real_replace = os.replace
    failed = False

    def fail_report_once(source, destination):
        nonlocal failed
        if Path(destination) == report and not failed:
            failed = True
            raise OSError("simulated publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_report_once)
    with pytest.raises(OSError, match="simulated publication failure"):
        run_evaluation(files)
    if existing:
        assert report.read_bytes() == b"old report"
        assert sidecar.read_bytes() == b"old predictions"
    else:
        assert not report.exists()
        assert not sidecar.exists()
