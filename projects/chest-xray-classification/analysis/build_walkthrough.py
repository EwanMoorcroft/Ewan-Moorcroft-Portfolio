"""Build public X-ray result artifacts from compact committed evidence.

This script does not access images, ignored run outputs, checkpoints or the full
perceptual-review report. Every generated table, figure and notebook reads the
same compact grouped-split evidence file.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"
NOTEBOOKS = ROOT / "notebooks"


def load_json(name: str) -> dict:
    with (EVIDENCE / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_csv(name: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def svg_text(x: float, y: float, value: str, *, size: int = 14, anchor: str = "start") -> str:
    return (
        f'<text x="{x:g}" y="{y:g}" font-family="Arial, sans-serif" '
        f'font-size="{size}px" text-anchor="{anchor}" fill="#17202a">'
        f"{html.escape(value)}</text>"
    )


def write_svg(name: str, content: str, width: int, height: int) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    document = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        f"{content}</svg>\n"
    )
    (ASSETS / name).write_text(document, encoding="utf-8")


def composition_svg(dataset: dict) -> None:
    classes = dataset["classes"]
    colours = ["#3264a8", "#d97732", "#4f8a5b"]
    width, height = 900, 520
    left, chart_top, chart_width, chart_height = 100, 145, 690, 270
    maximum = max(item["expected_count"] for item in classes.values())
    parts = [
        svg_text(45, 42, "Chest X-ray dataset composition", size=26),
        svg_text(45, 72, "Expected counts from the repository data contract", size=16),
        svg_text(45, 98, "Not a fresh scan of the excluded image files", size=14),
        f'<line x1="{left}" y1="{chart_top + chart_height}" x2="{left + chart_width}" '
        f'y2="{chart_top + chart_height}" stroke="#34495e" stroke-width="2"/>',
    ]
    for tick in range(0, maximum + 1, 250):
        y = chart_top + chart_height - (tick / maximum) * chart_height
        parts.append(
            f'<line x1="{left}" y1="{y:g}" x2="{left + chart_width}" y2="{y:g}" '
            'stroke="#d9e1e8" stroke-width="1"/>'
        )
        parts.append(svg_text(left - 12, y + 5, f"{tick:,}", size=12, anchor="end"))
    bar_width = 150
    gap = 75
    for index, (label, item) in enumerate(classes.items()):
        x = left + gap + index * (bar_width + gap)
        bar_height = item["expected_count"] / maximum * chart_height
        y = chart_top + chart_height - bar_height
        parts.append(
            f'<rect x="{x}" y="{y:g}" width="{bar_width}" height="{bar_height:g}" '
            f'fill="{colours[index]}" rx="4"/>'
        )
        parts.append(
            svg_text(x + bar_width / 2, y - 10, f"{item['expected_count']:,}", anchor="middle")
        )
        parts.append(
            svg_text(x + bar_width / 2, chart_top + chart_height + 30, label, anchor="middle")
        )
    parts.append(svg_text(45, height - 22, "Source: data/dataset-spec.json", size=12))
    write_svg("dataset-composition.svg", "".join(parts), width, height)


def model_results_svg(results: dict) -> None:
    width, height = 1100, 650
    parts = [
        svg_text(42, 42, "Chest X-ray model results", size=26),
        svg_text(42, 72, "Exact-copy-grouped split | 522 held-out test images", size=16),
        svg_text(
            42,
            98,
            "Selected epoch 7 by validation macro F1 (0.8244)",
            size=14,
        ),
    ]
    overall = [
        ("Accuracy", results["test"]["accuracy"]),
        ("Balanced accuracy", results["test"]["balanced_accuracy"]),
        ("Macro F1", results["test"]["macro_f1"]),
        ("MCC", results["test"]["matthews_correlation_coefficient"]),
    ]
    classes = results["test"]["per_class"]
    left_x, top, panel_w, panel_h = 75, 155, 410, 330
    right_x = 610
    for panel_x, title in ((left_x, "Aggregate test metrics"), (right_x, "Per-class F1")):
        parts.append(svg_text(panel_x, top - 28, title, size=18))
        parts.append(
            f'<line x1="{panel_x}" y1="{top + panel_h}" x2="{panel_x + panel_w}" '
            f'y2="{top + panel_h}" stroke="#34495e" stroke-width="2"/>'
        )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + panel_h - tick * panel_h
            parts.append(
                f'<line x1="{panel_x}" y1="{y:g}" x2="{panel_x + panel_w}" y2="{y:g}" '
                'stroke="#d9e1e8" stroke-width="1"/>'
            )
            parts.append(svg_text(panel_x - 10, y + 5, f"{tick:.2f}", size=11, anchor="end"))
    colours = ["#3264a8", "#d97732", "#4f8a5b", "#8556a8"]
    bar_width = 62
    gap = 34
    for index, (label, value) in enumerate(overall):
        x = left_x + 28 + index * (bar_width + gap)
        bar_height = value * panel_h
        y = top + panel_h - bar_height
        parts.append(
            f'<rect x="{x}" y="{y:g}" width="{bar_width}" height="{bar_height:g}" '
            f'fill="{colours[index]}" rx="3"/>'
        )
        parts.append(svg_text(x + bar_width / 2, y - 9, f"{value:.4f}", size=11, anchor="middle"))
        parts.append(
            svg_text(x + bar_width / 2, top + panel_h + 28, label, size=11, anchor="middle")
        )
    class_colours = ["#3264a8", "#d97732", "#4f8a5b"]
    for index, (label, values) in enumerate(classes.items()):
        value = values["f1"]
        x = right_x + 48 + index * 116
        bar_height = value * panel_h
        y = top + panel_h - bar_height
        parts.append(
            f'<rect x="{x}" y="{y:g}" width="{bar_width}" height="{bar_height:g}" '
            f'fill="{class_colours[index]}" rx="3"/>'
        )
        parts.append(svg_text(x + bar_width / 2, y - 9, f"{value:.4f}", size=11, anchor="middle"))
        parts.append(
            svg_text(x + bar_width / 2, top + panel_h + 28, label, size=11, anchor="middle")
        )
    parts.append(svg_text(42, height - 35, "Source: evidence/retained-results.json", size=12))
    write_svg("model-results.svg", "".join(parts), width, height)


def notebook_lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


def markdown_cell(cell_id: str, source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": notebook_lines(source),
    }


def code_cell(cell_id: str, execution_count: int, source: str, output: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "id": cell_id,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": notebook_lines(output),
            }
        ],
        "source": notebook_lines(source),
    }


def write_notebook(results: dict, spec: dict) -> None:
    dataset = results["dataset"]
    split = results["split"]
    model = results["model"]
    test = results["test"]
    labels = results["provenance"]["label_order"]

    setup_source = """import json
from pathlib import Path

for candidate in (Path.cwd(), *Path.cwd().parents):
    if (candidate / "evidence" / "retained-results.json").is_file():
        ROOT = candidate
        break
else:
    raise FileNotFoundError("Run this notebook from the project or one of its subdirectories")

spec = json.loads((ROOT / "data" / "dataset-spec.json").read_text())
results = json.loads((ROOT / "evidence" / "retained-results.json").read_text())

assert results["evidence_status"] == "retained grouped-split evaluation"
assert results["split"]["split_ready"] is True
print(f"Dataset: {spec['name']} v{spec['version']}")
print(f"DOI: {results['dataset']['doi']}")
print(f"Verified images: {results['dataset']['image_count']}")
print(f"Exact-identity groups: {results['dataset']['exact_identity_group_count']}")
print(f"Grouped split ready: {results['split']['split_ready']}")"""
    setup_output = (
        f"Dataset: {spec['name']} v{spec['version']}\n"
        f"DOI: {dataset['doi']}\n"
        f"Verified images: {dataset['image_count']}\n"
        f"Exact-identity groups: {dataset['exact_identity_group_count']}\n"
        f"Grouped split ready: {split['split_ready']}\n"
    )

    composition_source = """total = spec["expected_total"]
print("class\texpected_count\tshare")
for label, item in spec["classes"].items():
    print(f"{label}\t{item['expected_count']}\t{item['expected_count'] / total:.2%}")"""
    composition_output = "class\texpected_count\tshare\n" + "".join(
        f"{label}\t{item['expected_count']}\t{item['expected_count'] / spec['expected_total']:.2%}\n"
        for label, item in spec["classes"].items()
    )

    split_source = """split = results["split"]
labels = results["provenance"]["label_order"]
print("partition\timages\texact_groups\t" + "\t".join(labels))
for name, values in split["partitions"].items():
    counts = "\t".join(str(values["class_counts"][label]) for label in labels)
    print(f"{name}\t{values['images']}\t{values['exact_identity_groups']}\t{counts}")
print(f"cross-partition SHA-256 violations\t{split['sha256_cross_partition_violation_count']}")
print(
    "perceptual review pairs (not split groups)\t"
    f"{results['dataset']['perceptual_review_candidate_pair_count']}"
)"""
    split_output = "partition\timages\texact_groups\t" + "\t".join(labels) + "\n"
    for name, values in split["partitions"].items():
        counts = "\t".join(str(values["class_counts"][label]) for label in labels)
        split_output += f"{name}\t{values['images']}\t{values['exact_identity_groups']}\t{counts}\n"
    split_output += (
        f"cross-partition SHA-256 violations\t{split['sha256_cross_partition_violation_count']}\n"
        "perceptual review pairs (not split groups)\t"
        f"{dataset['perceptual_review_candidate_pair_count']}\n"
    )

    metrics_source = """test = results["test"]
print(f"Selected epoch\t{results['model']['selected_epoch']}")
print(f"Validation macro F1\t{results['model']['validation']['macro_f1']:.4f}")
print("Test metrics")
for name in ("accuracy", "balanced_accuracy", "macro_f1", "matthews_correlation_coefficient"):
    print(f"{name}\t{test[name]:.4f}")
print("Per-class F1")
for label, values in test["per_class"].items():
    print(f"{label}\t{values['f1']:.4f}")"""
    metrics_output = (
        f"Selected epoch\t{model['selected_epoch']}\n"
        f"Validation macro F1\t{model['validation']['macro_f1']:.4f}\n"
        "Test metrics\n"
    )
    for name in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "matthews_correlation_coefficient",
    ):
        metrics_output += f"{name}\t{test[name]:.4f}\n"
    metrics_output += "Per-class F1\n" + "".join(
        f"{label}\t{values['f1']:.4f}\n" for label, values in test["per_class"].items()
    )

    provenance_source = """labels = results["provenance"]["label_order"]
print(r"actual\\predicted" + "\t" + "\t".join(labels))
for label, row in zip(labels, results["test"]["confusion_matrix"], strict=True):
    print(label + "\t" + "\t".join(str(value) for value in row))
print("Provenance SHA-256")
for name in ("split_file_sha256", "config_file_sha256", "checkpoint_file_sha256"):
    print(f"{name}\t{results['provenance'][name]}")"""
    provenance_output = "actual\\predicted\t" + "\t".join(labels) + "\n"
    for label, row in zip(labels, test["confusion_matrix"], strict=True):
        provenance_output += label + "\t" + "\t".join(str(value) for value in row) + "\n"
    provenance_output += "Provenance SHA-256\n" + "".join(
        f"{name}\t{results['provenance'][name]}\n"
        for name in ("split_file_sha256", "config_file_sha256", "checkpoint_file_sha256")
    )

    cells = [
        markdown_cell(
            "overview",
            """# Chest X-ray classification walkthrough

This project fine-tunes ResNet18 for three-class chest X-ray classification. The current result uses a deterministic split that keeps SHA-256-identical files together, selects one checkpoint by validation macro F1 and evaluates it once on the held-out test partition.

> Exact-copy grouping prevents byte-identical leakage. It does not establish patient-level independence because patient identifiers are unavailable.""",
        ),
        code_cell("load-evidence", 1, setup_source, setup_output),
        markdown_cell(
            "dataset",
            """## Dataset

The verified Mendeley Data V1 collection contains 3,475 images across Normal, Lung Opacity and Viral Pneumonia. Images remain excluded from the repository.

![Verified dataset composition](../assets/dataset-composition.svg)""",
        ),
        code_cell("dataset-counts", 2, composition_source, composition_output),
        markdown_cell(
            "split-policy",
            """## Leakage-resistant split

Five exact duplicate groups were found. SHA-256 identity is the only automatic grouping rule; every exact group stays within one partition. Perceptual hashes produce direct review candidates only and never form transitive split groups.""",
        ),
        code_cell("split-summary", 3, split_source, split_output),
        markdown_cell(
            "implementation",
            """## Implementation

| Area | Repository implementation |
| --- | --- |
| Data checks | Dataset contract, class counts, readable-image checks and SHA-256 identity |
| Split | Deterministic 70/15/15 exact-group allocation with leakage and class-coverage audits |
| Training | ImageNet-initialised ResNet18, dropout, class-weighted loss and light rotations |
| Selection | Highest validation macro F1 with early stopping |
| Evaluation | Accuracy, balanced accuracy, macro F1, MCC, per-class metrics and calibration measures |
| Provenance | Split, config and selected-checkpoint SHA-256 digests |""",
        ),
        markdown_cell(
            "results",
            """## Grouped-split result

The selected checkpoint reached validation macro F1 0.8244 at epoch 7. On 522 held-out images, test macro F1 was 0.8097 and accuracy was 0.8065.

![Grouped-split model results](../assets/model-results.svg)""",
        ),
        code_cell("result-metrics", 4, metrics_source, metrics_output),
        markdown_cell(
            "limits",
            """## Confusion matrix, provenance and limits

The largest error count is 38 Normal images predicted as Lung Opacity. Perceptual-hash candidates remain unadjudicated review prompts, patient independence is not established, and no external population has been evaluated. This is a research benchmark, not a medical device.""",
        ),
        code_cell("provenance", 5, provenance_source, provenance_output),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    (NOTEBOOKS / "chest_xray_walkthrough.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    retained_results = load_json("retained-results.json")
    spec = json.loads((ROOT / "data" / "dataset-spec.json").read_text(encoding="utf-8"))
    duplicates = load_json("known-exact-duplicates.json")
    if retained_results.get("reproduced_in_this_repository") is not True:
        raise ValueError("Grouped evidence must remain explicitly reproduced")
    if retained_results.get("evidence_status") != "retained grouped-split evaluation":
        raise ValueError("Unexpected retained evidence status")
    if retained_results["dataset"]["doi"] != spec["doi"]:
        raise ValueError("Evidence DOI does not match the dataset contract")
    if not retained_results["split"]["split_ready"]:
        raise ValueError("Grouped split evidence is not ready")
    write_csv(
        "overall-metrics.csv",
        ["metric", "value", "image_count", "evidence_status"],
        [
            {
                "metric": label,
                "value": value,
                "image_count": retained_results["test"]["image_count"],
                "evidence_status": retained_results["evidence_status"],
            }
            for label, value in (
                ("accuracy", retained_results["test"]["accuracy"]),
                ("balanced_accuracy", retained_results["test"]["balanced_accuracy"]),
                ("macro_f1", retained_results["test"]["macro_f1"]),
                (
                    "matthews_correlation_coefficient",
                    retained_results["test"]["matthews_correlation_coefficient"],
                ),
            )
        ],
    )
    write_csv(
        "per-class-metrics.csv",
        ["class", "precision", "recall", "f1", "support", "evidence_status"],
        [
            {"class": label, **values, "evidence_status": retained_results["evidence_status"]}
            for label, values in retained_results["test"]["per_class"].items()
        ],
    )
    write_csv(
        "duplicate-audit.csv",
        ["class", "left_path", "right_path", "sha256", "evidence_status"],
        [
            {
                "class": duplicates["class"],
                "left_path": pair["members"][0],
                "right_path": pair["members"][1],
                "sha256": pair["sha256"],
                "evidence_status": duplicates["evidence_status"],
            }
            for pair in duplicates["duplicate_groups"]
        ],
    )
    write_csv(
        "split-summary.csv",
        [
            "partition",
            "images",
            "exact_identity_groups",
            "Normal",
            "Lung Opacity",
            "Viral Pneumonia",
        ],
        [
            {
                "partition": name,
                "images": values["images"],
                "exact_identity_groups": values["exact_identity_groups"],
                **values["class_counts"],
            }
            for name, values in retained_results["split"]["partitions"].items()
        ],
    )
    labels = retained_results["provenance"]["label_order"]
    write_csv(
        "confusion-matrix.csv",
        ["actual_class", *labels],
        [
            {"actual_class": label, **dict(zip(labels, row, strict=True))}
            for label, row in zip(
                labels,
                retained_results["test"]["confusion_matrix"],
                strict=True,
            )
        ],
    )
    composition_svg(spec)
    model_results_svg(retained_results)
    write_notebook(retained_results, spec)


if __name__ == "__main__":
    main()
