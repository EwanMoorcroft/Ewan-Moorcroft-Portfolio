"""Check existing QGIS exports without rebuilding or exporting the project.

Run with scripts/run_python.sh. Requires Poppler (pdfinfo, pdftotext, pdffonts,
pdftoppm) and Pillow. Previews are written under work/qgis-qa/pdf by default.
Visual review is carried forward only when all decoded PNG and PDF-preview
pixels match a previously reviewed report. --record-visual-review records an
actual review already performed by the caller; it does not perform one.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile
from xml.etree import ElementTree

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
NAMES = ["01_overview", "02_flood", "03_heat", "04_accessibility", "05_priorities", "06_robustness"]
REPORT = ROOT / "metadata/qgis_pdf_qa.json"
BUILD = ROOT / "metadata/qgis_build.json"
PROJECT = ROOT / "qgis/project/gm_climate_resilience.qgz"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixels(path):
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        return {"pixels": list(rgba.size), "rgba_sha256": hashlib.sha256(rgba.tobytes()).hexdigest()}


def poppler(name):
    located = shutil.which(name)
    if located:
        return located
    # The macOS QGIS launcher deliberately has a minimal PATH.
    for directory in ("/opt/homebrew/bin", "/usr/local/bin"):
        candidate = Path(directory) / name
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(f"Poppler command {name} is required on PATH for independent PDF QA.")


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, check=True, timeout=60)
    return result.stdout, result.stderr.strip()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run(args):
    build = json.loads(BUILD.read_text())
    previous = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    paths = [PROJECT, ROOT / build["input"]]
    paths += [ROOT / "outputs/maps" / (name + extension) for name in NAMES for extension in (".png", ".pdf")]
    hashes = {path: sha256(path) for path in paths}
    input_path = ROOT / build["input"]
    if hashes[input_path] != build["input_integrity"]["sha256_after"]:
        raise ValueError("Current processed input differs from the successful build report.")
    with zipfile.ZipFile(PROJECT) as archive:
        xml = ElementTree.fromstring(archive.read(next(name for name in archive.namelist() if name.endswith(".qgs"))))
    labels = {
        layout.get("name"): [item.get("labelText") for item in layout.findall(".//LayoutItem") if item.get("labelText")]
        for layout in xml.findall(".//Layouts/Layout")
    }
    if sorted(labels) != NAMES:
        raise ValueError("Expected exactly the six named print layouts.")
    executables = {name: poppler(name) for name in ("pdfinfo", "pdftotext", "pdffonts", "pdftoppm")}
    args.previews.mkdir(parents=True, exist_ok=True)
    normalize = lambda value: " ".join(value.split())

    def check(name):
        pdf = ROOT / "outputs/maps" / (name + ".pdf")
        png = pdf.with_suffix(".png")
        info, _ = command(executables["pdfinfo"], str(pdf))
        text, _ = command(executables["pdftotext"], str(pdf), "-")
        font_output, _ = command(executables["pdffonts"], str(pdf))
        details = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in info.splitlines() if ":" in line}
        if details.get("Pages") != "1":
            raise ValueError(f"{name}: expected one PDF page.")
        width, height = map(float, re.match(r"([0-9.]+) x ([0-9.]+)", details["Page size"]).groups())
        if abs(width - 1190.55) >= 1 or abs(height - 841.89) >= 1:
            raise ValueError(f"{name}: expected A3 landscape, found {details['Page size']}.")
        required_text = ["Independent portfolio case study", "OpenStreetMap contributors", "Insufficient evidence"]
        if any(fragment not in text for fragment in required_text):
            raise ValueError(f"{name}: required source, limitation or missing-evidence label absent.")
        missing = [label for label in labels[name] if normalize(label) not in normalize(text)]
        if missing:
            raise ValueError(f"{name}: PDF does not contain current layout labels: {missing}")
        fonts = font_output.splitlines()[2:]
        if not fonts or any("Arial" not in row or row.split()[-5] != "yes" for row in fonts):
            raise ValueError(f"{name}: expected embedded Arial fonts, found {fonts}")
        preview = args.previews / name
        _, stderr = command(executables["pdftoppm"], "-scale-to", "1600", "-png", "-singlefile", str(pdf), str(preview))
        rendered = pixels(preview.with_suffix(".png"))
        if args.reference_render_directory:
            reference = pixels(args.reference_render_directory / (name + ".png"))
            if reference != rendered:
                raise ValueError(f"{name}: current PDF render differs from the reference render.")
        return {
            "file": str(pdf.relative_to(ROOT)), "sha256": hashes[pdf], "pages": 1,
            "page_size": details["Page size"], "text_characters": len(text), "fonts": fonts,
            "footer_and_missing_class_present": True, "all_current_layout_labels_present": True,
            "layout_labels_checked": len(labels[name]), "render_status": "passed", "render_stderr": stderr,
            "render": rendered,
            "png": {"file": str(png.relative_to(ROOT)), "sha256": hashes[png], **pixels(png)},
            "reference_render_pixel_identical": True if args.reference_render_directory else None,
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(check, NAMES))
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError("An input or export changed during QA; evidence was not recorded.")
    old_rows = {row["file"]: row for row in previous.get("pdfs", [])}
    review = previous.get("visual_review", {})
    old_passed = isinstance(review, dict) and review.get("status") == "passed"
    identical = old_passed and all(
        row["render"] == old_rows.get(row["file"], {}).get("render")
        and row["png"]["rgba_sha256"] == old_rows.get(row["file"], {}).get("png", {}).get("rgba_sha256")
        and row["png"]["pixels"] == old_rows.get(row["file"], {}).get("png", {}).get("pixels")
        for row in rows
    )
    reviewed = args.record_visual_review or identical
    now = datetime.now(timezone.utc).isoformat()
    if args.record_visual_review:
        method = "Caller confirmed visual review of current PNGs and PDF previews, including comparison with previously reviewed identical renders where recorded."
    elif identical:
        method = "All six PNG exports and all six independent PDF renders are pixel-identical to the previously visually reviewed report."
    else:
        method = "Automated export checks passed; inspect all six PNGs and the PDF previews before recording visual review."
    report = {
        "checked_at_utc": now, "status": "passed", "project_sha256": hashes[PROJECT],
        "input_sha256": hashes[input_path], "pdfs": rows,
        "visual_review": {"status": "passed" if reviewed else "pending", "method": method,
                          "reviewed_at_utc": now if args.record_visual_review else review.get("reviewed_at_utc") if old_passed else None,
                          "pixel_identical_to_previous_report": identical,
                          "scope": "Visual/export integrity; not independent scientific verification of the analysis."},
    }
    for row in rows:
        row["visually_reviewed"] = reviewed
    save(REPORT, report)
    build["visual_qa"] = {
        "status": "passed" if reviewed else "pending", "report": str(REPORT.relative_to(ROOT)),
        "checked_at_utc": now, "project_sha256": hashes[PROJECT],
        "reviewed_pngs": 6 if reviewed else 0, "independently_rendered_pdfs": 6,
        "all_current_pdf_layout_labels_present": True, "fonts_embedded": ["ArialMT", "Arial-BoldMT"],
    }
    build["input_integrity"]["sha256_at_export_qa"] = hashes[input_path]
    save(BUILD, build)
    print(f"PDF checks: 6 passed; current layout labels: all present; visual review: {'passed' if reviewed else 'pending'}")
    print(f"Report: {REPORT.relative_to(ROOT)}")
    print(f"Previews: {args.previews}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previews", type=Path, default=ROOT / "work/qgis-qa/pdf")
    parser.add_argument("--record-visual-review", action="store_true", help="Record actual visual review already performed by the caller.")
    parser.add_argument("--reference-render-directory", type=Path, help="Require exact decoded pixel identity with six existing PDF preview PNGs.")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        failure = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "status": "failed", "error": str(error), "visual_review": {"status": "pending"}}
        save(REPORT, failure)
        if BUILD.exists():
            build = json.loads(BUILD.read_text())
            build["visual_qa"] = {"status": "failed", "report": str(REPORT.relative_to(ROOT)), "error": str(error)}
            save(BUILD, build)
        raise


if __name__ == "__main__":
    main()
