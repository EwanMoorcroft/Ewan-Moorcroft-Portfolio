"""Validate relative file links in Markdown documents."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote

LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")
SKIP_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}


def _destination(raw_target: str) -> str:
    """Extract a destination while discarding an optional Markdown link title."""
    target = raw_target.strip()
    if target.startswith("<"):
        closing = target.find(">")
        return target[1:closing] if closing >= 0 else target.strip("<>")
    return target.split(maxsplit=1)[0]


def broken_links(root: Path) -> list[tuple[Path, str]]:
    """Return relative Markdown links whose local targets do not exist."""
    root = root.resolve()
    broken: list[tuple[Path, str]] = []
    for markdown in sorted(root.rglob("*.md")):
        if any(part in SKIP_PARTS for part in markdown.parts):
            continue
        for raw_target in LINK_PATTERN.findall(markdown.read_text(encoding="utf-8")):
            target = _destination(raw_target)
            if target.lower().startswith(SKIP_PREFIXES):
                continue
            path_part = unquote(target.split("#", maxsplit=1)[0])
            if not path_part:
                continue
            candidate = (markdown.parent / path_part).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                broken.append((markdown.relative_to(root), raw_target))
                continue
            if not candidate.exists():
                broken.append((markdown.relative_to(root), raw_target))
    return broken


def main() -> int:
    """Run the link check and return a shell-friendly status code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    failures = broken_links(root)
    if failures:
        for path, target in failures:
            print(f"{path}: invalid or missing target {target}")
        return 1
    print("Markdown link check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
