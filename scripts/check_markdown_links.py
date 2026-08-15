"""Validate relative file links in Markdown documents."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import unquote

SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")
SKIP_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}


def _parenthesized_target(markdown: str, start: int) -> tuple[str, int] | None:
    """Parse one inline-link target, respecting balanced destination parentheses."""
    depth = 1
    escaped = False
    in_angle_destination = False
    title_quote: str | None = None
    index = start
    while index < len(markdown):
        character = markdown[index]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif in_angle_destination:
            if character == ">":
                in_angle_destination = False
        elif title_quote is not None:
            if character == title_quote:
                title_quote = None
        elif character == "<" and not markdown[start:index].strip():
            in_angle_destination = True
        elif character in {'"', "'"} and index > start and markdown[index - 1].isspace():
            title_quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return markdown[start:index], index
        elif character in "\r\n":
            return None
        index += 1
    return None


def _inline_destinations(markdown: str) -> list[str]:
    """Return valid inline Markdown destinations, including nested image links."""
    destinations: list[str] = []
    bracket_stack: list[int] = []
    escaped = False
    index = 0
    while index < len(markdown):
        character = markdown[index]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "[":
            bracket_stack.append(index)
        elif character == "]" and bracket_stack:
            bracket_stack.pop()
            if index + 1 < len(markdown) and markdown[index + 1] == "(":
                parsed = _parenthesized_target(markdown, index + 2)
                if parsed is not None:
                    target, closing = parsed
                    destinations.append(target)
                    index = closing
        index += 1
    return destinations


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
        for raw_target in _inline_destinations(markdown.read_text(encoding="utf-8")):
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
