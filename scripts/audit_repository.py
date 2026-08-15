"""Check a repository tree for public-safety and portability risks."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".csv",
    ".dockerfile",
    ".html",
    ".ini",
    ".ipynb",
    ".json",
    ".key",
    ".md",
    ".pem",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".svg",
    ".xml",
    ".yaml",
    ".yml",
}
SKIP_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}
SKIP_FILENAMES = {".DS_Store", "Thumbs.db"}
MAX_PUBLIC_FILE_BYTES = 10 * 1024 * 1024


def _joined(*parts: str) -> str:
    """Build a scan term without embedding that term in the scanner itself."""
    return "".join(parts)


BLOCKED_PHRASES = (
    _joined("assign", "ment"),
    _joined("course", "work"),
    _joined("assess", "ment"),
    _joined("lec", "turer"),
    _joined("gra", "de"),
    _joined("mar", "ks"),
    _joined("sub", "mission"),
    _joined("mo", "dule"),
    _joined("disser", "tation"),
    _joined("chat", "gpt"),
    _joined("co", "dex"),
    _joined("open", "a", "i"),
    _joined("clau", "de"),
    _joined("gem", "ini"),
    _joined("a", "i", "-generated"),
    _joined("llm", "-generated"),
    _joined("assis", "tant", "-generated"),
    _joined("assis", "tant"),
    _joined("auto", "mation"),
    _joined("auto", "mated"),
    _joined("auto", "generated"),
)
BLOCKED_PATTERNS = tuple(
    (
        phrase,
        re.compile(r"(?i)(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])"),
    )
    for phrase in BLOCKED_PHRASES
)
PRIVATE_FRAGMENTS = (
    _joined("/Users/", "ew", "and1"),
    _joined("ew", "and1"),
    _joined("bar", "kla"),
    _joined("slu", "rm"),
)
SECRET_PATTERNS = (
    re.compile(_joined("AK", "IA") + r"[0-9A-Z]{16}"),
    re.compile(_joined("gh", "p_") + r"[A-Za-z0-9]{30,}"),
    re.compile(r"\b" + _joined("github", "_pat_") + r"[A-Za-z0-9_]{20,}"),
    re.compile(r"\b" + _joined("sk", "-") + r"(?:proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(_joined("-----BEGIN ", "PRIVATE KEY-----")),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )PRIVATE KEY-----"),
)
STANDALONE_SHORT_TERM = re.compile(r"(?i)\b" + "A" + "I" + r"\b")
LEGACY_LABEL = re.compile(r"(?i)\bC" + r"A(?:\s*[12])?\b")
HOME_PATH = re.compile(r"(?:/Users/|/home/)[A-Za-z0-9._-]+/")
WINDOWS_HOME_PATH = re.compile(r"(?i)\b[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9._-]+[\\/]")
TECHNICAL_BASE_TYPE = re.compile(r"\b(?:nn|torch\.nn)\." + _joined("Mo", "dule") + r"\b")
TECHNICAL_IDENTIFIER_TERM = re.compile(
    r"(?i)(?:_" + _joined("mo", "dule") + r"\b|\b" + _joined("mo", "dule") + r"_)"
)


@dataclass(frozen=True)
class Finding:
    """One repository audit finding."""

    path: Path
    detail: str


def _is_text_candidate(path: Path) -> bool:
    """Return whether a file should be decoded and inspected as text."""
    return path.name == "Dockerfile" or path.suffix.lower() in TEXT_SUFFIXES


def _iter_files(root: Path):
    """Yield files while ignoring local environments and tool caches."""
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.name in SKIP_FILENAMES:
            continue
        if path.is_file() or path.is_symlink():
            yield path


def audit_tree(root: Path) -> list[Finding]:
    """Return all portability, terminology, secret, and size findings."""
    findings: list[Finding] = []
    for path in _iter_files(root):
        relative = path.relative_to(root)
        searchable_name = relative.as_posix().lower()

        if path.is_symlink():
            findings.append(Finding(relative, "symbolic links are not allowed in the public tree"))
            continue
        if path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
            findings.append(Finding(relative, "file is larger than the 10 MiB public limit"))

        for phrase, pattern in BLOCKED_PATTERNS:
            if pattern.search(searchable_name):
                findings.append(Finding(relative, f"blocked filename term: {phrase}"))
        if STANDALONE_SHORT_TERM.search(relative.as_posix()):
            findings.append(Finding(relative, "blocked standalone acronym in filename"))
        if LEGACY_LABEL.search(relative.as_posix()):
            findings.append(Finding(relative, "blocked legacy label in filename"))

        if not _is_text_candidate(path):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(Finding(relative, "expected text file is not valid UTF-8"))
            continue

        lowered = content.lower()
        terminology_content = content
        if path.suffix == ".py":
            terminology_content = TECHNICAL_BASE_TYPE.sub("", terminology_content)
            terminology_content = TECHNICAL_IDENTIFIER_TERM.sub("_", terminology_content)
        for phrase, pattern in BLOCKED_PATTERNS:
            if pattern.search(terminology_content):
                findings.append(Finding(relative, f"blocked content term: {phrase}"))
        for fragment in PRIVATE_FRAGMENTS:
            if fragment.lower() in lowered:
                findings.append(Finding(relative, "private environment fragment"))
        if STANDALONE_SHORT_TERM.search(content):
            findings.append(Finding(relative, "blocked standalone acronym in content"))
        if LEGACY_LABEL.search(content):
            findings.append(Finding(relative, "blocked legacy label in content"))
        if HOME_PATH.search(content):
            findings.append(Finding(relative, "machine-specific home path"))
        if WINDOWS_HOME_PATH.search(content):
            findings.append(Finding(relative, "machine-specific home path"))
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                findings.append(Finding(relative, "possible secret material"))
    return findings


def main() -> int:
    """Run the repository audit and return a shell-friendly status code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    findings = audit_tree(root)
    if findings:
        for finding in findings:
            print(f"{finding.path}: {finding.detail}")
        print(f"Repository audit failed with {len(findings)} finding(s).", file=sys.stderr)
        return 1
    print("Repository audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
