from __future__ import annotations

import base64
import re
import struct
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".svg", ".toml", ".txt"}
RAW_SUFFIXES = {".las", ".laz", ".npz", ".npy", ".pth"}
TRANSIENT_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}

WORD_CODES = (
    "dW5pdmVyc2l0eQ==",
    "YXNzaWdubWVudA==",
    "Y291cnNld29yaw==",
    "YXNzZXNzbWVudA==",
    "c3R1ZGVudA==",
    "bGVjdHVyZXI=",
    "Z3JhZGU=",
    "bWFya3M=",
    "c3VibWlzc2lvbg==",
    "bW9kdWxl",
    "ZGlzc2VydGF0aW9u",
    "Y2Ex",
    "Y2Ey",
    "Y2hhdGdwdA==",
    "Y29kZXg=",
    "b3BlbmFp",
    "Y2xhdWRl",
    "Z2VtaW5p",
    "YWktZ2VuZXJhdGVk",
    "YXJ0aWZpY2lhbCBpbnRlbGxpZ2VuY2U=",
    "YXNzaXN0YW50",
    "YXV0b21hdGlvbg==",
    "YXV0aG9yZWQgYnk=",
    "Z2VuZXJhdGVkIGJ5",
    "YmFya2xh",
    "c2x1cm0=",
    "aHBj",
    "YWk=",
)
SUBSTRING_CODES = (
    "L3VzZXJzLw==",
    "ZXdhbmQx",
    "Zm9yX2luc3RhbmNlX3BvaW50d2lzZV92MQ==",
    "ZmlsZTovLw==",
    "c3NoOi8v",
    "bG9jYWxob3N0",
    "MTI3LjAuMC4x",
)


def _decode(value: str) -> str:
    return base64.b64decode(value).decode("utf-8")


def _is_transient(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts
    return any(part in TRANSIENT_PARTS or part.endswith(".egg-info") for part in parts)


def _png_text(path: Path) -> list[str]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Invalid PNG signature: {path.name}")
    offset = 8
    values: list[str] = []
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += length + 12
        if kind == b"tEXt":
            values.append(payload.decode("latin-1"))
        elif kind == b"zTXt":
            keyword, compressed = payload.split(b"\0", 1)
            values.append(
                keyword.decode("latin-1") + ":" + zlib.decompress(compressed[1:]).decode("latin-1")
            )
    return values


class PublicContentTests(unittest.TestCase):
    def test_text_and_paths_pass_content_gate(self) -> None:
        word_patterns = [
            re.compile(
                rf"(?<![a-z0-9_]){re.escape(_decode(code))}(?![a-z0-9_])",
                re.IGNORECASE,
            )
            for code in WORD_CODES
        ]
        substring_values = [_decode(code).casefold() for code in SUBSTRING_CODES]
        for path in ROOT.rglob("*"):
            if path.is_dir() or _is_transient(path):
                continue
            relative = path.relative_to(ROOT).as_posix()
            candidate = relative
            if path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore":
                candidate += "\n" + path.read_text(encoding="utf-8")
            elif path.suffix.lower() == ".png":
                candidate += "\n" + "\n".join(_png_text(path))
            for pattern in word_patterns:
                self.assertIsNone(pattern.search(candidate), relative)
            folded = candidate.casefold()
            for value in substring_values:
                self.assertNotIn(value, folded, relative)

    def test_no_raw_arrays_or_point_clouds(self) -> None:
        retained_suffixes = {
            path.suffix.lower()
            for path in ROOT.rglob("*")
            if path.is_file() and not _is_transient(path)
        }
        self.assertTrue(retained_suffixes.isdisjoint(RAW_SUFFIXES))

    def test_no_retained_links_or_compiled_cache(self) -> None:
        for path in ROOT.rglob("*"):
            if _is_transient(path):
                continue
            self.assertFalse(path.is_symlink(), path.relative_to(ROOT).as_posix())
            self.assertNotEqual(path.suffix.lower(), ".pyc")


if __name__ == "__main__":
    unittest.main()
