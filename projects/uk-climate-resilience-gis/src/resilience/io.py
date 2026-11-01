"""Small auditable I/O primitives shared by pipeline stages."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger(__name__)


def configuration() -> dict:
    return json.loads((ROOT / "config/analysis.json").read_text())


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def session() -> requests.Session:
    client = requests.Session()
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"])
    client.mount("https://", HTTPAdapter(max_retries=retry))
    client.headers["User-Agent"] = "GM-Climate-Resilience-Research/1.0 (open-data portfolio analysis)"
    return client


def get_json(url: str, params: dict | None = None) -> Any:
    with session() as client:
        response = client.get(url, params=params, timeout=(30, 180))
        response.raise_for_status()
        return response.json()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_source(path: Path, url: str, *, params: dict | None = None, **metadata: Any) -> None:
    """Each source has its own sidecar, avoiding concurrent manifest writes."""
    save_json(path.with_name(path.name + ".source.json"), {
        "file": str(path.relative_to(ROOT)), "source_url": url,
        "parameters": params, "acquired_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256(path), "bytes": path.stat().st_size, **metadata,
    })


def download(url: str, path: Path, *, params: dict | None = None, **metadata: Any) -> Path:
    if path.exists():
        sidecar = path.with_name(path.name + ".source.json")
        if sidecar.exists() and json.loads(sidecar.read_text())["sha256"] != sha256(path):
            raise ValueError(f"Cached source checksum changed: {path}")
        LOG.info("Cached %s", path.name)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with session() as client, client.get(url, params=params, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                stream.write(block)
        expected = response.headers.get("Content-Length")
        if expected and not response.headers.get("Content-Encoding") and temporary.stat().st_size != int(expected):
            raise ValueError(f"Truncated response for {url}")
    temporary.replace(path)
    record_source(path, url, params=params, **metadata)
    LOG.info("Downloaded %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    return path


def cached_json(url: str, path: Path, params: dict | None = None, **metadata: Any) -> Any:
    download(url, path, params=params, **metadata)
    return json.loads(path.read_text())

