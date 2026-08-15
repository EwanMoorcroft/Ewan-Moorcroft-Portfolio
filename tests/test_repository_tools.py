"""Tests for repository-level quality tools."""

from __future__ import annotations

import re
import runpy
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> dict[str, object]:
    """Load one repository script for direct function testing."""
    path = ROOT / "scripts" / f"{name}.py"
    return runpy.run_path(str(path), run_name=f"quality_{name}")


def test_link_checker_accepts_existing_relative_target(tmp_path: Path) -> None:
    """Existing relative Markdown targets should pass the link check."""
    checker = _load_script("check_markdown_links")
    (tmp_path / "target.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "README.md").write_text("[target](target.txt)", encoding="utf-8")
    assert checker["broken_links"](tmp_path) == []


def test_link_checker_reports_missing_target(tmp_path: Path) -> None:
    """Missing relative Markdown targets should be reported."""
    checker = _load_script("check_markdown_links")
    (tmp_path / "README.md").write_text("[missing](missing.txt)", encoding="utf-8")
    assert checker["broken_links"](tmp_path) == [(Path("README.md"), "missing.txt")]


def test_link_checker_accepts_optional_title(tmp_path: Path) -> None:
    """A standard optional link title should not become part of the path."""
    checker = _load_script("check_markdown_links")
    (tmp_path / "target.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "README.md").write_text('[target](target.txt "More detail")', encoding="utf-8")
    assert checker["broken_links"](tmp_path) == []


def test_link_checker_accepts_balanced_parentheses_in_destination(tmp_path: Path) -> None:
    """Balanced parentheses are valid characters in an inline link destination."""
    checker = _load_script("check_markdown_links")
    (tmp_path / "report_(final).txt").write_text("ok", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "[report](report_(final).txt)",
        encoding="utf-8",
    )
    assert checker["broken_links"](tmp_path) == []


def test_link_checker_keeps_parentheses_inside_optional_title(tmp_path: Path) -> None:
    """Parentheses in a quoted title must not terminate the inline destination."""
    checker = _load_script("check_markdown_links")
    (tmp_path / "report.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        '[report](report.txt "Final (reviewed) version")',
        encoding="utf-8",
    )
    assert checker["broken_links"](tmp_path) == []


def test_link_checker_reports_full_missing_parenthesized_target(tmp_path: Path) -> None:
    """A missing parenthesized path should be reported without truncation."""
    checker = _load_script("check_markdown_links")
    (tmp_path / "README.md").write_text(
        "[report](missing_(final).txt)",
        encoding="utf-8",
    )
    assert checker["broken_links"](tmp_path) == [(Path("README.md"), "missing_(final).txt")]


def test_link_checker_rejects_target_outside_root(tmp_path: Path) -> None:
    """A path escape must not pass merely because it exists on the machine."""
    checker = _load_script("check_markdown_links")
    repository = tmp_path / "repository"
    repository.mkdir()
    (tmp_path / "outside.txt").write_text("private", encoding="utf-8")
    (repository / "README.md").write_text("[outside](../outside.txt)", encoding="utf-8")
    assert checker["broken_links"](repository) == [(Path("README.md"), "../outside.txt")]


def test_link_checker_skips_local_environment(tmp_path: Path) -> None:
    """Local environment documentation should not affect repository links."""
    checker = _load_script("check_markdown_links")
    environment = tmp_path / ".venv"
    environment.mkdir()
    (environment / "README.md").write_text("[missing](missing.txt)", encoding="utf-8")
    assert checker["broken_links"](tmp_path) == []


def test_repository_auditor_flags_machine_home_path(tmp_path: Path) -> None:
    """Machine-specific home paths should fail the public-safety audit."""
    auditor = _load_script("audit_repository")
    home_fragment = "/Users/" + "example" + "/private.csv"
    (tmp_path / "example.py").write_text(f"path = '{home_fragment}'", encoding="utf-8")
    findings = auditor["audit_tree"](tmp_path)
    assert any(finding.detail == "machine-specific home path" for finding in findings)


def test_repository_auditor_flags_windows_home_path(tmp_path: Path) -> None:
    """Machine-specific Windows home paths should also fail the audit."""
    auditor = _load_script("audit_repository")
    home_fragment = "C:\\" + "Users\\example\\private.csv"
    (tmp_path / "example.py").write_text(f"path = {home_fragment!r}", encoding="utf-8")
    findings = auditor["audit_tree"](tmp_path)
    assert any(finding.detail == "machine-specific home path" for finding in findings)


def test_repository_auditor_scans_vector_text(tmp_path: Path) -> None:
    """Text-bearing vector assets should receive the same privacy scan."""
    auditor = _load_script("audit_repository")
    home_fragment = "/home/" + "example/private.csv"
    (tmp_path / "example.svg").write_text(
        f"<svg><text>{home_fragment}</text></svg>", encoding="utf-8"
    )
    findings = auditor["audit_tree"](tmp_path)
    assert any(finding.detail == "machine-specific home path" for finding in findings)


def test_repository_auditor_scans_r_source(tmp_path: Path) -> None:
    """R source must receive the same machine-path and terminology checks."""
    auditor = _load_script("audit_repository")
    home_fragment = "/home/" + "example/private.csv"
    (tmp_path / "analysis.R").write_text(f"path <- '{home_fragment}'", encoding="utf-8")
    findings = auditor["audit_tree"](tmp_path)
    assert any(finding.detail == "machine-specific home path" for finding in findings)


def test_repository_auditor_scans_geojson(tmp_path: Path) -> None:
    """Text geodata properties must not bypass public-safety checks."""
    auditor = _load_script("audit_repository")
    home_fragment = "/home/" + "example/private.geojson"
    (tmp_path / "areas.geojson").write_text(
        '{"type":"Feature","properties":{"source":"' + home_fragment + '"}}',
        encoding="utf-8",
    )
    findings = auditor["audit_tree"](tmp_path)
    assert any(finding.detail == "machine-specific home path" for finding in findings)


def test_repository_auditor_flags_authorship_term(tmp_path: Path) -> None:
    """Public prose should not contain tool-authorship terminology."""
    auditor = _load_script("audit_repository")
    term = "auto" + "mation"
    (tmp_path / "README.md").write_text(f"Created through {term}.", encoding="utf-8")
    findings = auditor["audit_tree"](tmp_path)
    assert any(finding.detail == f"blocked content term: {term}" for finding in findings)


def test_repository_auditor_flags_typographic_dashes(tmp_path: Path) -> None:
    """Public text should use plain punctuation instead of typographic dash characters."""
    auditor = _load_script("audit_repository")
    dash = chr(0x2014)
    (tmp_path / "README.md").write_text(f"First thought {dash} second thought.", encoding="utf-8")
    findings = auditor["audit_tree"](tmp_path)
    assert any(finding.detail == "typographic dash in public text" for finding in findings)


def test_repository_auditor_flags_unfinished_markers(tmp_path: Path) -> None:
    """Unfinished source markers should not be published as completed work."""
    auditor = _load_script("audit_repository")
    marker = "TO" + "DO"
    (tmp_path / "model.py").write_text(f"# {marker}: add the real fit path", encoding="utf-8")
    findings = auditor["audit_tree"](tmp_path)
    assert any(finding.detail == f"unfinished marker: {marker}" for finding in findings)


def test_repository_auditor_flags_leaked_instruction_text(tmp_path: Path) -> None:
    """Prompt-like directions and implementation stubs should not reach public prose."""
    auditor = _load_script("audit_repository")
    leaked = "replace with your " + "implementation"
    (tmp_path / "README.md").write_text(leaked, encoding="utf-8")
    findings = auditor["audit_tree"](tmp_path)
    assert any(finding.detail == "possible leaked instruction or stub text" for finding in findings)


def test_repository_auditor_allows_confirmed_education(tmp_path: Path) -> None:
    """Confirmed public education should be valid portfolio content."""
    auditor = _load_script("audit_repository")
    (tmp_path / "README.md").write_text(
        "MSc Data Science and Artificial Intelligence student at the "
        "University of Liverpool, with a BSc in Geography.",
        encoding="utf-8",
    )
    assert auditor["audit_tree"](tmp_path) == []


def test_repository_auditor_still_blocks_raw_academic_material(tmp_path: Path) -> None:
    """Allowing education must not allow source academic material into the public tree."""
    auditor = _load_script("audit_repository")
    restricted = "assign" + "ment"
    (tmp_path / "README.md").write_text(
        f"Original {restricted} materials are included.", encoding="utf-8"
    )
    findings = auditor["audit_tree"](tmp_path)
    assert any(finding.detail == f"blocked content term: {restricted}" for finding in findings)


def test_repository_auditor_skips_operating_system_metadata(tmp_path: Path) -> None:
    """Ignored desktop metadata should not be decoded as repository text."""
    auditor = _load_script("audit_repository")
    (tmp_path / ".DS_Store").write_bytes(b"\x00\xff")
    (tmp_path / "Thumbs.db").write_bytes(b"\x00\xff")
    assert auditor["audit_tree"](tmp_path) == []


def test_repository_auditor_allows_technical_identifier(tmp_path: Path) -> None:
    """A Python alias may use the otherwise restricted term as a technical suffix."""
    auditor = _load_script("audit_repository")
    term = "mo" + "dule"
    (tmp_path / "example.py").write_text(f"evaluation_{term} = object()", encoding="utf-8")
    findings = auditor["audit_tree"](tmp_path)
    assert not any(finding.detail == f"blocked content term: {term}" for finding in findings)


def test_repository_auditor_flags_modern_key_shapes(tmp_path: Path) -> None:
    """Common current credential shapes should fail without storing literal examples here."""
    auditor = _load_script("audit_repository")
    tokens = [
        "github" + "_pat_" + "A" * 24,
        "sk" + "-proj-" + "A" * 24,
    ]
    for index, token in enumerate(tokens):
        (tmp_path / f"secret-{index}.txt").write_text(token, encoding="utf-8")
    findings = auditor["audit_tree"](tmp_path)
    secret_findings = [item for item in findings if item.detail == "possible secret material"]
    assert len(secret_findings) == len(tokens)


def test_ci_matrix_matches_project_test_contracts() -> None:
    """Every packaged project should expose the commands used by the quality matrix."""
    workflow = (ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    matrix_projects = re.findall(r"^ {10}- ([a-z0-9-]+)$", workflow, flags=re.MULTILINE)
    project_root = ROOT / "projects"
    packaged_projects = sorted(
        path.name for path in project_root.iterdir() if (path / "pyproject.toml").is_file()
    )
    assert len(matrix_projects) == len(set(matrix_projects))
    assert set(matrix_projects) == set(packaged_projects)

    for command in (
        "uv sync --extra test",
        "uv run pytest",
        "uv run ruff check .",
        "uv run ruff format --check .",
    ):
        assert command in workflow

    for project in packaged_projects:
        config = tomllib.loads(
            (project_root / project / "pyproject.toml").read_text(encoding="utf-8")
        )
        test_requirements = config["project"]["optional-dependencies"]["test"]
        assert any(requirement.startswith("pytest") for requirement in test_requirements)
        assert any(requirement.startswith("ruff") for requirement in test_requirements)

    assert "uv pip install --python .venv/bin/python -r requirements-dev.txt" in workflow
    assert "uv run --no-sync pytest" in workflow
    assert "uses: actions/checkout@v" not in workflow
    assert "uses: astral-sh/setup-uv@v" not in workflow
    assert "uses: r-lib/actions/setup-r@v" not in workflow
