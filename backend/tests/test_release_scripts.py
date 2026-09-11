# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""The release scripts under the repo-root ``scripts/`` — hermetic, on fixtures.

They are dependency-free and run by ``release.yml`` and ``make release-check``,
so nothing else exercises them. ``changelog_section`` matters most: its output
becomes the GitHub Release body, where a relative ``](./docs/x.md)`` link
resolves against ``/releases/tag/vX.Y.Z`` and 404s.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str) -> ModuleType:
    path = SCRIPTS / f"{name}.py"
    if not path.is_file():
        pytest.skip(f"{path} not found — running outside the monorepo")
    spec = importlib.util.spec_from_file_location(f"release_scripts.{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHANGELOG = """\
# Changelog

## [Unreleased]

## [0.4.0] — 2026-09-09

### Added

- An MCP server; see the [guide](./docs/mcp.md) and [`docs/finops.md`](./docs/finops.md).
- Nothing else.

## [0.3.0] — 2026-06-01

- Older.

[Unreleased]: https://github.com/ahmedmaaloul/synapse/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/ahmedmaaloul/synapse/releases/tag/v0.4.0
[0.3.0]: https://github.com/ahmedmaaloul/synapse/compare/v0.2.0...v0.3.0
"""


@pytest.fixture
def changelog_root(tmp_path: Path) -> Path:
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    return tmp_path


class TestChangelogSection:
    def test_prints_the_section_with_repo_links_pinned_to_the_tag(self, changelog_root, capsys):
        mod = _load("changelog_section")
        assert mod.main(["v0.4.0", "--root", str(changelog_root)]) == 0
        out = capsys.readouterr().out
        assert out.startswith("### Added")
        assert "Older." not in out and "[0.4.0]:" not in out
        assert "](./" not in out
        assert "](https://github.com/ahmedmaaloul/synapse/blob/v0.4.0/docs/mcp.md)" in out
        assert "](https://github.com/ahmedmaaloul/synapse/blob/v0.4.0/docs/finops.md)" in out

    def test_missing_or_empty_section_fails(self, changelog_root, capsys):
        mod = _load("changelog_section")
        assert mod.main(["9.9.9", "--root", str(changelog_root)]) == 1
        assert "no `## [9.9.9]` section" in capsys.readouterr().err
        assert mod.main(["Unreleased", "--root", str(changelog_root)]) == 1
        assert "is empty" in capsys.readouterr().err

    def test_the_notes_for_this_checkout_carry_no_relative_links(self, capsys):
        """What the release workflow would publish for the version in backend/pyproject.toml."""
        pyproject = (REPO_ROOT / "backend" / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
        assert match is not None
        mod = _load("changelog_section")
        if mod.main([match.group(1)]) != 0:
            pytest.skip(f"CHANGELOG.md has no `## [{match.group(1)}]` section yet")
        assert "](./" not in capsys.readouterr().out
