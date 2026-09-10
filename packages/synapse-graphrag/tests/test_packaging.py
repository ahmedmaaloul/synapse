# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Packaging and attribution invariants.

NOTICE requires LICENSE and NOTICE to travel with every distribution, so the
copies inside this package must be byte-identical to the repo-root originals.
Version numbers live in three places (pyproject, the installed metadata,
server.json) and the release workflow refuses to ship if they drift.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

import synapse_graphrag

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]

SPDX_HEADER = (
    "# SPDX-License-Identifier: AGPL-3.0-or-later\n"
    "# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>\n"
    "# Synapse — https://github.com/ahmedmaaloul/synapse\n"
)


def _pyproject() -> dict:
    return tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())


@pytest.mark.parametrize("name", ["LICENSE", "NOTICE"])
def test_license_files_identical_to_repo_root(name: str):
    original = REPO_ROOT / name
    if not original.is_file():
        pytest.skip(f"{name} not found at the repo root — running outside the monorepo")
    assert (PACKAGE_ROOT / name).read_bytes() == original.read_bytes(), (
        f"packages/synapse-graphrag/{name} drifted from the repo-root {name}"
    )


def test_pyproject_declares_license_files_and_scripts():
    project = _pyproject()["project"]
    assert project["license"] == "AGPL-3.0-or-later"
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    assert not any(c.startswith("License ::") for c in project["classifiers"])
    assert project["scripts"] == {
        "synapse-graphrag": "synapse_graphrag.cli:main",
        "synapse-mcp": "synapse_graphrag.mcp_server:main",
    }
    assert project["requires-python"] == ">=3.11"
    assert {d.split(">=")[0] for d in project["dependencies"]} == {"mcp", "httpx", "pydantic"}


def test_versions_agree():
    version = _pyproject()["project"]["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+([-.].+)?", version)
    assert synapse_graphrag.__version__ == version, "install the package (pip install -e .) first"
    registry = json.loads((PACKAGE_ROOT / "server.json").read_text())
    assert registry["version"] == version
    assert registry["packages"][0]["version"] == version
    assert registry["packages"][0]["identifier"] == "synapse-graphrag"


def test_every_python_file_has_the_spdx_header():
    files = sorted((PACKAGE_ROOT / "src").rglob("*.py")) + sorted((PACKAGE_ROOT / "tests").rglob("*.py"))
    assert files
    missing = [str(f.relative_to(PACKAGE_ROOT)) for f in files if not f.read_text().startswith(SPDX_HEADER)]
    assert missing == []


def test_dockerfile_has_the_spdx_header_and_ships_notices():
    dockerfile = (PACKAGE_ROOT / "Dockerfile").read_text()
    assert dockerfile.startswith(SPDX_HEADER)
    assert "COPY LICENSE NOTICE /app/" in dockerfile
    assert "EXPOSE 8765" in dockerfile
    assert '"synapse-mcp", "--transport", "streamable-http"' in dockerfile
    assert re.search(r"^USER \w+", dockerfile, re.MULTILINE), "the image must not run as root"


def test_server_json_matches_the_registry_schema_shape():
    registry = json.loads((PACKAGE_ROOT / "server.json").read_text())
    assert registry["$schema"].startswith("https://static.modelcontextprotocol.io/schemas/")
    assert registry["name"] == "io.github.ahmedmaaloul/synapse"
    # The registry schema caps `description` at 100 characters; mcp-publisher rejects longer.
    assert len(registry["description"]) <= 100
    package = registry["packages"][0]
    assert package["registryType"] == "pypi"
    assert package["transport"] == {"type": "stdio"}
    assert package["runtimeHint"] == "uvx"
    env_names = {e["name"] for e in package["environmentVariables"]}
    assert "SYNAPSE_URL" in env_names
