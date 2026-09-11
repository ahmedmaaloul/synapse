# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Packaging and attribution invariants.

This package is Apache-2.0 on its own — the backend it talks to is not — so the
LICENSE and NOTICE shipped here must be the Apache pair, declared in pyproject
and copied into the container image, never the repo-root PolyForm ones.
Version numbers live in three places (pyproject, the installed metadata,
server.json) and the release workflow refuses to ship if they drift.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import synapse_graphrag

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

SPDX_HEADER = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>\n"
    "# Synapse — https://github.com/ahmedmaaloul/synapse\n"
)


def _pyproject() -> dict:
    return tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())


def test_license_is_the_apache_2_0_text():
    text = (PACKAGE_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("Apache License")
    assert "Version 2.0" in text


def test_notice_names_the_apache_license_and_the_author():
    text = (PACKAGE_ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Apache License, Version 2.0" in text
    assert "Ahmed Maaloul" in text


def test_pyproject_declares_license_files_and_scripts():
    project = _pyproject()["project"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    assert project["urls"]["License"].endswith("/packages/synapse-graphrag/LICENSE")
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
    assert "COPY packages/synapse-graphrag/LICENSE packages/synapse-graphrag/NOTICE /app/" in dockerfile
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
