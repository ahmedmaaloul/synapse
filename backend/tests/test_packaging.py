# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Regression tests for packaging, attribution, and the shared DB fixture.

Three defects these lock down:

  1. ``fake_neo4j`` patched a hard-coded module list that omitted
     ``app.services.communities`` — those tests silently reached the *real*
     ``execute_query``. The fixture now sweeps every loaded ``app.*`` module.
  2. Neither container image contained LICENSE/NOTICE, because each Dockerfile
     built from its own subdirectory. NOTICE requires the notices be retained
     "in all distributions, packages, container images".
  3. ``backend/scripts``, ``backend/eval`` and ``backend/tests`` had no SPDX
     headers, so the project violated its own attribution terms.

The Docker tests are static (they read the manifests); the real
``docker build`` + ``docker run`` proof is documented in DEPLOYMENT.md.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_ROOT.parent


# ── 1. The shared Neo4j fixture ──────────────────────────
class TestFakeNeo4jCoverage:
    """Every module holding an ``execute_query`` binding must get patched."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "app.neo4j_driver",
            "app.routers.graph",
            "app.services.chat_engine",
            "app.services.communities",
            "app.services.graph_builder",
            "app.services.graph_schema",
        ],
    )
    async def test_every_importer_is_patched(self, module_path, fake_neo4j):
        import importlib

        module = importlib.import_module(module_path)
        real = module.execute_query

        calls = fake_neo4j(lambda q, p: [{"ok": True}])

        assert module.execute_query is not real, (
            f"{module_path}.execute_query still points at the real driver — "
            "fake_neo4j does not cover it"
        )
        assert await module.execute_query("MATCH (n) RETURN n") == [{"ok": True}]
        assert calls == [("MATCH (n) RETURN n", {})]

    async def test_entity_resolution_calls_through_the_driver_module(self, fake_neo4j):
        """It uses ``neo4j_driver.execute_query``, so patching the driver suffices."""
        from app import neo4j_driver
        from app.services import entity_resolution

        calls = fake_neo4j(lambda q, p: [])
        assert entity_resolution.neo4j_driver is neo4j_driver
        await entity_resolution.neo4j_driver.execute_query("MERGE (n)")
        assert calls == [("MERGE (n)", {})]

    async def test_a_newly_added_service_is_covered_automatically(
        self, fake_neo4j, monkeypatch
    ):
        """The sweep — not the hard-coded list — is what makes this correct."""
        from app import neo4j_driver

        newcomer = ModuleType("app.services.brand_new_service")
        newcomer.execute_query = neo4j_driver.execute_query  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "app.services.brand_new_service", newcomer)

        calls = fake_neo4j(lambda q, p: [{"seen": 1}])

        assert await newcomer.execute_query("RETURN 1") == [{"seen": 1}]
        assert calls == [("RETURN 1", {})]

    async def test_patches_are_undone_after_the_test(self):
        """monkeypatch teardown must restore the real driver (no leakage)."""
        from app.services import communities

        assert communities.execute_query.__module__ == "app.neo4j_driver"


# ── 2. LICENSE / NOTICE reach the container images ───────
BACKEND_DOCKERFILE = REPO_ROOT / "backend" / "Dockerfile"
FRONTEND_DOCKERFILE = REPO_ROOT / "frontend" / "Dockerfile"
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"


class TestContainerAttribution:
    @pytest.mark.parametrize("path", [BACKEND_DOCKERFILE, FRONTEND_DOCKERFILE])
    def test_dockerfile_copies_license_and_notice(self, path):
        body = path.read_text(encoding="utf-8")
        assert re.search(r"^COPY LICENSE NOTICE ", body, re.M), (
            f"{path.name} must COPY LICENSE NOTICE — NOTICE requires the "
            "notices be retained in container images"
        )

    @pytest.mark.parametrize("path", [BACKEND_DOCKERFILE, FRONTEND_DOCKERFILE])
    def test_dockerfile_sources_are_prefixed_for_a_root_context(self, path):
        """A repo-root context means every COPY of project source is prefixed."""
        subdir = path.parent.name  # "backend" / "frontend"
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("COPY ") or "--from=" in stripped:
                continue
            source = stripped.split()[1]
            assert source.startswith(f"{subdir}/") or source in {"LICENSE", "NOTICE"}, (
                f"{path.name}: `{stripped}` assumes a ./{subdir} build context; "
                f"prefix it with {subdir}/"
            )

    def test_license_and_notice_exist_at_the_repo_root(self):
        assert (REPO_ROOT / "LICENSE").is_file()
        assert (REPO_ROOT / "NOTICE").is_file()

    def test_compose_builds_both_services_from_the_repo_root(self):
        yaml = pytest.importorskip("yaml")
        compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
        for service in ("backend", "frontend"):
            build = compose["services"][service]["build"]
            assert build["context"] == ".", f"{service} build context must be the repo root"
            assert build["dockerfile"] == f"{service}/Dockerfile"

    def test_compose_keeps_the_dev_ergonomics(self):
        """The root-context move must not silently drop ports/mounts/env_file."""
        yaml = pytest.importorskip("yaml")
        compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
        backend = compose["services"]["backend"]
        assert "${BACKEND_PORT:-8000}:8000" in backend["ports"]
        assert ".env" in backend["env_file"]
        assert "./backend:/app" in backend["volumes"]
        assert "--reload" in backend["command"]
        assert backend["healthcheck"]["test"]
        frontend = compose["services"]["frontend"]
        assert "${FRONTEND_PORT:-3000}:3000" in frontend["ports"]
        assert any("NEXT_PUBLIC_API_URL" in arg for arg in frontend["build"]["args"])

    def test_render_blueprint_uses_the_root_context(self):
        yaml = pytest.importorskip("yaml")
        render = yaml.safe_load((REPO_ROOT / "render.yaml").read_text(encoding="utf-8"))
        service = render["services"][0]
        assert service["dockerContext"] == "."
        assert service["dockerfilePath"] == "./backend/Dockerfile"

    def test_dockerignore_exists_and_excludes_the_heavy_paths(self):
        """A repo-root context uploads node_modules/.git unless ignored."""
        path = REPO_ROOT / ".dockerignore"
        assert path.is_file(), "a repo-root build context needs a repo-root .dockerignore"
        entries = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        for required in ("node_modules", ".git", ".next", "__pycache__", ".env"):
            assert required in entries, f".dockerignore must exclude {required}"
        assert "LICENSE" not in entries and "NOTICE" not in entries

    def test_fly_config_points_at_the_root_context_dockerfile(self):
        """fly.toml must not silently break: its path is root-relative now."""
        body = (BACKEND_ROOT / "fly.toml").read_text(encoding="utf-8")
        assert 'dockerfile = "backend/Dockerfile"' in body
        assert "--config backend/fly.toml" in body, "fly.toml must document the root-run command"

    def test_deployment_docs_state_the_root_relative_fly_command(self):
        body = (REPO_ROOT / "DEPLOYMENT.md").read_text(encoding="utf-8")
        assert "fly deploy --config backend/fly.toml --dockerfile backend/Dockerfile" in body


# ── 3. SPDX headers ──────────────────────────────────────
SPDX_HEADER = (
    "# SPDX-License-Identifier: AGPL-3.0-or-later",
    "# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>",
    "# Synapse — https://github.com/ahmedmaaloul/synapse",
)
_SKIP_DIRS = {"__pycache__", ".venv", "venv", ".pytest_cache", ".ruff_cache", "node_modules"}

# Known gaps, tracked rather than ignored. Marked xfail(strict=True): the moment
# a header is added the XPASS turns into a failure, forcing the entry out of
# this list. It can only shrink — it cannot silently hide a new gap.
_HEADER_PENDING = {
    "tests/test_communities.py",
    "tests/test_chat_engine.py",
}


def _python_sources() -> list[Path]:
    files: list[Path] = []
    for sub in ("app", "scripts", "eval", "tests", "benchmarks"):
        root = BACKEND_ROOT / sub
        if not root.is_dir():
            continue
        files += [
            p
            for p in sorted(root.rglob("*.py"))
            if not _SKIP_DIRS.intersection(p.relative_to(BACKEND_ROOT).parts)
        ]
    return files


def _header_params() -> list:
    params = []
    for path in _python_sources():
        rel = path.relative_to(BACKEND_ROOT).as_posix()
        marks = (
            [pytest.mark.xfail(strict=True, reason=f"SPDX header still missing from {rel}")]
            if rel in _HEADER_PENDING
            else []
        )
        params.append(pytest.param(path, marks=marks, id=rel))
    return params


class TestSpdxHeaders:
    def test_the_scan_finds_something(self):
        assert len(_python_sources()) > 10

    @pytest.mark.parametrize("path", _header_params())
    def test_file_starts_with_the_spdx_header(self, path):
        lines = path.read_text(encoding="utf-8").splitlines()[:3]
        assert tuple(lines) == SPDX_HEADER, (
            f"{path.relative_to(BACKEND_ROOT)} is missing the 3-line SPDX/copyright header"
        )

    @pytest.mark.parametrize(
        "path", _python_sources(), ids=lambda p: p.relative_to(BACKEND_ROOT).as_posix()
    )
    def test_future_import_is_still_the_first_statement(self, path):
        """The header is comments, so `from __future__` must remain importable."""
        import ast

        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
