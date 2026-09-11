# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Refuse to release unless every version source agrees with the tag.

The version is declared in four independent places — the backend package, the
``APP_VERSION`` constant served by ``/api/about`` (the authorship-and-licence
endpoint), the ``synapse-graphrag`` package that goes to PyPI, and the frontend
``package.json``. Nothing ties them together at build time, so a tag pushed
after a partial bump would ship a release whose artifacts disagree about what
version they are. The release workflow runs this first and stops before
anything is published.

Usage::

    python scripts/check_versions.py v0.4.0       # tag form
    python scripts/check_versions.py 0.4.0        # bare version

Exit status is 0 when all sources match, 1 on any mismatch or unreadable source.
Standard library only (``tomllib`` needs Python ≥ 3.11) so it runs on a bare
CI runner without installing anything.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Semver as used for git tags and package.json (npm rejects anything else).
# Pre-releases use a dash (0.5.0-rc.1); pip normalises that to 0.5.0rc1 itself.
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
_APP_VERSION = re.compile(r"^APP_VERSION\s*=\s*[\"']([^\"']+)[\"']", re.M)


def _pyproject_version(path: Path) -> str:
    with path.open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def _app_version(path: Path) -> str:
    match = _APP_VERSION.search(path.read_text(encoding="utf-8"))
    if not match:
        raise KeyError('no `APP_VERSION = "..."` assignment found')
    return match.group(1)


def _package_json_version(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["version"]


# (relative path, human label, reader) — the order is the order printed.
SOURCES: tuple[tuple[str, str, Callable[[Path], str]], ...] = (
    ("backend/pyproject.toml", "[project].version", _pyproject_version),
    ("backend/app/main.py", "APP_VERSION", _app_version),
    ("packages/synapse-graphrag/pyproject.toml", "[project].version", _pyproject_version),
    ("frontend/package.json", "version", _package_json_version),
)


def normalise(tag_or_version: str) -> str:
    """``v0.4.0`` / ``refs/tags/v0.4.0`` / ``0.4.0`` → ``0.4.0``."""
    value = tag_or_version.strip().removeprefix("refs/tags/")
    return value[1:] if value.startswith("v") else value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tag", help="release tag (v0.4.0) or bare version (0.4.0)")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root to inspect (default: the checkout this script lives in)",
    )
    args = parser.parse_args(argv)

    expected = normalise(args.tag)
    if not _SEMVER.match(expected):
        print(f"error: {args.tag!r} is not a semver tag (expected vMAJOR.MINOR.PATCH)")
        return 1

    width = max(len(f"{rel} ({label})") for rel, label, _ in SOURCES)
    failures = 0
    print(f"expected version: {expected}")
    for rel, label, read in SOURCES:
        name = f"{rel} ({label})".ljust(width)
        path = args.root / rel
        try:
            found = read(path)
        except FileNotFoundError:
            print(f"  {name}  MISSING   file not found")
            failures += 1
            continue
        except (KeyError, ValueError, TypeError) as exc:
            print(f"  {name}  UNREADABLE  {exc}")
            failures += 1
            continue
        if found == expected:
            print(f"  {name}  {found}  ok")
        else:
            print(f"  {name}  {found}  MISMATCH (expected {expected})")
            failures += 1

    if failures:
        print(f"\n{failures} of {len(SOURCES)} version sources disagree with {expected}.")
        return 1
    print(f"\nAll {len(SOURCES)} version sources agree: {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
