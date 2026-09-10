# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Print one release's section of CHANGELOG.md — the GitHub Release notes.

The changelog follows Keep a Changelog: one ``## [X.Y.Z] — date`` heading per
release, newest first, with reference-style links (``[X.Y.Z]: https://…``) at
the very bottom of the file. The release workflow feeds this script's output to
``gh release create --notes-file`` so the notes are written exactly once, in the
changelog, and a tag whose section was never written fails loudly instead of
producing an empty release page.

Usage::

    python scripts/changelog_section.py 0.4.0 > notes.md

Exit status is 0 when the section exists and has content, 1 otherwise.
Standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/ahmedmaaloul/synapse"

# A section ends at the next release heading or at the trailing link-reference
# block — without the second stop the oldest section would swallow every link.
_NEXT_HEADING = re.compile(r"^## \[")
_LINK_REFERENCE = re.compile(r"^\[[^\]]+\]:\s*\S+")


def extract_section(changelog: str, version: str) -> str | None:
    """Body of ``## [version]`` (heading excluded, whitespace trimmed), or None."""
    heading = re.compile(r"^## \[" + re.escape(version) + r"\](?:\s|$)")
    lines = changelog.splitlines()
    start = next((i for i, line in enumerate(lines) if heading.match(line)), None)
    if start is None:
        return None
    body: list[str] = []
    for line in lines[start + 1 :]:
        if _NEXT_HEADING.match(line) or _LINK_REFERENCE.match(line):
            break
        body.append(line)
    return "\n".join(body).strip()


def absolutize_links(section: str, version: str) -> str:
    """Point ``](./path)`` links at the tagged tree on GitHub.

    In-repo markdown resolves relative links against the file; a Release body
    resolves them against ``/releases/tag/vX.Y.Z`` and 404s. Pinning to the
    tag also keeps the notes accurate after the files move on ``main``.
    """
    return re.sub(r"\]\(\./", f"]({REPO_URL}/blob/v{version}/", section)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", help="release version, e.g. 0.4.0 (a leading v is dropped)")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root holding CHANGELOG.md (default: this checkout)",
    )
    args = parser.parse_args(argv)

    version = args.version.strip().removeprefix("refs/tags/").removeprefix("v")
    changelog_path = args.root / "CHANGELOG.md"
    try:
        changelog = changelog_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"error: {changelog_path} not found", file=sys.stderr)
        return 1

    section = extract_section(changelog, version)
    if section is None:
        print(
            f"error: CHANGELOG.md has no `## [{version}]` section — move the [Unreleased] "
            f"entries under `## [{version}] — YYYY-MM-DD` before tagging",
            file=sys.stderr,
        )
        return 1
    if not section:
        print(f"error: the `## [{version}]` section of CHANGELOG.md is empty", file=sys.stderr)
        return 1

    print(absolutize_links(section, version))
    return 0


if __name__ == "__main__":
    sys.exit(main())
