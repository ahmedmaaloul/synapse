#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
#
# Apply (or re-apply) the `protect-main` repository ruleset from
# scripts/ruleset-main.json. Idempotent: an existing ruleset with that name is
# updated in place (PUT), otherwise it is created (POST).
#
# What the ruleset enforces on the default branch:
#   - no deletion, no force-push (non_fast_forward)
#   - every CI job in .github/workflows/ci.yml must be green before merging —
#     the `context` strings are the jobs' `name:` values, so renaming a job in
#     ci.yml means updating ruleset-main.json and re-running this script
#   - changes land through pull requests (0 approvals required: solo maintainer)
#
# bypass_actors — `RepositoryRole` 5 is "admin" (4 = maintain, 2 = write). It
# lets the maintainer keep pushing straight to main, e.g. version bumps and
# release tags, while still blocking accidental deletes/force-pushes for
# everyone else. To make the rules apply to admins too, set
# `"bypass_actors": []` in scripts/ruleset-main.json and re-run this script.
#
# Requirements: `gh` authenticated with admin rights on the repository
# (`gh auth login`; rulesets need the `repo` scope). Run from anywhere:
#
#   bash scripts/protect-main.sh      # or: make protect-main

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
RULESET_FILE="scripts/ruleset-main.json"
RULESET_NAME="protect-main"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: the GitHub CLI (gh) is not installed — https://cli.github.com" >&2
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "error: gh is not authenticated — run \`gh auth login\` first" >&2
  exit 1
fi

REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
echo "repository: $REPO"

# Rulesets are keyed by id, not name, so look the name up first.
EXISTING_ID="$(gh api "repos/$REPO/rulesets" --paginate \
  --jq ".[] | select(.name == \"$RULESET_NAME\") | .id" | head -n 1)"

if [ -n "$EXISTING_ID" ]; then
  echo "updating existing ruleset '$RULESET_NAME' (id $EXISTING_ID)"
  RESULT="$(gh api -X PUT "repos/$REPO/rulesets/$EXISTING_ID" --input "$RULESET_FILE" \
    --jq '"\(.id) \(._links.html.href // "")"')"
else
  echo "creating ruleset '$RULESET_NAME'"
  RESULT="$(gh api -X POST "repos/$REPO/rulesets" --input "$RULESET_FILE" \
    --jq '"\(.id) \(._links.html.href // "")"')"
fi

RULESET_ID="${RESULT%% *}"
RULESET_URL="${RESULT#* }"
[ -n "$RULESET_URL" ] || RULESET_URL="https://github.com/$REPO/rules/$RULESET_ID"

echo "ruleset '$RULESET_NAME' is active (id $RULESET_ID)"
echo "$RULESET_URL"
