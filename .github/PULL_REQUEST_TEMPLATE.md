<!--
  Thanks for contributing to Synapse!
  Maintainer: Ahmed Maaloul <ahmed.maaloul@proton.me>
  Keep the title in conventional-commit style: feat: / fix: / docs: / chore: / refactor: / test:
-->

## What & why

<!-- What does this change, and what problem does it solve? Link the issue: "Closes #123". -->

## Type of change

- [ ] 🐛 Bug fix (non-breaking)
- [ ] ✨ Feature (non-breaking)
- [ ] 🔌 New AI provider integration
- [ ] 💥 Breaking change (API, env vars, or graph schema)
- [ ] 📚 Documentation only
- [ ] 🧹 Refactor / chore / CI

## How was this tested?

<!-- Commands you ran, documents you ingested, questions you asked. Screenshots or a short clip for UI changes. -->

## Quality gates

Everything CI enforces, run locally before requesting review:

- [ ] `make test` — backend hermetic unit tests pass (no new failures, no new skips vs. `main`)
- [ ] `make lint` — clean: `ruff check .` (backend) + `npx eslint src` and `npx tsc --noEmit` (frontend)
- [ ] New or changed behaviour is covered by a test — hermetic unit test preferred, integration test if it genuinely needs Neo4j
- [ ] Touched retrieval or ingestion? Ran `make test-int` and `make eval` against a local Neo4j (`docker compose up -d neo4j`) and retrieval metrics did not regress

## Dependencies

- [ ] No new dependencies **— or —** new dependencies resolve against **`langchain-core >=0.3.33,<0.4`**, verified with a real `pip install` (this cap is load-bearing: provider SDKs otherwise drag `langchain-core` to 1.x and break `langchain` / `langchain-community` / `langchain-ollama`)
- [ ] `backend/requirements.txt` / `frontend/package.json` and the corresponding lockfile are both updated

## Documentation & config

- [ ] New env vars are documented in `.env.example` **and** wired into `backend/app/config.py`
- [ ] README / `ARCHITECTURE.md` / `DEPLOYMENT.md` updated if behaviour, setup, or architecture changed
- [ ] `CHANGELOG.md` has an entry under `## [Unreleased]`
- [ ] No secrets, API keys, or personal data in the diff (including test fixtures and logs)

## Licensing

- [ ] I have read and agree to the [Contributor License Agreement](../CLA.md).

<sub>In short: **you keep the copyright to your work.** You additionally grant Ahmed Maaloul the right to ship your contribution under both **AGPL-3.0-or-later** and the project's [commercial licence](../COMMERCIAL-LICENSE.md) — the grant that makes dual-licensing legally possible. You only need to accept this once.</sub>

## Notes for the reviewer

<!-- Anything intentionally left out, open questions, or areas you'd like scrutinised. -->
