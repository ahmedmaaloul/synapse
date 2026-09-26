# Development, tests and evaluation

How to work on Synapse, what the test suites cover, which commands delete data, what CI and
the release workflow do, and which measurements exist.

## Contents

1. [Local development](#1-local-development)
2. [Test suites](#2-test-suites)
3. [Destructive operations](#3-destructive-operations)
4. [Continuous integration and releases](#4-continuous-integration-and-releases)
5. [Evaluation status](#5-evaluation-status)

## 1. Local development

Run the backend and frontend on the host against the Neo4j container of Docker Compose. Set
`NEO4J_URI=bolt://localhost:7687` in `.env` first. The backend started from `backend/` reads the
root `.env`; the backend container started by Docker Compose overrides `NEO4J_URI` with its own
value.

```bash
docker compose up -d neo4j
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cd ../frontend && npm install && cd ..
pip install -e "packages/synapse-graphrag[dev]"   # client package, needed by make mcp-test and make release-check
make backend-dev       # first shell: uvicorn with reload on port 8000
make frontend-dev      # second shell: Next.js dev server
```

`make` without a target prints all 21 targets. `make providers` installs the optional provider
SDKs into the current Python environment.

A [GitHub Codespaces devcontainer](../.devcontainer/devcontainer.json) is available. It requests
4 CPUs, 8 GB of RAM and 32 GB of storage, provides Python 3.12, Node 20, Docker-in-Docker and the
GitHub CLI, forwards ports 3000, 8000, 8765, 7474 and 7687, installs the backend runtime and
development dependencies, the client package and the frontend packages, and copies `.env.example`
to `.env`. It does not install the optional provider SDKs, and it does not start any service:
run `docker compose up -d neo4j`, `make backend-dev` and `make frontend-dev` yourself. Its
environment sets `EMBEDDING_PROVIDER=fastembed`, which the test configuration does not override,
so run the unit tests there with `EMBEDDING_PROVIDER=fake make test`.

## 2. Test suites

Sizes measured at commit `def7fab`.

| Suite | Command | Size | External dependencies |
| --- | --- | --- | --- |
| Backend unit | `make test` | 2,149 tests collected, 2,041 of them outside `tests/integration` | none: fake embedder, no database, no network, no LLM; tests that need an opt-in variable (`SYNAPSE_IT`, `SYNAPSE_HOTPOT`, `SYNAPSE_2WIKI`) are skipped |
| Backend integration | `make test-int` | 108 tests | Neo4j at `localhost:7687` (destructive, see [3](#3-destructive-operations)), the fastembed model |
| Client package | `make mcp-test` | 218 tests, plus ruff | none |
| Lint | `make lint` | ruff (backend), ESLint and `tsc --noEmit` (frontend); there are no frontend tests | none |

In the CI run on `def7fab` (2026-09-25) the unit job reported 2,036 passed, 111 skipped and
2 expected failures, and the integration job 108 passed. The expected failures are strict `xfail`
entries that track two test files still missing an SPDX licence header.

Hermetic backend tests of five subsystems (not every hermetic test belongs to one of them, so the
rows do not add up to 2,041):

| Subsystem | Tests |
| --- | --- |
| Procedural memory, Navigator agent and procedural benchmark harness | 530 |
| Synapse Lab (backend) | 440 |
| Ingestion, entity resolution, themes, source chunks, job bus, demo seed | 282 |
| Retrieval engine and HTTP API | 136 |
| Provider factory and reasoning-model handling | 114, plus 1 skipped (it downloads a model) |

The test configuration sets `EMBEDDING_PROVIDER=fake`, `EMBEDDING_DIM=384` and
`LLM_PROVIDER=ollama` only where the shell has not set them; unset provider variables before
running the suite. Under pytest the Lab reader refuses to build a real OpenAI client.

## 3. Destructive operations

| Operation | Deletes | Confirmation |
| --- | --- | --- |
| `make demo`, `make demo-local` (`seed_demo --clear`) | All `:Entity`, `:Community` and `:Chunk` nodes | none |
| `make benchmark` | All `:Entity`, `:Community` and `:Chunk` nodes at `localhost:7687` | none |
| `make eval` | Every node at `localhost:7687`, procedural graphs included; overwrites `backend/eval/results.md` | none |
| `make test-int` | Every node at `localhost:7687`, before and after each test | none |
| UI button **Clear Database**, `DELETE /api/graph` | Every node except the procedural graphs | none |
| MCP `synapse_clear_graph` | As `DELETE /api/graph` | `confirm=true` required |
| `docker compose down -v` | The Neo4j data and log volumes | none |

Run `eval`, `benchmark` and `test-int` only against a Neo4j instance you can lose. The bundled
procedural graphs are seeded again at the next backend start; evolved versions are lost.

## 4. Continuous integration and releases

[ci.yml](../.github/workflows/ci.yml) runs on every push to `main` and every pull request into
`main`, with Python 3.12 and Node 20:

| Job | Steps |
| --- | --- |
| Backend · lint & unit tests | ruff, pytest |
| Backend · integration (live Neo4j) | Neo4j 5 Community service container with APOC; integration tests; the retrieval smoke test |
| Frontend · lint, typecheck & build | `npm ci`, ESLint, `tsc --noEmit`, `next build` |
| MCP package · lint, tests & build | ruff, pytest, sdist and wheel, `twine check`, install of the wheel in a clean venv and a run of both commands |
| Docker · build images | Builds the backend, frontend and MCP images; checks that `LICENSE` and `NOTICE` are inside the backend and MCP images |

Branch protection is defined as code in [scripts/ruleset-main.json](../scripts/ruleset-main.json)
and applied with `make protect-main`. It requires the five jobs above, blocks force-push and
deletion, and requires pull requests; the admin role can bypass it.

A pushed `v*` tag runs [release.yml](../.github/workflows/release.yml): version and changelog
checks, the client package build, images on GHCR (`synapse-backend`, `synapse-frontend`,
`synapse-mcp`), and a GitHub Release with the changelog section as notes and the client wheel and
sdist attached. Its PyPI step runs only when the repository variable `PYPI_PUBLISH` is `true`,
which is not set. The release workflow runs none of the test suites; it only checks that the
built wheel installs and that both commands start. The release procedure (bump the four
version sources, move the `[Unreleased]` changelog section, run `make release-check`, `make test`,
`make lint` and `make mcp-test`, push an annotated tag) is described in
[CONTRIBUTING.md](../CONTRIBUTING.md#cutting-a-release).

## 5. Evaluation status

| Evidence | Scope | Result | Remarks |
| --- | --- | --- | --- |
| Retrieval smoke test, [backend/eval](../backend/eval/) (`make eval`) | 8 hand-written questions over a 15-entity, 10-relationship fixture, k = 8 | Hit@1 87.5% (7 of 8), Recall@8 100%, Precision@8 17.19%, MRR 0.917 | k = 8 returns 8 of the 15 entities, so Recall@8 is close to saturated. No keyword-only baseline is run. CI prints the scores and enforces no threshold. |
| Retrieval benchmark, [backend/benchmarks](../backend/benchmarks/results.md) (`make benchmark`) | 14 multi-hop questions over 34 passages; retrieval only, no LLM | Shipped path: Fact Recall 94.6%, Full-Coverage 85.7% (permissive rule) on 3,516 characters. Passage baseline at a matched budget (top 20, 4,072 characters): 100% on both. | The results file concludes that a plain passage baseline matches the shipped path on every metric while reading less text: it reaches the shipped path's Fact Recall at top 8 with 48% of its context, and its Full-Coverage at top 6 with 36%. Last regenerated 2026-07-19 (commit `33402ec`); the retrieval logic has not changed since. |
| Procedural Graphs | Navigator with and without procedural guidance | No Synapse results are published. The paper's results have not been reproduced. | A cost-capped harness exists in [backend/benchmarks/procedural](../backend/benchmarks/procedural/README.md). |
| Synapse Lab | your own questions | No results are committed (`backend/lab_runs/` is ignored by git). | The Lab exists to run this measurement on your documents. |

Synapse does not claim that graph retrieval answers better than passage retrieval. The repository
contains no result that shows graph retrieval ahead of passage retrieval at a matched context
budget. Use the Lab to measure the trade-off on your own data.

