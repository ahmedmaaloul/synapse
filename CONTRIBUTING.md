# Contributing to Synapse

Thanks for being here. Synapse is built to be **forked and extended** — issues and pull requests
are genuinely welcome, and the codebase is organised so that a first contribution is small and
easy to land.

Maintainer: **Ahmed Maaloul** <maaloulahmed25@gmail.com> ·
[github.com/ahmedmaaloul/synapse](https://github.com/ahmedmaaloul/synapse)

**Contents**

- [Good first contributions](#good-first-contributions)
- [Dev setup](#dev-setup)
- [⭐ Add a new AI provider in ~20 lines](#add-a-new-ai-provider-in-20-lines)
- [Adding an embedding provider](#adding-an-embedding-provider)
- [Before opening a PR](#before-opening-a-pr)
- [Conventions](#conventions)
- [The dependency constraint you must respect](#the-dependency-constraint-you-must-respect)
- [Licensing of contributions](#licensing-of-contributions)

---

## Good first contributions

| Idea | Why it's a good start |
| --- | --- |
| **[Add an AI provider](#add-a-new-ai-provider-in-20-lines)** | Self-contained, ~20 lines + a test, immediately useful to someone else. [Open a provider request →](https://github.com/ahmedmaaloul/synapse/issues/new?template=provider_request.yml) |
| Add an eval query to `backend/eval/dataset.json` | Sharpens the retrieval metrics with no new code paths |
| A new ingest format (`.md`, `.txt`, URLs) | Clean seam next to `services/pdf_parser.py` |
| Frontend polish in `frontend/src/app/components/` | Presentational, isolated, visible |
| Docs fixes | Always welcome — if something confused you, it will confuse the next person |

Anything labelled [`good first issue`](https://github.com/ahmedmaaloul/synapse/labels/good%20first%20issue)
is fair game. If you're unsure whether an idea fits, open an issue first — that's cheaper than a
rejected PR for both of us.

---

## Dev setup

```bash
git clone https://github.com/ahmedmaaloul/synapse.git
cd synapse
cp .env.example .env

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Frontend
cd ../frontend && npm install
```

Run it with `make up` (full Docker stack) or `make backend-dev` / `make frontend-dev` for local
hot-reload against `docker compose up -d neo4j`.

Get real data in the graph without any API key:

```bash
make demo          # via Docker (needs `make up`)
make demo-local    # via your local venv (needs `docker compose up -d neo4j`)
```

Optional cloud provider SDKs (Vertex AI, Bedrock, Groq, Mistral, Cohere):

```bash
make providers     # pip install -r backend/requirements-providers.txt
```

Zero-install alternative: **[open the repo in GitHub Codespaces](https://codespaces.new/ahmedmaaloul/synapse)** —
the [devcontainer](./.devcontainer/devcontainer.json) provisions Python 3.12, Node 20, every backend
dependency and a ready `.env`.

---

## Add a new AI provider in ~20 lines

Everything AI-related lives behind one factory:
[`backend/app/services/llm_provider.py`](./backend/app/services/llm_provider.py). Adding a provider
touches **five files** and nothing else. We'll use a fictional *Together AI* as the running example.

> **First, check you actually need to.** `LLM_PROVIDER=openai_compatible` is `ChatOpenAI` pointed at
> a custom `base_url` and already covers OpenRouter, Together, DeepSeek, Fireworks, vLLM, LM Studio
> and llama.cpp. A dedicated branch is worth it when the provider has a first-party LangChain
> package with capabilities the OpenAI shim can't express.

### 1. Declare the provider in `backend/app/config.py`

Add the value to the `LLMProvider` literal — this makes a typo in `.env` a startup-time validation
error rather than a mystery at request time:

```python
LLMProvider = Literal[
    "gemini",
    "claude",
    # ...
    "openai_compatible",
    "together",          # ← new
]
```

Then add its settings to the `Settings` class, next to the other providers. Every field needs a
default so the app still boots with an empty `.env`:

```python
    # Together AI (cloud). https://api.together.xyz/settings/api-keys
    together_api_key: str = ""
    together_chat_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
```

Pydantic maps these to `TOGETHER_API_KEY` and `TOGETHER_CHAT_MODEL` automatically.

### 2. Register it in `llm_provider.py`

Add the string to the `CHAT_PROVIDERS` tuple at the top of the module. It's what builds the
"Unknown LLM_PROVIDER" error message, and a test asserts it stays in sync with `LLMProvider`:

```python
CHAT_PROVIDERS = (
    "gemini",
    # ...
    "openai_compatible",
    "together",          # ← new
)
```

### 3. Add the branch to `get_chat_llm`

This is the whole implementation. Copy the shape of the `groq` branch and follow three rules:

1. **Validate credentials first**, before importing anything — the error must be actionable and
   name the env var and where to get the key.
2. **Import the SDK lazily, inside the branch** — the app (and the test suite) must boot even when
   the package isn't installed.
3. **Use `_missing_package()`** for the ImportError, with the version pin Synapse is tested against.
   Pass `optional=True` if the SDK lives in `requirements-providers.txt`.

```python
    if provider == "together":
        if not settings.together_api_key:
            raise ProviderConfigError(
                "LLM_PROVIDER=together but TOGETHER_API_KEY is empty. "
                "Get a key at https://api.together.xyz/settings/api-keys"
            )
        try:
            from langchain_together import ChatTogether
        except ImportError as e:  # pragma: no cover - env-dependent
            raise _missing_package(
                "langchain-together", pin="==0.3.0", optional=True
            ) from e
        kwargs: dict = {
            "model": settings.together_chat_model,
            "api_key": settings.together_api_key,
            "temperature": temp,
            "streaming": streaming,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatTogether(**kwargs)
```

Place it before the final `raise ProviderConfigError(f"Unknown LLM_PROVIDER=...")`. That's it —
`graph_builder` (extraction) and `chat_engine` (generation) both call `get_chat_llm()` and now work
with your provider, streaming and JSON mode included.

### 4. Write a hermetic test

In [`backend/tests/test_llm_provider.py`](./backend/tests/test_llm_provider.py). Because the branch
validates credentials *before* importing the SDK, you can assert the whole contract with **no
network, no key and no package installed** — copy `test_groq_without_key`:

```python
    def test_together_without_key(self):
        s = Settings(llm_provider="together", together_api_key="")
        with pytest.raises(ProviderConfigError) as exc:
            get_chat_llm(settings=s)
        msg = str(exc.value)
        assert "TOGETHER_API_KEY" in msg
        assert "https://api.together.xyz/settings/api-keys" in msg
```

The existing suite already checks that `CHAT_PROVIDERS` and the `LLMProvider` literal agree, so
forgetting step 2 fails CI on its own.

### 5. Document it

- **`.env.example`** — a new commented block: which vars are required, which are optional, and the
  link to get a key. Match the existing grouping.
- **`README.md`** — one row in the [provider matrix](./README.md#-provider-matrix): provider, env
  value, credential, free tier, install.
- **`backend/requirements-providers.txt`** — only if the SDK is heavy or optional. Pin it exactly,
  and add a comment saying which version of `langchain-core` it resolves against. See
  [the dependency constraint](#the-dependency-constraint-you-must-respect).

### Checklist

```
[ ] config.py      — LLMProvider literal + Settings fields (all defaulted)
[ ] llm_provider.py — CHAT_PROVIDERS tuple + branch in get_chat_llm
[ ] tests           — hermetic missing-credential test
[ ] .env.example    — commented block
[ ] README.md       — provider matrix row
[ ] requirements-providers.txt — exact pin, if the SDK is optional
[ ] make test && make lint both clean
```

---

## Adding an embedding provider

Same shape, one extra step, because embedding construction is cached.

1. `config.py` — add the value to the `EmbeddingProvider` literal and add the model/credential
   fields to `Settings`.
2. `llm_provider.py` — add the string to `EMBEDDING_PROVIDERS`.
3. `get_embeddings()` — pass your new settings through as **keyword arguments** to
   `_load_embeddings(...)`.
4. `_load_embeddings()` — add matching keyword parameters (with defaults) and your branch.
   ⚠️ This function is `@lru_cache`d, so **only hashable primitives** may be parameters — that's
   exactly why the settings object is exploded into scalars rather than passed whole.
5. Document the **embedding dimension** everywhere: `.env.example`, the README embeddings table, and
   the error message. `EMBEDDING_DIM` sizes the Neo4j vector index, and a mismatch breaks retrieval
   silently — this is the single most common way to get an embedding provider wrong.

Look at the `cohere` branch: it is the smallest complete example, credential check and all.

---

## Before opening a PR

Everything CI enforces, run locally:

```bash
make lint     # ruff (backend) + eslint + tsc (frontend)
make test     # backend hermetic unit tests
```

If you touched retrieval or ingestion, also run the integration tests and the eval against a local
Neo4j — retrieval metrics must not regress:

```bash
docker compose up -d neo4j
make test-int
make eval
```

Then open the PR using the [template](./.github/PULL_REQUEST_TEMPLATE.md) — it mirrors these gates.

---

## Conventions

- **Backend:** typed, `ruff`-clean, small focused functions. New behaviour needs a test (hermetic
  unit test preferred; integration test only if it genuinely needs Neo4j).
- **Frontend:** no `any` at call sites, components stay presentational where possible, shared logic
  lives in `src/app/lib/`.
- **Commits & PR titles:** conventional style — `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`,
  `test:`.
- **New source files** carry the project header, matching the existing modules:

  ```python
  # SPDX-License-Identifier: AGPL-3.0-or-later
  # Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
  # Synapse — https://github.com/ahmedmaaloul/synapse
  ```

- **Error messages are documentation.** Every provider error in this codebase names the env var, says
  what's wrong, and links to where the credential comes from. Keep that bar.

---

## The dependency constraint you must respect

`backend/requirements.txt` pins **`langchain-core>=0.3.33,<0.4`**. That cap is **load-bearing**:
without it, pip upgrades `langchain-core` to 1.x to satisfy a provider package, which breaks
`langchain`, `langchain-community` and `langchain-ollama` (all of which require core `<0.4`).

So: **any dependency you add must resolve against `langchain-core <0.4`**, and you must verify it
with a real `pip install` — not by reading the changelog. Practically this means picking the
*newest* release of an SDK that still allows core 0.3.x, and pinning it exactly (open ranges make
pip explore the whole langchain 0.3 matrix and abort with `resolution-too-deep`).
`backend/requirements-providers.txt` documents the exact reasoning per package — read it before
adding a pin, and add the same kind of comment for yours.

---

## Licensing of contributions

Synapse is licensed under **AGPL-3.0-or-later** and is additionally offered under a separate
commercial license by the author (see [`COMMERCIAL-LICENSE.md`](./COMMERCIAL-LICENSE.md)).

By submitting a pull request you agree that:

1. Your contribution is licensed under **AGPL-3.0-or-later**, and
2. **Ahmed Maaloul may also license it under the commercial terms** described in
   `COMMERCIAL-LICENSE.md` — this is what makes the dual-licensing model possible, and
3. You have the right to contribute the code (it's yours, or your employer permits it).

**You keep the copyright in your own contribution.** You may add your own copyright line for
substantial work; you may not remove or replace the existing copyright notices — see
[`NOTICE`](./NOTICE). If a formal CLA is ever introduced, it will be added to the repository and you
will be asked to sign it explicitly.

Forks are welcome and encouraged. A fork must stay under AGPL-3.0-or-later and keep `LICENSE` and
`NOTICE` intact; mark your changes as yours and it's yours to take wherever you want.

---

By participating you agree to the [Code of Conduct](./CODE_OF_CONDUCT.md). Found a security issue?
Please follow [SECURITY.md](./SECURITY.md) instead of opening a public issue.
