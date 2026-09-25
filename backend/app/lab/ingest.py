# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse Lab — ingest a corpus with its extraction calls sent through the Batch API.

The expensive half of building a graph is extraction: one LLM call per
paragraph. The Batch API runs those calls at half price within 24 hours, so
the Lab splits the real pipeline in two (see ``graph_builder``):

  1. :func:`plan_ingest`          — free: count the documents, render every
                                    extraction prompt, tokenise it, price it
                                    (point + upper bound, ``estimate.py``).
  2. :func:`submit_ingest_batch`  — one request per document, with the SAME
                                    prompt the realtime pipeline sends
                                    (``graph_builder.render_extraction_request``),
                                    the same ``response_format`` (JSON mode) and
                                    ``temperature = settings.extraction_temperature``
                                    (a reasoning model gets ``reasoning_effort``
                                    instead, exactly as ``llm_provider`` builds
                                    it). gpt-4o-mini by default. Refused up
                                    front when the upper bound breaks ``max_usd``.
  3. :func:`apply_ingest_batch`   — once the batch is done: every reply is
                                    parsed by the SAME parser and written by the
                                    SAME post-extraction code
                                    (``graph_builder.build_knowledge_graph_from_extractions``),
                                    one document per paragraph and one paragraph
                                    at a time, in corpus order — exactly what
                                    ``benchmarks.public.run_hotpotqa.ingest_corpus``
                                    does with the realtime pipeline, so entity
                                    resolution across documents sees the same
                                    sequence. Community detection (Louvain, free)
                                    then runs once at the end.

COMMUNITY SUMMARIES are OFF by default (``community_summaries=False``): they
are one realtime LLM call per community on the configured chat provider (not
batched, not half price), and no Lab arm reads them except ``synapse_d``'s
global route — which only fires on broad, corpus-level questions and otherwise
falls back to local search. With summaries off, communities are still detected
and written, with the product's own no-LLM fallback title and summary (what
``communities.detect_and_summarize`` writes when no LLM is configured).

RESUMABLE. Everything lives under ``<run_dir>/ingest/`` (so an ingest can
share a directory with a Lab run without touching its ``batches.json``):
``ingest.json`` (config, estimate, status, actual spend), ``documents.jsonl``
(the paragraph texts), the batch state and files, and ``applied.jsonl`` — one
line per document already written, so an interrupted apply resumes where it
stopped. A failed extraction is never silently dropped: apply refuses until it
is resubmitted (:func:`resubmit_failed_ingest`) or explicitly accepted
(``allow_failed=True`` — the document is then stored with no entities, which
is what the realtime pipeline does when an extraction call fails).

A graph built this way satisfies the Lab's HotpotQA check (every paragraph is a
``:Chunk`` document named by its title); :func:`apply_ingest_batch` verifies it
read-only before reporting success.

WARNING: apply WRITES to the knowledge graph (``clear_graph=True`` also wipes
its entities, communities and chunks first). Plan, submit and poll never do.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.lab import batch, reader
from app.lab.estimate import (
    COMMUNITY_SUMMARY_COMPLETION_TOKENS,
    COMMUNITY_SUMMARY_PROMPT_TOKENS,
    INGEST_MODEL,
    IngestPlan,
    PhaseEstimate,
    calibrated_ingest,
    estimate_ingest,
    load_calibration,
)
from app.lab.tokens import tokenizer_label
from app.services import graph_builder
from app.services.llm_provider import is_reasoning_model, reasoning_effort_for
from benchmarks.public import cost

logger = logging.getLogger(__name__)

INGEST_SUBDIR = "ingest"
MANIFEST = "ingest.json"
DOCUMENTS = "documents.jsonl"
APPLIED = "applied.jsonl"
DEFAULT_MODEL = INGEST_MODEL
#: The theme ``run_hotpotqa`` ingests HotpotQA with (Wikipedia prose → generic schema).
DEFAULT_THEME = graph_builder.DEFAULT_THEME
#: Entities the extraction prompt asks for per chunk ("at least 5-8"): the top of
#: that range, used ONLY to size the community-summary estimate when summaries
#: are switched on. An assumption, and labelled as one.
ASSUMED_ENTITIES_PER_DOCUMENT = 8

ProgressCallback = Callable[[dict], Awaitable[None] | None]


class IngestError(RuntimeError):
    """An ingest that cannot proceed (foreign run dir, missing batch, bad corpus …)."""


class IngestRefused(IngestError):
    """The upper-bound cost of the extraction batch exceeds ``max_usd``."""

    def __init__(self, message: str, plan: IngestPlanReport) -> None:
        super().__init__(message)
        self.plan = plan


class IngestIncomplete(IngestError):
    """Some extractions failed; apply refuses until they are resubmitted or accepted."""

    def __init__(self, message: str, failed: list[str]) -> None:
        super().__init__(message)
        self.failed = failed


# ── Corpus ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Document:
    """One document to ingest: its name (the ``:Chunk`` document) and its text."""

    name: str
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(f"{self.name}\x1f{self.text}".encode()).hexdigest()


def corpus_documents(corpus: Any) -> list[Document]:
    """The documents of ``corpus``, in ingest order.

    Accepts a ``run_hotpotqa.Corpus`` (anything with ``titles`` and
    ``text(title)`` — ingested in ``titles`` order, as ``ingest_corpus`` does),
    a mapping ``name → text`` (insertion order), or an iterable of
    :class:`Document` / ``(name, text)`` pairs.
    """
    if hasattr(corpus, "titles") and callable(getattr(corpus, "text", None)):
        docs = [Document(str(t), str(corpus.text(t))) for t in corpus.titles]
    elif isinstance(corpus, Mapping):
        docs = [Document(str(k), str(v)) for k, v in corpus.items()]
    elif isinstance(corpus, Iterable) and not isinstance(corpus, str | bytes):
        docs = []
        for item in corpus:
            if isinstance(item, Document):
                docs.append(item)
            elif isinstance(item, tuple | list) and len(item) == 2:
                docs.append(Document(str(item[0]), str(item[1])))
            else:
                raise IngestError(f"cannot read a document from {type(item).__name__}")
    else:
        raise IngestError(f"cannot read a corpus from {type(corpus).__name__}")
    seen: set[str] = set()
    for doc in docs:
        if not doc.name.strip() or not doc.text.strip():
            raise IngestError("every document needs a non-empty name and text")
        if doc.name in seen:
            raise IngestError(f"duplicate document name {doc.name!r} (one document per name)")
        seen.add(doc.name)
    if not docs:
        raise IngestError("the corpus is empty")
    return docs


def load_hotpotqa_corpus(spec: Any = None, **overrides: Any) -> Any:
    """The HotpotQA corpus of a Lab dataset spec — the SAME sample the runner scores.

    ``spec`` is a ``runner.DatasetSpec`` (or its dict); keyword overrides
    (``n``, ``seed``, ``offset``, ``path``, ``allow_download``) replace fields.
    Sampling mirrors ``runner._load_hotpotqa`` exactly (``offset + n`` sampled
    with the seed, then sliced), so the titles here are the titles the
    runner's "graph is ingested" check demands. Free: no model, no Neo4j.
    """
    from app.lab.runner import DatasetSpec
    from benchmarks.public import hotpotqa, run_hotpotqa

    if spec is None:
        spec = DatasetSpec(name="hotpotqa")
    elif isinstance(spec, dict):
        spec = DatasetSpec.from_dict({"name": "hotpotqa", **spec})
    if overrides:
        spec = replace(spec, **overrides)
    seed = hotpotqa.DEFAULT_SEED if spec.seed is None else int(spec.seed)
    n = hotpotqa.DEFAULT_N if spec.n is None else int(spec.n)
    offset = max(0, int(spec.offset or 0))
    try:
        sample = hotpotqa.load_sample(
            offset + n, seed=seed, path=Path(spec.path) if spec.path else None,
            allow_download=spec.allow_download,
        )
    except (OSError, ValueError, RuntimeError) as e:
        raise IngestError(f"could not load HotpotQA: {e}") from e
    questions = sample.questions[offset : offset + n]
    if not questions:
        raise IngestError("the HotpotQA sample is empty (check n / offset)")
    return run_hotpotqa.build_corpus(questions, sample=sample)


# ── Requests ─────────────────────────────────────────────────────────────────
def extraction_request_body(
    text: str,
    document_name: str,
    *,
    model: str = DEFAULT_MODEL,
    theme: str = DEFAULT_THEME,
    settings: Settings | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """The ``chat.completions`` body the realtime pipeline would send for one chunk.

    Same messages (``render_extraction_request``), same JSON mode, and the same
    sampling ``llm_provider`` configures for an OpenAI model:
    ``temperature = settings.extraction_temperature``, or ``reasoning_effort``
    for a reasoning model (which rejects a temperature). The pipeline sets no
    output cap; ``max_output_tokens`` adds one (a deliberate, recorded
    departure that makes the upper bound a hard bound).
    """
    settings = settings or get_settings()
    body: dict[str, Any] = {
        "model": model,
        "messages": graph_builder.render_extraction_request(text, document_name, theme),
    }
    reasoning = is_reasoning_model(model)
    if reasoning:
        body["reasoning_effort"] = reasoning_effort_for(model, settings.openai_reasoning_effort)
    else:
        body["temperature"] = settings.extraction_temperature
    body["response_format"] = {"type": "json_object"}
    if max_output_tokens:
        body["max_completion_tokens" if reasoning else "max_tokens"] = int(max_output_tokens)
    return body


def custom_id(run_id: str, index: int) -> str:
    """``"<run>|ingest|extract|<doc#>"`` — the Lab's four-field custom_id shape."""
    return f"{run_id}|ingest|extract|{index:06d}"


def build_ingest_requests(
    documents: list[Document],
    *,
    run_id: str,
    model: str = DEFAULT_MODEL,
    theme: str = DEFAULT_THEME,
    max_output_tokens: int | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """One Batch input line per document, in corpus order."""
    settings = settings or get_settings()
    return [
        {
            "custom_id": custom_id(run_id, i),
            "method": "POST",
            "url": batch.ENDPOINT,
            "body": extraction_request_body(
                doc.text, doc.name, model=model, theme=theme, settings=settings,
                max_output_tokens=max_output_tokens,
            ),
        }
        for i, doc in enumerate(documents)
    ]


# ── Plan ─────────────────────────────────────────────────────────────────────
@dataclass
class IngestPlanReport:
    """What ingesting a corpus would cost, before anything is sent."""

    documents: int
    chars: int
    model: str
    theme: str
    batch: bool
    #: Real-tokenizer prompt tokens of every rendered request (framing included).
    prompt_tokens: int
    prompt_tokens_estimated: bool
    tokenizer: str
    #: Feed this to ``LabRun.ingest_plan`` / ``estimate_run(ingest=…)``.
    plan: IngestPlan
    extraction: PhaseEstimate
    community_summaries: bool = False
    #: Realtime summary calls on the configured chat model (only when enabled).
    summaries: PhaseEstimate | None = None
    max_output_tokens: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def point_usd(self) -> float | None:
        return _add(self.extraction.point_usd, self.summaries.point_usd if self.summaries else 0.0)

    @property
    def upper_usd(self) -> float | None:
        return _add(self.extraction.upper_usd, self.summaries.upper_usd if self.summaries else 0.0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["point_usd"] = self.point_usd
        data["upper_usd"] = self.upper_usd
        return data


def _add(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a + b


def _chat_model(settings: Settings) -> str:
    from benchmarks.public import run_hotpotqa

    return run_hotpotqa.chat_model_name(settings)


def _summaries_estimate(documents: int, settings: Settings) -> PhaseEstimate:
    """Community-summary calls, sized by an explicit assumption (see the notes)."""
    min_size = max(1, int(settings.community_min_size))
    calls = math.ceil(documents * ASSUMED_ENTITIES_PER_DOCUMENT / min_size)
    model = _chat_model(settings)
    prompt = calls * COMMUNITY_SUMMARY_PROMPT_TOKENS
    completion = calls * COMMUNITY_SUMMARY_COMPLETION_TOKENS
    usd = cost.usd(cost.Usage(prompt_tokens=prompt, completion_tokens=completion),
                   cost.resolve_price(model))
    return PhaseEstimate(
        phase="community_summaries",
        model=model,
        batch=False,
        calls=calls,
        prompt_tokens=prompt,
        completion_tokens=completion,
        upper_prompt_tokens=prompt,
        upper_completion_tokens=completion,
        point_usd=usd,
        upper_usd=usd,
        price_checked_on=cost.price_checked_on(model),
        notes=[
            f"ASSUMED ceiling: {ASSUMED_ENTITIES_PER_DOCUMENT} entities per document / "
            f"community_min_size {min_size} = at most {calls} communities; the real count is "
            "only known after Louvain runs",
            "realtime calls on the configured chat provider (not batched, not half price); "
            "an unpriced provider makes this None",
        ],
    )


def plan_ingest(
    corpus: Any,
    *,
    model: str = DEFAULT_MODEL,
    theme: str = DEFAULT_THEME,
    batch: bool = True,
    community_summaries: bool = False,
    max_output_tokens: int | None = None,
    calibration: Mapping[str, Any] | None = None,
    settings: Settings | None = None,
) -> IngestPlanReport:
    """Count and price the ingest of ``corpus``. Free: no model, no Neo4j, no network.

    Prompt tokens are counted with the real tokenizer on the rendered requests
    (exact up to chat framing); completion tokens per document come from the
    calibration file when it measured this model, else from the measured
    HotpotQA means (``estimate.INGEST_*``). The upper bound adds +10% on the
    prompt and +25% on the completion (extraction has no output cap) — see
    ``estimate.estimate_ingest``.
    """
    settings = settings or get_settings()
    docs = corpus_documents(corpus)
    bodies = [
        extraction_request_body(d.text, d.name, model=model, theme=theme, settings=settings,
                                max_output_tokens=max_output_tokens)
        for d in docs
    ]
    counted = [reader.request_prompt_tokens(b, model) for b in bodies]
    prompt_tokens = sum(n for n, _ in counted)
    estimated = any(est for _, est in counted)
    base = IngestPlan(paragraphs=len(docs), model=model, batch=batch)
    plan = calibrated_ingest(base, load_calibration() if calibration is None else calibration)
    notes = []
    if not estimated:
        completion_source = plan.source
        plan.prompt_tokens_per_paragraph = prompt_tokens / len(docs)
        plan.source = (
            f"prompt tokens counted on the rendered requests ({tokenizer_label(model)}); "
            f"completion tokens from {completion_source}"
        )
    else:
        notes.append("tokenizer unavailable: prompt tokens per document from " + plan.source)
    extraction = estimate_ingest(plan)
    if max_output_tokens:
        cap_total = int(max_output_tokens) * len(docs)
        if cap_total < extraction.upper_completion_tokens:
            extraction.upper_completion_tokens = cap_total
            extraction.upper_usd = cost.usd(
                cost.Usage(prompt_tokens=extraction.upper_prompt_tokens,
                           completion_tokens=cap_total),
                cost.resolve_price(model), batch=batch,
            )
        extraction.notes.append(
            f"output capped at {int(max_output_tokens)} tokens per document "
            "(the realtime pipeline sets no cap)"
        )
    summaries = _summaries_estimate(len(docs), settings) if community_summaries else None
    if not community_summaries:
        notes.append("community summaries OFF: Louvain detection only (free, no LLM)")
    return IngestPlanReport(
        documents=len(docs),
        chars=sum(len(d.text) for d in docs),
        model=model,
        theme=theme,
        batch=batch,
        prompt_tokens=prompt_tokens,
        prompt_tokens_estimated=estimated,
        tokenizer=tokenizer_label(model),
        plan=plan,
        extraction=extraction,
        community_summaries=community_summaries,
        summaries=summaries,
        max_output_tokens=max_output_tokens,
        notes=notes,
    )


# ── Run directory ────────────────────────────────────────────────────────────
def ingest_dir(run_dir: Path | str) -> Path:
    """Where an ingest keeps its state: ``<run_dir>/ingest/``."""
    return Path(run_dir) / INGEST_SUBDIR


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", "utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict]:
    """Every parseable line (a torn last line, left by a crash, is ignored)."""
    if not path.exists():
        return []
    out = []
    # "\n" only — splitlines() would also break on U+2028/NEL inside a paragraph.
    for raw in path.read_text(encoding="utf-8").split("\n"):
        if not raw.strip():
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            logger.warning("skipping a malformed line in %s", path.name)
    return out


def _append_jsonl(path: Path, record: dict) -> None:
    data = path.read_bytes() if path.exists() else b""
    if data and not data.endswith(b"\n"):  # repair a torn last line before appending
        cut = data.rfind(b"\n")
        path.write_bytes(data[: cut + 1] if cut >= 0 else b"")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        fh.flush()


def load_manifest(run_dir: Path | str) -> dict[str, Any]:
    path = ingest_dir(run_dir) / MANIFEST
    if not path.exists():
        raise IngestError(f"no Lab ingest at {ingest_dir(run_dir)} (submit_ingest_batch first)")
    return json.loads(path.read_text(encoding="utf-8"))


def load_documents(run_dir: Path | str) -> list[Document]:
    return [Document(r["name"], r["text"])
            for r in _read_jsonl(ingest_dir(run_dir) / DOCUMENTS)]


def _config(docs: list[Document], body: dict[str, Any], theme: str) -> dict[str, Any]:
    """Everything that shapes the extractions — hashed to refuse a foreign run dir."""
    probe = {k: v for k, v in body.items() if k != "messages"}
    template = graph_builder.render_extraction_request("", "", theme)
    return {
        "theme": theme,
        "request": probe,
        "prompt_sha256": hashlib.sha256(
            json.dumps(template, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "corpus_sha256": hashlib.sha256(
            "\x1e".join(d.sha256 for d in docs).encode("utf-8")
        ).hexdigest(),
        "documents": len(docs),
    }


# ── Submit / poll ────────────────────────────────────────────────────────────
async def submit_ingest_batch(
    corpus: Any,
    model: str = DEFAULT_MODEL,
    run_dir: Path | str | None = None,
    *,
    theme: str = DEFAULT_THEME,
    client: Any = None,
    run_id: str | None = None,
    max_usd: float | None = None,
    max_output_tokens: int | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Submit one extraction request per document as an OpenAI batch. Resumable.

    Refused (:class:`IngestRefused`, nothing sent) when ``max_usd`` is given and
    the upper bound exceeds it or cannot be computed. Re-submitting the same
    corpus + config to the same ``run_dir`` reuses what is already in flight; a
    different one is refused. Never writes to the knowledge graph.
    """
    if run_dir is None:
        raise IngestError("submit_ingest_batch needs a run_dir")
    settings = settings or get_settings()
    docs = corpus_documents(corpus)
    root = ingest_dir(run_dir)
    probe_body = extraction_request_body(docs[0].text, docs[0].name, model=model, theme=theme,
                                         settings=settings, max_output_tokens=max_output_tokens)
    config = _config(docs, probe_body, theme)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    manifest_path = root / MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != config_hash:
            raise IngestError(
                f"{root} holds a different ingest (config hash {manifest.get('config_hash')} "
                f"≠ {config_hash}); pick another run_dir"
            )
    else:
        manifest = {}
    if manifest and run_id and run_id != manifest.get("run_id"):
        # A new run id would give every request a new custom_id, so the batch
        # already in flight would not be recognised — and would be paid twice.
        raise IngestError(
            f"{root} was submitted as run {manifest.get('run_id')!r}; resume it under that "
            f"run id, not {run_id!r}"
        )
    run_id = manifest.get("run_id") or run_id or f"ingest-{config_hash[:12]}"
    if "|" in run_id:
        raise IngestError("run_id must not contain '|' (it separates custom_id fields)")

    plan = plan_ingest(docs, model=model, theme=theme, batch=True,
                       max_output_tokens=max_output_tokens, settings=settings)
    upper = plan.extraction.upper_usd
    if max_usd is not None and (upper is None or upper > max_usd):
        reason = (
            f"no price on file for {model!r}: the cap cannot be checked" if upper is None else
            f"upper bound {cost.format_usd(upper)} exceeds the cap {cost.format_usd(max_usd)}"
        )
        raise IngestRefused(f"ingest refused: {reason}", plan)

    if not manifest:
        manifest = {
            "run_id": run_id,
            "config_hash": config_hash,
            "config": config,
            "model": model,
            "theme": theme,
            "max_output_tokens": max_output_tokens,
            "created_at": _now(),
            "status": "planned",
            "estimate": plan.to_dict(),
            "max_usd": max_usd,
            "documents": [{"index": i, "name": d.name, "sha256": d.sha256, "chars": len(d.text)}
                          for i, d in enumerate(docs)],
        }
        _write_jsonl(root / DOCUMENTS, ({"index": i, "name": d.name, "text": d.text}
                                        for i, d in enumerate(docs)))
        _write_json(manifest_path, manifest)

    requests = build_ingest_requests(docs, run_id=run_id, model=model, theme=theme,
                                     max_output_tokens=max_output_tokens, settings=settings)
    # A request with a collected result is never resent from here — a failed one
    # only through resubmit_failed_ingest (an explicit decision to spend again).
    # Requests still in flight are resent unchanged, which batch.submit
    # recognises and reuses instead of creating a second batch.
    already = batch.latest_records(root)
    pending = [r for r in requests if r["custom_id"] not in already]
    info = await batch.submit(root, pending, run_id=run_id, client=client,
                              metadata={"synapse_phase": "ingest"})
    if pending:  # nothing left to send leaves an applied/collected ingest as it is
        manifest["status"] = "submitted"
        manifest.setdefault("submitted_at", _now())
        manifest["batch"] = info
        _write_json(manifest_path, manifest)
    return {"run_id": run_id, "ingest_dir": str(root), "documents": len(docs),
            "requests": len(pending), "batch": info, "estimate": plan.to_dict()}


async def poll_ingest(run_dir: Path | str, *, client: Any = None) -> dict[str, Any]:
    """The extraction batch's status (``batch.poll`` on ``<run_dir>/ingest``)."""
    load_manifest(run_dir)
    return await batch.poll(ingest_dir(run_dir), client=client)


async def resubmit_failed_ingest(run_dir: Path | str, *, client: Any = None) -> dict[str, Any]:
    """Resend exactly the extraction requests whose latest result is an error."""
    manifest = load_manifest(run_dir)
    info = await batch.resubmit_failed(ingest_dir(run_dir), run_id=manifest["run_id"],
                                       client=client)
    if info["requests"]:
        manifest["status"] = "submitted"
        manifest.setdefault("resubmissions", []).append({"at": _now(), "batch": info})
        _write_json(ingest_dir(run_dir) / MANIFEST, manifest)
    return info


# ── Apply ────────────────────────────────────────────────────────────────────
def reply_content(record: dict) -> tuple[str | None, str | None]:
    """``(content, finish_reason)`` of a successful Batch output line, else ``(None, None)``."""
    body = batch.record_body(record)
    if body is None:
        return None, None
    choices = body.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") or {}
    content = message.get("content")
    return (content if isinstance(content, str) else None), first.get("finish_reason")


async def _emit(on_progress: ProgressCallback | None, event: dict) -> None:
    if on_progress is None:
        return
    maybe = on_progress(event)
    if asyncio.iscoroutine(maybe):
        await maybe


async def detect_communities_without_summaries(
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Louvain community detection and persistence with NO LLM call.

    The same steps as ``communities.detect_and_summarize`` on its no-LLM path:
    load, cluster (seeded), then persist each community with the fallback
    title and summary that path writes, embedded and linked to its members.
    """
    from app.services import communities

    settings = get_settings()
    await _emit(on_progress, {"type": "progress", "stage": "loading_graph"})
    try:
        rows = await communities._fetch_graph_rows()
    except communities.GraphLoadError as e:
        logger.warning("community detection aborted (%s); communities left untouched", e)
        return {"communities": 0, "summarized": 0, "modularity": 0.0, "error": str(e)}
    _graph, _meta, _edges, found, modularity = await asyncio.to_thread(
        communities._build_and_detect, rows, settings
    )
    if not found:
        await communities._wipe_communities()
        return {"communities": 0, "summarized": 0, "modularity": modularity}
    records = [
        {
            "id": communities.community_id(members),
            "title": communities._fallback_title(members),
            # Verbatim the fallback detect_and_summarize writes without an LLM.
            "summary": f"A cluster of {len(members)} closely related entities.",
            "size": len(members),
            "members": members,
            "llm_ok": False,
        }
        for members in found
    ]
    await _emit(on_progress, {"type": "progress", "stage": "writing_communities",
                              "total": len(records)})
    vectors = await communities._embed_communities(records)
    written = await communities._persist_communities(records, vectors)
    return {"communities": written, "summarized": 0, "modularity": modularity}


async def apply_ingest_batch(
    run_dir: Path | str,
    *,
    client: Any = None,
    community_summaries: bool = False,
    detect_communities: bool = True,
    allow_failed: bool = False,
    clear_graph: bool = False,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Write the batch's extractions to the graph, one document per paragraph. Resumable.

    Returns the ingest manifest. While the batch is still running it returns
    ``{"status": "pending", ...}`` and writes nothing. WRITES TO NEO4J.
    """
    root = ingest_dir(run_dir)
    manifest = load_manifest(run_dir)
    if manifest.get("status") == "applied":
        return manifest  # already written; delete applied.jsonl to write it again
    status = await batch.poll(root, client=client)
    if not status["done"]:
        return {"status": "pending", "batch": status, "run_id": manifest["run_id"]}
    await batch.collect(root, client=client)
    latest = batch.latest_records(root)
    docs = load_documents(run_dir)
    expected = {d["name"]: d["sha256"] for d in manifest["documents"]}
    if [d.name for d in docs] != [d["name"] for d in manifest["documents"]] or any(
        expected[d.name] != d.sha256 for d in docs
    ):
        raise IngestError(f"{root / DOCUMENTS} does not match the manifest's corpus")

    run_id = manifest["run_id"]
    replies: list[tuple[str | None, str | None]] = []
    failed: list[str] = []
    for i, doc in enumerate(docs):
        record = latest.get(custom_id(run_id, i))
        content, finish = reply_content(record) if record else (None, None)
        if content is None:
            failed.append(doc.name)
        replies.append((content, finish))
    if failed and not allow_failed:
        raise IngestIncomplete(
            f"{len(failed)} of {len(docs)} extraction(s) have no successful reply "
            f"(e.g. {failed[:3]}). Resend them with resubmit_failed_ingest(run_dir) and apply "
            "again, or pass allow_failed=True to store them with no entities (what the "
            "realtime pipeline does when an extraction call fails). Nothing was written.",
            failed,
        )
    settings = get_settings()
    if not settings.store_source_chunks:
        raise IngestError(
            "STORE_SOURCE_CHUNKS is off: the documents would leave no :Chunk nodes, and the "
            "Lab's HotpotQA check (and every passage arm) needs them. Nothing was written."
        )

    applied_path = root / APPLIED
    applied = {r["index"]: r for r in _read_jsonl(applied_path)
               if r.get("sha256") == expected.get(r.get("name"))}
    if clear_graph and not applied:
        from benchmarks.public import run_hotpotqa

        await run_hotpotqa.clear_graph()
        manifest["cleared_at"] = _now()
    manifest["status"] = "applying"
    manifest.setdefault("apply_started_at", _now())
    _write_json(root / MANIFEST, manifest)

    total = len(docs)
    for i, doc in enumerate(docs):
        if i in applied:
            continue
        content, finish = replies[i]
        parsed = graph_builder.parse_extraction(content)
        result = await graph_builder.build_knowledge_graph_from_extractions(
            [doc.text], [content], doc.name, manifest.get("theme") or DEFAULT_THEME
        )
        if content is None:
            outcome = "failed"
        elif not parsed["entities"] and not parsed["relationships"]:
            outcome = "empty"
        else:
            outcome = "ok"
        record = {"index": i, "name": doc.name, "sha256": doc.sha256, "extraction": outcome,
                  "finish_reason": finish, "result": result, "at": _now()}
        _append_jsonl(applied_path, record)
        applied[i] = record
        await _emit(on_progress, {"type": "progress", "stage": "applying",
                                  "processed": len(applied), "total": total,
                                  "document": doc.name})

    stats = {"documents": 0, "nodes": 0, "edges": 0, "chunks": 0, "merged": 0,
             "failed_extractions": 0, "empty_extractions": 0, "truncated": 0}
    for record in applied.values():
        result = record.get("result") or {}
        stats["documents"] += 1
        stats["nodes"] += int(result.get("nodes_created", 0))
        stats["edges"] += int(result.get("relationships_created", 0))
        stats["chunks"] += int(result.get("chunks_stored", 0))
        stats["merged"] += int(result.get("entities_merged", 0))
        stats["failed_extractions"] += int(record.get("extraction") == "failed")
        stats["empty_extractions"] += int(record.get("extraction") == "empty")
        stats["truncated"] += int(record.get("finish_reason") == "length")

    if detect_communities:
        await _emit(on_progress, {"type": "progress", "stage": "communities",
                                  "summaries": community_summaries})
        if community_summaries:
            from app.services.communities import detect_and_summarize

            manifest["communities"] = await detect_and_summarize()
        else:
            manifest["communities"] = await detect_communities_without_summaries(on_progress)
        manifest["communities"]["summaries"] = community_summaries
    else:
        manifest["communities"] = None

    from benchmarks.public import run_hotpotqa

    present = await run_hotpotqa.graph_documents()
    missing = [d.name for d in docs if d.name not in present]
    spend = batch.status(root)
    manifest.update(
        status="applied" if not missing else "applied_with_missing",
        applied_at=_now(),
        stats=stats,
        verified={"documents_in_graph": len(docs) - len(missing), "missing": missing[:50],
                  "missing_count": len(missing)},
        actual={"usd": spend["usd"], "usage": spend["usage"], "batch": True,
                "model": manifest.get("model")},
    )
    _write_json(root / MANIFEST, manifest)
    return manifest


def ingest_usd(run_dir: Path | str) -> float | None:
    """The measured extraction spend at batch prices (for ``LabRun.ingest_usd``)."""
    return batch.status(ingest_dir(run_dir))["usd"]
