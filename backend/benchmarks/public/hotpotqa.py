# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""HotpotQA dev/distractor: acquisition, deterministic capped sampling, corpus stats.

Why this file exists
====================
The in-repo benchmark (``benchmarks/``) is self-authored — 34 passages and 14
questions produced by the same process that built the graph. Its own "Threats to
validity" concedes that, and a reviewer is right to discount it. HotpotQA (Yang et
al., EMNLP 2018) is a recognised multi-hop QA benchmark with published baselines,
and both its corpus *and* its questions come from outside this project.

Scope of THIS module: get the data, sample it reproducibly, describe it.
**It makes no LLM calls and costs nothing but bandwidth.** Ingestion, retrieval and
scoring live elsewhere; keeping acquisition separate is what lets the expensive
phases be re-run against a provably identical sample.

Provenance (checked 2026-07-21)
===============================
The official host published on hotpotqa.github.io —

    http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

— **did not respond**: TCP connect timed out on both http and https (not a 404 —
the host is unreachable). The fallback is the HuggingFace datasets-server rows API
for ``hotpotqa/hotpot_qa``, config ``distractor``, split ``validation``: plain HTTP
returning JSON, paginated 100 rows at a time, no ``datasets`` package required.
``ensure_dataset()`` still tries the official URL first (cheaply, short timeout) so
this code starts working again the day CMU brings the host back, and it records
which source actually served the bytes in ``data/provenance.json``.

The fallback rows carry a column-oriented encoding of the same records; they are
transcoded to the *official* on-disk schema before caching, so exactly one schema
reaches the parser regardless of source.

What the real file actually contains (measured 2026-07-21, sha256 a9b4c590…)
===========================================================================
7,405 records, **0 skipped as malformed**, all labelled ``hard``; 5,918 bridge /
1,487 comparison; 73,700 paragraphs, 14,810 gold, 42.0 M characters. Two upstream
quirks worth knowing before trusting any downstream metric:

* "10 paragraphs per question" is the design, not an invariant — 7,345 questions
  have 10, the other 60 have between 2 and 9.
* Exactly one supporting fact in the whole split points past the end of its
  paragraph. It is dropped and counted (``dangling_supporting_facts``) rather
  than discarding an otherwise sound question.

**Two gold paragraphs per question holds for all 7,405.** That is the invariant
downstream retrieval metrics may lean on; the other two are not.

Caching
=======
``data/`` is gitignored (~45 MB). Re-running never re-downloads unless the cache is
missing or ``force=True``. The cache is written atomically (temp file + rename), so
an interrupted download cannot leave a half-file that a later run trusts.

Sampling
========
Canonicalise by sorting on question id, seeded-shuffle, take the first *n*. Two
consequences worth stating: the same ``(seed, n)`` always yields the same set, and
the sample for n=20 is a strict prefix of the sample for n=50 — so growing a run
re-uses everything already extracted. ``random.Random`` is CPython's Mersenne
Twister; the shuffle is stable across CPython 3.x but is not a cross-language
guarantee, which is precisely why ``SampleResult`` records the chosen ids verbatim.
The ids, not the algorithm, are the reproducibility contract.

CLI
===
    python -m benchmarks.public.hotpotqa --n 20 --seed 20260721
    python -m benchmarks.public.hotpotqa --n 20 --json      # machine-readable
    python -m benchmarks.public.hotpotqa --probe            # is the official host up?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ── Where the data lives ─────────────────────────────────
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CACHE_PATH = DATA_DIR / "hotpot_dev_distractor_v1.json"
PROVENANCE_PATH = DATA_DIR / "provenance.json"

# ── Sources ──────────────────────────────────────────────
#: The canonical host from hotpotqa.github.io. Unreachable as of 2026-07-21;
#: tried first anyway, with a short timeout, so recovery is automatic.
OFFICIAL_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

#: Fallback: HuggingFace datasets-server. Plain HTTP + json — deliberately NOT the
#: ``datasets`` package, which would drag pyarrow into a backend that has no other
#: use for it.
HF_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "hotpotqa/hotpot_qa"
HF_CONFIG = "distractor"
HF_SPLIT = "validation"
HF_PAGE_SIZE = 100  # server-side maximum
#: Seconds between pages. The anonymous rows API rate-limits around 80 req/min and
#: the full split is 75 pages — an unthrottled loop trips a 429 near the end.
HF_PAGE_PAUSE = 1.0

#: Rows in the full dev/distractor split. Used only as a sanity check on a
#: completed fallback download — never as a substitute for counting.
EXPECTED_DEV_RECORDS = 7405

_USER_AGENT = "synapse-benchmarks/1.0 (+https://github.com/ahmedmaaloul/synapse)"
_OFFICIAL_PROBE_TIMEOUT = 15.0
_DEFAULT_TIMEOUT = 120.0

#: HotpotQA's own hop labels. ``bridge`` questions are the genuinely multi-hop
#: ones — you must resolve an entity in paragraph A to know that paragraph B is
#: even relevant. ``comparison`` questions name both entities up front, so a
#: retriever can find both paragraphs without ever traversing anything. Any
#: aggregate that mixes them hides the result that actually matters.
HOP_TYPES = ("bridge", "comparison")
UNKNOWN_HOP_TYPE = "unknown"


# ── Normalised structures ────────────────────────────────
@dataclass(frozen=True, slots=True)
class Paragraph:
    """One of the 10 candidate paragraphs attached to a question."""

    title: str
    sentences: tuple[str, ...]
    is_gold: bool
    #: Indices into ``sentences`` that HotpotQA labels as supporting facts.
    #: Always empty for a distractor.
    supporting_sentences: tuple[int, ...] = ()

    @property
    def text(self) -> str:
        return "".join(self.sentences)

    @property
    def char_count(self) -> int:
        return len(self.title) + sum(len(s) for s in self.sentences)

    def supporting_text(self) -> tuple[str, ...]:
        return tuple(self.sentences[i] for i in self.supporting_sentences)


@dataclass(frozen=True, slots=True)
class HotpotQuestion:
    """A HotpotQA record, normalised and validated."""

    id: str
    question: str
    answer: str
    #: ``bridge`` | ``comparison`` | ``unknown``
    hop_type: str
    #: ``easy`` | ``medium`` | ``hard`` | ``unknown`` (dev/distractor is all ``hard``)
    level: str
    paragraphs: tuple[Paragraph, ...]
    #: Supporting facts whose sentence index pointed past the end of its
    #: paragraph. A known upstream quirk; dropped, counted, never silently fixed.
    dangling_supporting_facts: tuple[tuple[str, int], ...] = ()

    @property
    def gold_paragraphs(self) -> tuple[Paragraph, ...]:
        return tuple(p for p in self.paragraphs if p.is_gold)

    @property
    def distractor_paragraphs(self) -> tuple[Paragraph, ...]:
        return tuple(p for p in self.paragraphs if not p.is_gold)

    @property
    def supporting_facts(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (p.title, i) for p in self.paragraphs for i in p.supporting_sentences
        )

    @property
    def char_count(self) -> int:
        return sum(p.char_count for p in self.paragraphs)


@dataclass(frozen=True, slots=True)
class SkippedRecord:
    """A record the parser refused, with the reason. Skipped, never crashed on."""

    index: int
    id: str
    reason: str


class MalformedRecord(ValueError):
    """Raised by :func:`parse_record`; caught and tallied by :func:`parse_records`."""


# ── Parsing ──────────────────────────────────────────────
def _as_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _coerce_context(raw: Any) -> list[tuple[str, list[str]]]:
    """Accept both encodings of ``context`` and return one shape.

    Official file: ``[[title, [sentence, ...]], ...]``
    HuggingFace rows: ``{"title": [...], "sentences": [[...], ...]}``
    """
    if isinstance(raw, dict):
        titles = raw.get("title")
        sentence_lists = raw.get("sentences")
        if not isinstance(titles, list) or not isinstance(sentence_lists, list):
            raise MalformedRecord("context dict lacks parallel title/sentences lists")
        if len(titles) != len(sentence_lists):
            raise MalformedRecord("context title/sentences lists differ in length")
        pairs: Iterable[Any] = zip(titles, sentence_lists, strict=True)
    elif isinstance(raw, list):
        pairs = raw
    else:
        raise MalformedRecord(f"context has type {type(raw).__name__}, expected list or dict")

    out: list[tuple[str, list[str]]] = []
    for entry in pairs:
        if not isinstance(entry, list | tuple) or len(entry) != 2:
            raise MalformedRecord("context entry is not a [title, sentences] pair")
        title, sentences = entry
        if not isinstance(title, str) or not title:
            raise MalformedRecord("context entry has an empty or non-string title")
        if not isinstance(sentences, list | tuple):
            raise MalformedRecord(f"paragraph {title!r} has non-list sentences")
        if not all(isinstance(s, str) for s in sentences):
            raise MalformedRecord(f"paragraph {title!r} contains a non-string sentence")
        out.append((title, list(sentences)))
    if not out:
        raise MalformedRecord("context is empty")
    return out


def _coerce_supporting_facts(raw: Any) -> list[tuple[str, int]]:
    """Accept both encodings of ``supporting_facts`` and return ``[(title, sent_id)]``."""
    if isinstance(raw, dict):
        titles = raw.get("title")
        sent_ids = raw.get("sent_id")
        if not isinstance(titles, list) or not isinstance(sent_ids, list):
            raise MalformedRecord("supporting_facts dict lacks parallel title/sent_id lists")
        if len(titles) != len(sent_ids):
            raise MalformedRecord("supporting_facts title/sent_id lists differ in length")
        pairs: Iterable[Any] = zip(titles, sent_ids, strict=True)
    elif isinstance(raw, list):
        pairs = raw
    else:
        raise MalformedRecord(
            f"supporting_facts has type {type(raw).__name__}, expected list or dict"
        )

    out: list[tuple[str, int]] = []
    for entry in pairs:
        if not isinstance(entry, list | tuple) or len(entry) != 2:
            raise MalformedRecord("supporting_facts entry is not a [title, sent_id] pair")
        title, sent_id = entry
        if not isinstance(title, str) or not title:
            raise MalformedRecord("supporting fact has an empty or non-string title")
        if isinstance(sent_id, bool) or not isinstance(sent_id, int):
            raise MalformedRecord(f"supporting fact for {title!r} has a non-integer sent_id")
        out.append((title, sent_id))
    if not out:
        raise MalformedRecord("supporting_facts is empty")
    return out


def parse_record(raw: Any, *, index: int = -1) -> HotpotQuestion:
    """Normalise one raw record, or raise :class:`MalformedRecord`.

    The validation is deliberately strict about anything that would silently
    corrupt a downstream measurement, and deliberately tolerant of the one known
    upstream quirk (a supporting-fact sentence index past the end of its
    paragraph), which is dropped and reported rather than treated as fatal.
    """
    if not isinstance(raw, dict):
        raise MalformedRecord(f"record is a {type(raw).__name__}, expected an object")

    qid = _as_text(raw.get("_id")) or _as_text(raw.get("id"))
    if not qid:
        raise MalformedRecord("record has no non-empty _id/id")

    question = _as_text(raw.get("question"))
    if not question or not question.strip():
        raise MalformedRecord("record has no non-empty question")

    answer = _as_text(raw.get("answer"))
    if answer is None or not answer.strip():
        raise MalformedRecord("record has no non-empty answer")

    context = _coerce_context(raw.get("context"))
    facts = _coerce_supporting_facts(raw.get("supporting_facts"))

    # title -> every paragraph index carrying it (duplicates exist upstream and
    # are not an error; every copy is treated as gold).
    by_title: dict[str, list[int]] = {}
    for i, (title, _) in enumerate(context):
        by_title.setdefault(title, []).append(i)

    supporting: dict[int, set[int]] = {}
    dangling: list[tuple[str, int]] = []
    for title, sent_id in facts:
        positions = by_title.get(title)
        if positions is None:
            # The gold paragraph itself is absent, so the question is not
            # answerable from its own context. Not repairable — skip the record.
            raise MalformedRecord(f"supporting fact cites {title!r}, absent from context")
        for pos in positions:
            if 0 <= sent_id < len(context[pos][1]):
                supporting.setdefault(pos, set()).add(sent_id)
            else:
                dangling.append((title, sent_id))

    if not supporting:
        raise MalformedRecord("no supporting fact resolved to a real sentence")

    # Gold == carries at least one supporting fact that resolved to a real
    # sentence. Deriving it from ``supporting`` rather than from the raw titles
    # keeps a single definition: nothing can be gold with nothing to support.
    paragraphs = tuple(
        Paragraph(
            title=title,
            sentences=tuple(sentences),
            is_gold=i in supporting,
            supporting_sentences=tuple(sorted(supporting.get(i, ()))),
        )
        for i, (title, sentences) in enumerate(context)
    )

    hop_type = _as_text(raw.get("type")) or UNKNOWN_HOP_TYPE
    level = _as_text(raw.get("level")) or UNKNOWN_HOP_TYPE

    return HotpotQuestion(
        id=qid,
        question=question,
        answer=answer,
        hop_type=hop_type,
        level=level,
        paragraphs=paragraphs,
        dangling_supporting_facts=tuple(dict.fromkeys(dangling)),
    )


def parse_records(raws: Sequence[Any]) -> tuple[list[HotpotQuestion], list[SkippedRecord]]:
    """Parse a whole file. A malformed record is skipped and reported, not fatal."""
    parsed: list[HotpotQuestion] = []
    skipped: list[SkippedRecord] = []
    for index, raw in enumerate(raws):
        try:
            parsed.append(parse_record(raw, index=index))
        except MalformedRecord as exc:
            qid = ""
            if isinstance(raw, dict):
                qid = _as_text(raw.get("_id")) or _as_text(raw.get("id")) or ""
            skipped.append(SkippedRecord(index=index, id=qid, reason=str(exc)))
    return parsed, skipped


# ── Deterministic sampling ───────────────────────────────
def sample_questions(
    questions: Sequence[HotpotQuestion], n: int, *, seed: int
) -> list[HotpotQuestion]:
    """Pick *n* questions reproducibly: sort by id, seeded shuffle, take the head.

    ``n`` is a hard cap, and it is the ONLY thing that bounds the cost of every
    downstream phase — there is no code path here that processes "the rest".
    Requesting more than the pool holds returns the whole pool rather than
    quietly padding or repeating.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    pool = sorted(questions, key=lambda q: q.id)
    random.Random(seed).shuffle(pool)
    return pool[:n]


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Everything a reader needs to judge what the sample actually contains."""

    n_questions: int
    n_paragraphs: int
    n_gold_paragraphs: int
    n_distractor_paragraphs: int
    n_sentences: int
    n_supporting_facts: int
    n_dangling_supporting_facts: int
    total_context_chars: int
    total_gold_chars: int
    total_question_chars: int
    hop_types: dict[str, int]
    levels: dict[str, int]
    mean_paragraphs_per_question: float
    mean_gold_paragraphs_per_question: float
    mean_context_chars_per_question: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_questions": self.n_questions,
            "n_paragraphs": self.n_paragraphs,
            "n_gold_paragraphs": self.n_gold_paragraphs,
            "n_distractor_paragraphs": self.n_distractor_paragraphs,
            "n_sentences": self.n_sentences,
            "n_supporting_facts": self.n_supporting_facts,
            "n_dangling_supporting_facts": self.n_dangling_supporting_facts,
            "total_context_chars": self.total_context_chars,
            "total_gold_chars": self.total_gold_chars,
            "total_question_chars": self.total_question_chars,
            "hop_types": dict(self.hop_types),
            "levels": dict(self.levels),
            "mean_paragraphs_per_question": round(self.mean_paragraphs_per_question, 3),
            "mean_gold_paragraphs_per_question": round(
                self.mean_gold_paragraphs_per_question, 3
            ),
            "mean_context_chars_per_question": round(self.mean_context_chars_per_question, 1),
        }


def corpus_stats(questions: Sequence[HotpotQuestion]) -> CorpusStats:
    """Describe a sample. Every number here is computed, never hand-copied."""
    n = len(questions)
    paragraphs = [p for q in questions for p in q.paragraphs]
    gold = [p for p in paragraphs if p.is_gold]
    context_chars = sum(p.char_count for p in paragraphs)
    hop_types = Counter(q.hop_type for q in questions)
    levels = Counter(q.level for q in questions)
    return CorpusStats(
        n_questions=n,
        n_paragraphs=len(paragraphs),
        n_gold_paragraphs=len(gold),
        n_distractor_paragraphs=len(paragraphs) - len(gold),
        n_sentences=sum(len(p.sentences) for p in paragraphs),
        n_supporting_facts=sum(len(q.supporting_facts) for q in questions),
        n_dangling_supporting_facts=sum(len(q.dangling_supporting_facts) for q in questions),
        total_context_chars=context_chars,
        total_gold_chars=sum(p.char_count for p in gold),
        total_question_chars=sum(len(q.question) for q in questions),
        hop_types={k: hop_types[k] for k in sorted(hop_types)},
        levels={k: levels[k] for k in sorted(levels)},
        mean_paragraphs_per_question=len(paragraphs) / n if n else 0.0,
        mean_gold_paragraphs_per_question=len(gold) / n if n else 0.0,
        mean_context_chars_per_question=context_chars / n if n else 0.0,
    )


@dataclass(frozen=True, slots=True)
class SampleResult:
    """A sample plus the provenance needed to reproduce it exactly."""

    questions: list[HotpotQuestion]
    seed: int
    requested_n: int
    pool_size: int
    skipped: list[SkippedRecord] = field(default_factory=list)
    source: str = "unknown"

    @property
    def ids(self) -> list[str]:
        return [q.id for q in self.questions]

    @property
    def stats(self) -> CorpusStats:
        return corpus_stats(self.questions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": "hotpotqa-dev-distractor",
            "source": self.source,
            "seed": self.seed,
            "requested_n": self.requested_n,
            "sampled_n": len(self.questions),
            "pool_size": self.pool_size,
            "sampling": "sort by id, then random.Random(seed).shuffle, take the first n",
            "question_ids": self.ids,
            "skipped_records": [
                {"index": s.index, "id": s.id, "reason": s.reason} for s in self.skipped
            ],
            "stats": self.stats.to_dict(),
        }


# ── Acquisition ──────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Provenance:
    """Which bytes came from where, recorded next to the cache."""

    source_url: str
    retrieved_at: str
    n_records: int
    sha256: str
    n_bytes: int
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": "hotpotqa-dev-distractor",
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
            "n_records": self.n_records,
            "sha256": self.sha256,
            "n_bytes": self.n_bytes,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        return cls(
            source_url=str(data.get("source_url", "unknown")),
            retrieved_at=str(data.get("retrieved_at", "unknown")),
            n_records=int(data.get("n_records", 0)),
            sha256=str(data.get("sha256", "")),
            n_bytes=int(data.get("n_bytes", 0)),
            truncated=bool(data.get("truncated", False)),
        )


def _open(url: str, timeout: float):
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 — fixed https hosts


def probe_official(timeout: float = _OFFICIAL_PROBE_TIMEOUT) -> tuple[bool, str]:
    """Is the canonical CMU host serving the file? Returns ``(ok, detail)``.

    Cheap by construction: a Range request for the first 64 bytes, so a live host
    costs nothing and a dead one costs only the connect timeout.
    """
    request = urllib.request.Request(
        OFFICIAL_URL, headers={"User-Agent": _USER_AGENT, "Range": "bytes=0-63"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            head = response.read(64)
            code = getattr(response, "status", 0)
            if head.lstrip()[:1] != b"[":
                return False, f"HTTP {code} but the body does not start with a JSON array"
            return True, f"HTTP {code}, body starts with a JSON array"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} {exc.reason}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"unreachable: {exc}"


def _download_official(dest: Path, timeout: float) -> int:
    """Stream the official file to ``dest``. Returns the record count."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _open(OFFICIAL_URL, timeout) as response, tmp.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    records = json.loads(tmp.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("official download did not decode to a non-empty JSON array")
    tmp.replace(dest)
    return len(records)


def _hf_page(offset: int, length: int, timeout: float, *, attempts: int = 6) -> dict[str, Any]:
    """One page of rows, with 429-aware backoff.

    The anonymous rows API rate-limits at roughly 80 requests/minute; the full
    split is 75 pages, so an unthrottled loop reliably trips it near the end.
    ``Retry-After`` is honoured when the server sends it.
    """
    query = urllib.parse.urlencode(
        {
            "dataset": HF_DATASET,
            "config": HF_CONFIG,
            "split": HF_SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with _open(f"{HF_ROWS_ENDPOINT}?{query}", timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = float(retry_after) if (retry_after or "").isdigit() else 15.0 * (
                    attempt + 1
                )
            elif 500 <= exc.code < 600:
                delay = min(2**attempt, 30)
            else:
                raise RuntimeError(
                    f"HuggingFace rows API returned HTTP {exc.code} at offset {offset}"
                ) from exc
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"HuggingFace rows API failed at offset {offset}: {last}")


def _to_official_schema(row: dict[str, Any]) -> dict[str, Any]:
    """Transcode a HuggingFace row into the official on-disk record shape.

    Column-oriented context/supporting_facts become the official nested lists, so
    the cache file has exactly one schema no matter which source produced it.
    """
    context = _coerce_context(row.get("context"))
    facts = _coerce_supporting_facts(row.get("supporting_facts"))
    return {
        "_id": row.get("id") or row.get("_id"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "type": row.get("type"),
        "level": row.get("level"),
        "supporting_facts": [[title, sent_id] for title, sent_id in facts],
        "context": [[title, sentences] for title, sentences in context],
    }


def _read_shard(shard: Path) -> tuple[list[dict[str, Any]], int]:
    """Replay a resume shard: ``(records so far, rows already consumed)``.

    One JSONL line per *row* — ``null`` for a row that failed transcoding — so
    the line count is exactly the API offset to resume from even when rows were
    dropped. A truncated final line (killed mid-write) is discarded.
    """
    records: list[dict[str, Any]] = []
    rows = 0
    if not shard.exists():
        return records, rows
    for line in shard.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            break  # torn last line
        rows += 1
        if entry is not None:
            records.append(entry)
    return records, rows


def _download_huggingface(dest: Path, timeout: float, max_records: int | None) -> int:
    """Page the datasets-server rows API into ``dest`` in the official schema.

    Resumable: each page is appended to a ``.jsonl.part`` shard before the next
    request goes out, so a rate-limit or a Ctrl-C costs only the current page
    rather than the whole 45 MB.
    """
    shard = dest.with_suffix(".jsonl.part")
    records, offset = _read_shard(shard)
    if records:
        print(f"resuming HuggingFace download at row {offset}", file=sys.stderr)
    total: int | None = None

    with shard.open("a", encoding="utf-8") as sink:
        while True:
            want = HF_PAGE_SIZE
            if max_records is not None:
                want = min(want, max_records - len(records))
            if want <= 0:
                break
            page = _hf_page(offset, want, timeout)
            if total is None:
                total = int(page.get("num_rows_total") or 0)
            rows = page.get("rows") or []
            if not rows:
                break
            for entry in rows:
                row = entry.get("row") if isinstance(entry, dict) else None
                record: dict[str, Any] | None = None
                if isinstance(row, dict):
                    try:
                        record = _to_official_schema(row)
                    except MalformedRecord:
                        # Transcoding refuses only structurally broken rows.
                        # Recorded as a null line so the resume offset stays exact.
                        record = None
                sink.write(json.dumps(record) + "\n")
                if record is not None:
                    records.append(record)
            sink.flush()
            offset += len(rows)
            if total and offset >= total:
                break
            if max_records is not None and len(records) >= max_records:
                break
            time.sleep(HF_PAGE_PAUSE)  # stay under the anonymous rate limit

    if not records:
        raise RuntimeError("HuggingFace fallback returned no usable rows")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_text(json.dumps(records), encoding="utf-8")
    tmp.replace(dest)
    shard.unlink(missing_ok=True)
    return len(records)


def ensure_dataset(
    *,
    force: bool = False,
    max_records: int | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    allow_download: bool = True,
) -> Provenance:
    """Return provenance for a usable local cache, downloading only if needed.

    ``max_records`` truncates a *fallback* download (useful for a smoke run). It is
    recorded as ``truncated=True`` in the provenance and surfaced as ``pool_size``
    in every sample, because sampling from a truncated pool is a different
    experiment and must never look like sampling from the full split.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists() and not force:
        if PROVENANCE_PATH.exists():
            return Provenance.from_dict(json.loads(PROVENANCE_PATH.read_text(encoding="utf-8")))
        records = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        provenance = _record_provenance("unknown (pre-existing cache)", len(records), False)
        return provenance

    if not allow_download:
        raise FileNotFoundError(
            f"no cached dataset at {CACHE_PATH} and downloading is disabled "
            "(drop --no-download to fetch it)"
        )

    ok, detail = probe_official()
    if ok:
        n_records = _download_official(CACHE_PATH, timeout)
        source = OFFICIAL_URL
        truncated = False
    else:
        print(
            f"official host unavailable ({detail}) — falling back to the "
            f"HuggingFace-hosted copy of {HF_DATASET}",
            file=sys.stderr,
        )
        n_records = _download_huggingface(CACHE_PATH, timeout, max_records)
        source = (
            f"{HF_ROWS_ENDPOINT}?dataset={HF_DATASET}&config={HF_CONFIG}&split={HF_SPLIT}"
        )
        truncated = max_records is not None and n_records >= max_records

    return _record_provenance(source, n_records, truncated)


def _record_provenance(source: str, n_records: int, truncated: bool) -> Provenance:
    raw = CACHE_PATH.read_bytes()
    provenance = Provenance(
        source_url=source,
        retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
        n_records=n_records,
        sha256=hashlib.sha256(raw).hexdigest(),
        n_bytes=len(raw),
        truncated=truncated,
    )
    PROVENANCE_PATH.write_text(
        json.dumps(provenance.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def load_raw_records(path: Path | None = None) -> list[Any]:
    """Read the cached file. Does not touch the network."""
    target = path or CACHE_PATH
    records = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{target} does not contain a JSON array")
    return records


def load_sample(
    n: int,
    *,
    seed: int,
    path: Path | None = None,
    force: bool = False,
    max_records: int | None = None,
    allow_download: bool = True,
) -> SampleResult:
    """The one call the rest of the wave needs: cached data in, capped sample out."""
    if path is None:
        provenance = ensure_dataset(
            force=force, max_records=max_records, allow_download=allow_download
        )
        source = provenance.source_url
    else:
        source = str(path)
    parsed, skipped = parse_records(load_raw_records(path))
    return SampleResult(
        questions=sample_questions(parsed, n, seed=seed),
        seed=seed,
        requested_n=n,
        pool_size=len(parsed),
        skipped=skipped,
        source=source,
    )


# ── CLI ──────────────────────────────────────────────────
DEFAULT_SEED = 20260721
DEFAULT_N = 20


def format_report(result: SampleResult) -> str:
    stats = result.stats
    lines = [
        "HotpotQA — dev / distractor split",
        "=" * 64,
        f"source          : {result.source}",
        f"pool (parsed)   : {result.pool_size} questions"
        + (f"  [{len(result.skipped)} skipped as malformed]" if result.skipped else ""),
        f"sample          : n={len(result.questions)} (requested {result.requested_n}), "
        f"seed={result.seed}",
        "sampling        : sort by id -> random.Random(seed).shuffle -> first n",
        "",
        "Corpus statistics for this sample",
        "-" * 64,
        f"  questions                 : {stats.n_questions}",
        f"  paragraphs                : {stats.n_paragraphs} "
        f"({stats.mean_paragraphs_per_question:.1f} per question)",
        f"  gold paragraphs           : {stats.n_gold_paragraphs} "
        f"({stats.mean_gold_paragraphs_per_question:.2f} per question)",
        f"  distractor paragraphs     : {stats.n_distractor_paragraphs}",
        f"  sentences                 : {stats.n_sentences}",
        f"  supporting facts          : {stats.n_supporting_facts}",
        f"  dangling supporting facts : {stats.n_dangling_supporting_facts}",
        f"  total context characters  : {stats.total_context_chars:,}",
        f"  gold-only characters      : {stats.total_gold_chars:,} "
        f"({stats.total_gold_chars / stats.total_context_chars:.1%} of context)"
        if stats.total_context_chars
        else "  gold-only characters      : 0",
        f"  mean context chars / q    : {stats.mean_context_chars_per_question:,.0f}",
        "",
        "Hop-type distribution (bridge = genuinely multi-hop)",
        "-" * 64,
    ]
    for label, count in stats.hop_types.items():
        share = count / stats.n_questions if stats.n_questions else 0.0
        lines.append(f"  {label:<24}  {count:>4}  ({share:.0%})")
    lines += ["", "Difficulty levels", "-" * 64]
    for label, count in stats.levels.items():
        lines.append(f"  {label:<24}  {count:>4}")
    lines += [
        "",
        "Sampled question ids (the reproducibility contract — reproduce the sample",
        "from these ids alone, no matter what shuffle algorithm your runtime has)",
        "-" * 64,
    ]
    lines += [f"  {qid}" for qid in result.ids]
    if result.skipped:
        lines += ["", f"Skipped records ({len(result.skipped)})", "-" * 64]
        reasons = Counter(s.reason for s in result.skipped)
        lines += [f"  {count:>4}  {reason}" for reason, count in reasons.most_common()]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.public.hotpotqa",
        description="Fetch, cache and deterministically sample HotpotQA dev/distractor. "
        "Makes no LLM calls and costs nothing but bandwidth.",
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help=f"sample size (default {DEFAULT_N})")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"shuffle seed (default {DEFAULT_SEED})"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--refresh", action="store_true", help="re-download even if cached")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="cap a fallback download (smoke runs); marks the pool as truncated",
    )
    parser.add_argument(
        "--no-download", action="store_true", help="fail rather than hit the network"
    )
    parser.add_argument(
        "--probe", action="store_true", help="only check whether the official host responds"
    )
    args = parser.parse_args(argv)

    if args.probe:
        ok, detail = probe_official()
        print(f"{OFFICIAL_URL}\n  {'OK' if ok else 'DOWN'} — {detail}")
        return 0 if ok else 1

    result = load_sample(
        args.n,
        seed=args.seed,
        force=args.refresh,
        max_records=args.max_records,
        allow_download=not args.no_download,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_report(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
