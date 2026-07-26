# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""2WikiMultihopQA (dev): acquisition, deterministic capped sampling, and a thin
runner that scores it with the EXACT HotpotQA harness — a replication study.

Why this file exists
====================
``benchmarks/public/run_hotpotqa.py`` found something specific and uncomfortable
on HotpotQA: the graph-only retrieval path (system C) scores ~100 % Both-gold
under the *permissive* (containment) rule and 0.0 % under the *strict* (prose)
rule, because it returns entity blocks, never the corpus text — and 93 % of its
credited hits come from an entity NAME appearing in the context, not from any
retrieved evidence. One dataset is an anecdote. This module runs the identical
argument on a SECOND public multi-hop benchmark, **2WikiMultihopQA** (Ho et al.,
COLING 2020), whose corpus, questions and gold labels come from outside this
project — so the question "does the scoring bias replicate?" can be answered with
numbers rather than a promise.

Scope split — the same discipline as ``hotpotqa.py``
====================================================
The **dataset half** of this module (top) gets the data, samples it reproducibly
and describes it; it makes NO LLM calls and costs nothing but bandwidth. The
**runner half** (bottom, guarded behind ``run``) is the only part that spends
money, and it does not re-implement a single metric: it imports
``run_hotpotqa``'s scoring, effect-size floor, channel attribution and reporting
helpers and drives them, so 2Wiki and HotpotQA are scored by *exactly the same
rules*. The two are not allowed to drift because there is only one copy of the
metric code and this module is downstream of it.

2Wiki is HotpotQA-shaped
========================
Each 2Wiki record is, on disk, the official HotpotQA schema —
``context`` is ``[[title, [sentence, ...]], ...]`` and ``supporting_facts`` is
``[[title, sent_id], ...]`` — with two extra fields (``type`` takes four values,
``evidences`` carries the gold triples) and no ``level``. Because the shape is
identical, this module REUSES ``hotpotqa``'s strict record parser, deterministic
sampler and corpus-stats code rather than forking them; only acquisition is new.

Measured over the whole 2026-07-26 dev split (12,576 records, sha256 48b9bdc6…):
0 skipped as malformed, all carry exactly 10 paragraphs, and gold-paragraph count
is **2 for comparison / inference / compositional and 4 for bridge_comparison**
(2,751 of 12,576). That last fact is the one real difference from HotpotQA's
two-gold invariant: for a bridge_comparison question "both-gold" is really
"all-gold", i.e. every one of its four gold paragraphs retrieved. The metric —
"were all the gold paragraphs found" — is unchanged; only its arity varies, and
it is disclosed in the report's threats section rather than assumed away.

Provenance (checked 2026-07-26)
===============================
The HuggingFace **datasets-server rows API** — the plain-HTTP/JSON path
``hotpotqa.py`` uses, no ``datasets`` package — was PROBED for both community
mirrors and does not currently serve this dataset:

* ``xanhho/2WikiMultihopQA``  → HTTP 501 (the viewer refuses datasets that run a
  loading script),
* ``voidful/2WikiMultihopQA`` → HTTP 500 (``ArrowTypeError``: the server cannot
  materialise the parquet).

``ensure_dataset()`` therefore PROBES the rows API first (so this code starts
using it the day the viewer is fixed) and, finding it down, falls back to the
canonical ``dev.json`` published in the same ``voidful/2WikiMultihopQA`` repo,
fetched over plain HTTP from the HuggingFace ``resolve`` endpoint. That file is
the original authors' split in the official schema — no transcoding, one shape
reaches the parser — and which source actually served the bytes is recorded in
``data/2wiki_provenance.json``. This mirrors ``hotpotqa.py`` exactly: try the
preferred source, fall back to a canonical copy, record what happened.

Caching / sampling
==================
``data/`` is gitignored (~56 MB); a re-run never re-downloads unless the cache is
missing or ``force=True``. Sampling is ``hotpotqa.sample_questions`` verbatim:
sort by id, seeded shuffle, take the first *n* — so a fixed ``(seed, n)`` always
yields the same set and growing a run re-uses everything already extracted.

CLI
===
    python -m benchmarks.public.twowiki --n 50 --seed 20260721        # dataset stats
    python -m benchmarks.public.twowiki --n 50 --seed 20260721 --json # machine-readable
    python -m benchmarks.public.twowiki --probe                       # is the rows API up?
    python -m benchmarks.public.twowiki run --questions 5 --dry-run   # estimate the bill
    python -m benchmarks.public.twowiki run --questions 50 --seed 20260721
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The dataset half REUSES hotpotqa's parser/sampler/stats — 2Wiki is HotpotQA-
# shaped, so forking them would only invite the two to disagree. Acquisition is
# the only thing genuinely new here.
from benchmarks.public import hotpotqa
from benchmarks.public.hotpotqa import (
    CorpusStats,
    HotpotQuestion,
    MalformedRecord,
    SkippedRecord,
    corpus_stats,
    parse_records,
    sample_questions,
)

# ── Where the data lives ─────────────────────────────────
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CACHE_PATH = DATA_DIR / "2wikimultihopqa_dev_v1.json"
PROVENANCE_PATH = DATA_DIR / "2wiki_provenance.json"

# ── Sources ──────────────────────────────────────────────
#: Preferred: the datasets-server rows API — plain HTTP + JSON, no ``datasets``
#: package. As of 2026-07-26 it is DOWN for both community mirrors (see the
#: module docstring), so ``ensure_dataset`` probes it and falls back. Kept so the
#: code recovers automatically if the viewer is fixed.
HF_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "voidful/2WikiMultihopQA"
HF_CONFIG = "default"
HF_SPLIT = "validation"
HF_PAGE_SIZE = 100  # server-side maximum
HF_PAGE_PAUSE = 1.0  # stay under the anonymous rate limit

#: Fallback (the path that actually runs today): the canonical ``dev.json`` from
#: the same repo, fetched over plain HTTP from the HuggingFace ``resolve``
#: endpoint. It is the original 2Wiki split in the official on-disk schema.
RESOLVE_URL = "https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/main/dev.json"

#: Records in the full dev split. A sanity check on a completed download, never a
#: substitute for counting.
EXPECTED_DEV_RECORDS = 12576

_USER_AGENT = "synapse-benchmarks/1.0 (+https://github.com/ahmedmaaloul/synapse)"
_ROWS_PROBE_TIMEOUT = 15.0
_DEFAULT_TIMEOUT = 180.0

#: 2Wiki's four reasoning types. ``comparison`` names both entities up front (two
#: independent look-ups); the other three hide a second entity behind a bridge —
#: the case a graph edge is supposed to help with — so any aggregate that pools
#: them hides the result that matters. ``bridge_comparison`` questions carry FOUR
#: gold paragraphs, not two.
HOP_TYPES = ("comparison", "inference", "compositional", "bridge_comparison")
#: The three types whose second entity is not stated in the question. Reported as
#: a derived "bridge-like vs comparison" split alongside the per-type breakdown.
BRIDGE_LIKE_TYPES = frozenset({"inference", "compositional", "bridge_comparison"})

#: 2WikiMultihopQA is Apache-2.0 (Ho et al., COLING 2020). ``xanhho`` is the
#: official mirror of github.com/Alab-NII/2wikimultihop and declares it; nothing
#: from it enters the repository — the cache lives in the gitignored ``data/``.
TWOWIKI_LICENSE = (
    "Apache-2.0 (Ho et al., COLING 2020) — github.com/Alab-NII/2wikimultihop"
)


# ── Normalised structures ────────────────────────────────
# 2Wiki records parse into ``hotpotqa.HotpotQuestion`` unchanged, so the only new
# containers are the provenance and sample records — and they exist purely so the
# reproduced experiment cannot be confused with a HotpotQA one.
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
            "dataset": "2wikimultihopqa-dev",
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
            "dataset": "2wikimultihopqa-dev",
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
def _open(url: str, timeout: float, *, headers: dict[str, str] | None = None):
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 — fixed https hosts


def probe_rows_api(timeout: float = _ROWS_PROBE_TIMEOUT) -> tuple[bool, str]:
    """Does the datasets-server rows API serve this dataset? ``(ok, detail)``.

    Cheap: a single one-row request. A live server costs one round-trip; a dead
    one costs a 4xx/5xx we read and report. This is what ``ensure_dataset`` uses
    to decide whether to page the rows API or fall back to the canonical file.
    """
    query = urllib.parse.urlencode(
        {"dataset": HF_DATASET, "config": HF_CONFIG, "split": HF_SPLIT, "offset": 0, "length": 1}
    )
    try:
        with _open(f"{HF_ROWS_ENDPOINT}?{query}", timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not rows:
            return False, "rows API responded but returned no rows"
        return True, f"rows API live, {payload.get('num_rows_total', '?')} rows total"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} {exc.reason}"
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return False, f"unreachable: {exc}"


def _to_official_schema(row: dict[str, Any]) -> dict[str, Any]:
    """Transcode a datasets-server row into the official on-disk record shape.

    Reuses ``hotpotqa``'s coercers, which already accept both the column-oriented
    (rows API) and nested (file) encodings, so the cache holds exactly one schema
    regardless of source. 2Wiki carries ``type`` (four values) and no ``level``.
    """
    context = hotpotqa._coerce_context(row.get("context"))
    facts = hotpotqa._coerce_supporting_facts(row.get("supporting_facts"))
    return {
        "_id": row.get("_id") or row.get("id"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "type": row.get("type"),
        "supporting_facts": [[title, sent_id] for title, sent_id in facts],
        "context": [[title, sentences] for title, sentences in context],
    }


def _hf_page(offset: int, length: int, timeout: float, *, attempts: int = 6) -> dict[str, Any]:
    """One page of rows, with 429/5xx-aware backoff — the ``hotpotqa`` shape."""
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
                delay = float(retry_after) if (retry_after or "").isdigit() else 15.0 * (attempt + 1)
            elif 500 <= exc.code < 600:
                delay = min(2**attempt, 30)
            else:
                raise RuntimeError(
                    f"2Wiki rows API returned HTTP {exc.code} at offset {offset}"
                ) from exc
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"2Wiki rows API failed at offset {offset}: {last}")


def _download_rows_api(dest: Path, timeout: float, max_records: int | None) -> int:
    """Page the datasets-server rows API into ``dest`` (official schema).

    Resumable exactly as ``hotpotqa._download_huggingface`` is: every page is
    appended to a ``.jsonl.part`` shard before the next request, so a rate-limit
    or a Ctrl-C costs only the current page. Runs only if ``probe_rows_api`` says
    the viewer is live.
    """
    shard = dest.with_suffix(".jsonl.part")
    records, offset = hotpotqa._read_shard(shard)
    if records:
        print(f"resuming 2Wiki rows download at row {offset}", file=sys.stderr)
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
            time.sleep(HF_PAGE_PAUSE)

    if not records:
        raise RuntimeError("2Wiki rows API returned no usable rows")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_text(json.dumps(records), encoding="utf-8")
    tmp.replace(dest)
    shard.unlink(missing_ok=True)
    return len(records)


def _download_resolve_json(dest: Path, timeout: float) -> int:
    """Stream the canonical ``dev.json`` to ``dest``. Returns the record count.

    Atomic (temp file + rename), so an interrupted download cannot leave a
    half-file a later run would trust. The file is already the official schema,
    so nothing is transcoded.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _open(RESOLVE_URL, timeout) as response, tmp.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    records = json.loads(tmp.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("2Wiki dev.json did not decode to a non-empty JSON array")
    tmp.replace(dest)
    return len(records)


def ensure_dataset(
    *,
    force: bool = False,
    max_records: int | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    allow_download: bool = True,
) -> Provenance:
    """Return provenance for a usable local cache, downloading only if needed.

    Probes the datasets-server rows API first (the ``hotpotqa`` path); if it is
    down — which it is, today — falls back to the canonical ``dev.json``. Records
    which source served the bytes. ``max_records`` truncates the *rows API* path
    only and is flagged ``truncated=True`` so a truncated pool can never be
    mistaken for the full split.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists() and not force:
        if PROVENANCE_PATH.exists():
            return Provenance.from_dict(json.loads(PROVENANCE_PATH.read_text(encoding="utf-8")))
        records = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return _record_provenance("unknown (pre-existing cache)", len(records), False)

    if not allow_download:
        raise FileNotFoundError(
            f"no cached dataset at {CACHE_PATH} and downloading is disabled "
            "(drop --no-download to fetch it)"
        )

    ok, detail = probe_rows_api()
    if ok:
        n_records = _download_rows_api(CACHE_PATH, timeout, max_records)
        source = f"{HF_ROWS_ENDPOINT}?dataset={HF_DATASET}&config={HF_CONFIG}&split={HF_SPLIT}"
        truncated = max_records is not None and n_records >= max_records
    else:
        print(
            f"datasets-server rows API unavailable ({detail}) — falling back to the "
            f"canonical dev.json in {HF_DATASET}",
            file=sys.stderr,
        )
        n_records = _download_resolve_json(CACHE_PATH, timeout)
        source = RESOLVE_URL
        truncated = False

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
    """Cached data in, capped 2Wiki sample out — the one call the runner needs."""
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


# ── Dataset CLI (stats only — no LLM, no Neo4j) ──────────
DEFAULT_SEED = 20260721
DEFAULT_N = 20


def format_report(result: SampleResult) -> str:
    stats = result.stats
    lines = [
        "2WikiMultihopQA — dev split",
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
        f"  mean context chars / q    : {stats.mean_context_chars_per_question:,.0f}",
        "",
        "Reasoning-type distribution (comparison names both entities; the other",
        "three hide a second entity behind a bridge, where a graph edge should help)",
        "-" * 64,
    ]
    for label in HOP_TYPES:
        count = stats.hop_types.get(label, 0)
        share = count / stats.n_questions if stats.n_questions else 0.0
        lines.append(f"  {label:<24}  {count:>4}  ({share:.0%})")
    for label, count in sorted(stats.hop_types.items()):
        if label not in HOP_TYPES:
            lines.append(f"  {label:<24}  {count:>4}  (unexpected type)")
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


def build_stats_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.public.twowiki",
        description="Fetch, cache and deterministically sample 2WikiMultihopQA (dev). "
        "Makes no LLM calls and costs nothing but bandwidth.",
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help=f"sample size (default {DEFAULT_N})")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"shuffle seed (default {DEFAULT_SEED})"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--refresh", action="store_true", help="re-download even if cached")
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="cap a rows-API download (smoke runs); marks the pool as truncated",
    )
    parser.add_argument(
        "--no-download", action="store_true", help="fail rather than hit the network"
    )
    parser.add_argument(
        "--probe", action="store_true", help="only check whether the rows API serves 2Wiki"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dataset stats CLI. Mirrors ``hotpotqa.main`` — free, no model, no database."""
    args = build_stats_parser().parse_args(argv)

    if args.probe:
        ok, detail = probe_rows_api()
        print(f"{HF_ROWS_ENDPOINT} (dataset={HF_DATASET})\n  {'OK' if ok else 'DOWN'} — {detail}")
        if not ok:
            print(f"  fallback in use: {RESOLVE_URL}")
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


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER — the part that spends money. It imports run_hotpotqa's scoring, cost
# and reporting helpers and DRIVES them; it forks no metric. Everything below is
# lazy about ``run_hotpotqa`` (and therefore ``app``) so the dataset half above
# stays importable, and testable, without any of the heavy machinery.
# ══════════════════════════════════════════════════════════════════════════════
RESULTS_PATH = HERE / "results_2wiki.md"


def load_corpus(
    *,
    questions: int,
    seed: int,
    data_path: Path | None = None,
    allow_download: bool = True,
):
    """Load the capped 2Wiki sample and pool it into ``run_hotpotqa.Corpus``.

    Free: network at worst, never a model call. The pooling, deduplication and
    seam-disclosure are ``run_hotpotqa.build_corpus`` verbatim — the same code
    HotpotQA runs through — so nothing about how paragraphs become a corpus can
    differ between the two datasets.
    """
    from benchmarks.public.run_hotpotqa import build_corpus
    from benchmarks.run_benchmark import DatasetError

    if questions <= 0:
        raise DatasetError("--questions must be >= 1; an unbounded run is never allowed")
    try:
        sample = load_sample(questions, seed=seed, path=data_path, allow_download=allow_download)
    except (OSError, ValueError, RuntimeError) as e:
        raise DatasetError(f"could not load 2WikiMultihopQA: {e}") from e

    if len(sample.questions) < questions:
        print(
            f"⚠️  Asked for {questions} questions, the pool yielded "
            f"{len(sample.questions)} (pool size {sample.pool_size})."
        )
    return build_corpus(sample.questions, sample=sample)


def _rebrand(lines: list[str]) -> list[str]:
    """Rename the dataset noun in reused report prose. The NUMBERS are untouched —
    only the four-letter dataset name the shared helpers hard-code in their text."""
    return [line.replace("HotpotQA", "2WikiMultihopQA") for line in lines]


def _type_counts(corpus) -> dict[str, int]:
    """Reasoning-type counts read straight off the sample (no bridge/comparison
    scaffolding — ``Corpus.counts_by_type`` assumes HotpotQA's two types)."""
    counts = Counter(q.hop_type for q in corpus.questions)
    ordered = {t: counts[t] for t in HOP_TYPES if t in counts}
    ordered.update({t: c for t, c in sorted(counts.items()) if t not in HOP_TYPES})
    return ordered


def threats(report) -> list[str]:
    """Threats to validity for 2Wiki — the ``run_hotpotqa`` structure with 2Wiki's
    own numbers, and the one genuinely different quirk (variable gold count) made
    explicit. Generated, so the figures in it cannot drift from the table."""
    from benchmarks.public import cost
    from benchmarks.public.run_hotpotqa import (
        DEFAULT_QUESTIONS,
        LOUD_QUESTION_THRESHOLD,
        effect_floor,
    )

    corpus = report.corpus
    n = report.chunked.agg["n"]
    notes = corpus.notes
    counts = _type_counts(corpus)
    four_gold = notes.get("questions_without_two_gold") or []
    items = [
        f"SAMPLE SIZE. {n} questions ({', '.join(f'{k}={v}' for k, v in counts.items())}) out "
        f"of 2WikiMultihopQA's {EXPECTED_DEV_RECORDS:,}-question dev set. One question is worth "
        f"{effect_floor(n) * 100:.1f} pp, which is why gaps below that are refused rather than "
        "reported. No statistical test is run and no significance is claimed: this is a sample "
        "of dozens, chosen to bound cost, not to support inference.",
        "RETRIEVAL ONLY. Nothing here measures answer quality. 2Wiki's headline numbers are "
        "answer EM/F1; these are its *retrieval* metrics (gold-paragraph recall, all-gold rate), "
        "comparable to published retrieval numbers, not to leaderboard EM/F1.",
        f"POOL SIZE. Row A searches all {len(corpus.paragraphs)} sampled paragraphs, because "
        "that is the pool Synapse's graph was built over; 2Wiki gives each question 10 candidate "
        "paragraphs. Row A_d is that per-question setting, reported separately. Comparing A_d to "
        "A measures the pool, not the retriever.",
        "NON-DETERMINISTIC GRAPH. The graph is built by an LLM at temperature "
        f"{report.settings_summary.get('extraction_temperature', 'n/a')}; a re-ingest of the "
        "same paragraphs can produce a different graph, and therefore different C/D rows. The "
        "passage rows are deterministic. --reuse-graph scores the same graph twice; re-ingesting "
        "does not.",
        "VARIABLE GOLD COUNT. 2Wiki bridge_comparison questions carry FOUR gold paragraphs; the "
        "other three types carry two. 'Both-gold' therefore means 'all-gold' — every gold "
        f"paragraph retrieved — and is strictly harder on the {len(four_gold)} four-gold "
        "question(s) in this sample, especially for the top-2 passage baseline, which can cover "
        "at most two of four. Same rule for every system; counted, not assumed away.",
        "PROVENANCE IS WEAKER EVIDENCE THAN PROSE. Under the permissive rule a gold paragraph is "
        "credited to the graph rows when an entity extracted from it appears in the context — "
        "which is not the same as the model being able to read that paragraph. That is exactly "
        "why the strict rule is reported alongside, why C scores 0.0 under it by construction, "
        "and why the channel split is printed.",
        "PRICES ARE HAND-RECORDED. Every USD figure is an estimate from a constant table checked "
        f"on {cost.PRICES_CHECKED_ON}, not from an invoice.",
        "SENTENCE MATCHING IS LEXICAL. Supporting-fact recall is whitespace-normalised substring "
        "matching over retrieved prose. It cannot see a paraphrase, and it credits a sentence "
        "that arrived inside a paragraph retrieved for some other reason. Same rule for every "
        "system.",
        "ENTITY RESOLUTION CROSSES PARAGRAPHS. Merging an entity of the same name from a gold "
        "paragraph and from a distractor is the product behaving correctly, but it means one "
        "entity can carry provenance for several paragraphs. It is measured nowhere here and can "
        "help or hurt the graph rows.",
        f"DATASET PROVENANCE. The bytes scored came from {notes.get('source', 'unknown')}; the "
        f"sample is {n} of a {notes.get('pool_size', 'n/a')}-question pool, taken by sorting on "
        "id and shuffling with the printed seed. The datasets-server rows API was down for this "
        "dataset on 2026-07-26, so the canonical dev.json was used (see twowiki.py).",
    ]
    if n > LOUD_QUESTION_THRESHOLD:
        items.insert(
            0,
            f"OVERSIZED RUN. {n} questions is above the {LOUD_QUESTION_THRESHOLD} this harness "
            f"treats as its ceiling — one extraction call per paragraph, on the author's own "
            f"card. The committed default is {DEFAULT_QUESTIONS}; this run was asked for "
            "explicitly.",
        )
    if notes.get("skipped_records"):
        items.append(
            f"SKIPPED RECORDS. {notes['skipped_records']} record(s) in the dataset file were "
            "malformed (see benchmarks/public/twowiki.py) and never entered the sampling pool."
        )
    if notes.get("dangling_supporting_facts"):
        items.append(
            f"DROPPED SUPPORTING FACTS. {notes['dangling_supporting_facts']} supporting fact(s) "
            "pointed past the end of their paragraph and were removed from the denominator, "
            "because no system could ever retrieve them."
        )
    if notes.get("paragraph_text_conflicts"):
        items.append(
            f"TITLE COLLISIONS. {len(notes['paragraph_text_conflicts'])} title(s) appeared in "
            "two questions with different text; the first was kept, since the graph holds one "
            "document per name."
        )
    if report.reused_graph:
        items.append(
            "REUSED GRAPH. This run did not ingest; it scored the graph already in Neo4j after "
            "verifying it holds every sampled paragraph. It cannot verify that the graph was "
            "built from this code at this configuration."
        )
    return ["Threats to validity — the reasons not to over-read anything above:"] + [
        f"  {i}. {item}" for i, item in enumerate(items, start=1)
    ]


def bridge_vs_comparison(report) -> list[str]:
    """The paper's bridge-vs-not split, computed with the SAME aggregate/verdict
    code, over 2Wiki's own taxonomy: comparison (both entities named) against the
    three bridge-like types (a hop hidden behind the first entity)."""
    from benchmarks.public.run_hotpotqa import (
        METRICS,
        aggregate,
        effect_floor,
        metric_verdict,
    )

    def group(results, predicate):
        return [r for r in results if predicate(r.qtype)]

    is_bridge = lambda t: t in BRIDGE_LIKE_TYPES  # noqa: E731
    lines = [
        "Bridge-like vs comparison (derived from 2Wiki's four types with the same effect-size "
        "floor): 'comparison' names both entities up front; 'inference', 'compositional' and "
        "'bridge_comparison' each hide a second entity behind a bridge, where a graph edge "
        "should help.",
    ]
    for name, predicate in (
        ("bridge-like (inference + compositional + bridge_comparison)", is_bridge),
        ("comparison", lambda t: t not in BRIDGE_LIKE_TYPES),
    ):
        base = aggregate(group(report.baseline.results, predicate))
        system = aggregate(group(report.chunked.results, predicate))
        graph = aggregate(group(report.graph.results, predicate))
        n = system["n"]
        lines.append(f"  {name} (N={n}, floor = {effect_floor(n) * 100:.1f} pp):")
        for label, key in METRICS:
            lines.append(
                "    " + metric_verdict(label, key, base, system, baseline_label="A", system_label="D")
            )
        lines.append(
            f"    Graph-only C here: {graph['both_gold']:.1%} all-gold permissive / "
            f"{graph['strict_both_gold']:.1%} strict."
        )
    return lines


def _sweep_table(sweep, total_paragraphs: int) -> list[str]:
    lines = [
        "| top-k | Gold-paragraph recall | All-gold rate | Supporting-fact recall | "
        "Mean chars | Share of pool |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k, agg in sweep:
        lines.append(
            f"| {k} | {agg['gold_recall']:.1%} | {agg['both_gold']:.1%} | "
            f"{agg['support_recall']:.1%} | {agg['mean_context_chars']:,.0f} | "
            f"{k / max(1, total_paragraphs):.0%} |"
        )
    return lines


def results_markdown(report) -> str:
    """The 2Wiki report. Every NUMBER comes from ``run_hotpotqa`` helpers; only the
    dataset-facing prose (title, licence, threats, taxonomy note) is written here."""
    from benchmarks.public.run_hotpotqa import summary_block
    from benchmarks.run_benchmark import _markdown_lines as markdown_lines

    corpus = report.corpus
    counts = _type_counts(corpus)
    n_four_gold = len(corpus.notes.get("questions_without_two_gold") or [])
    lines = [
        "# Synapse on 2WikiMultihopQA — replicating the scoring-bias finding",
        "",
        f"`2wikimultihopqa_dev` — {report.chunked.agg['n']} sampled questions "
        f"({', '.join(f'{k} {v}' for k, v in counts.items())}), "
        f"{len(corpus.paragraphs)} paragraphs, {corpus.total_chars:,} characters. "
        f"Dataset: {TWOWIKI_LICENSE}.",
        "",
        "This is the second public multi-hop dataset for the argument in "
        "`results_hotpotqa.md`: on HotpotQA the graph-only system C scored ~100% Both-gold "
        "under the permissive (containment) rule and 0.0% under the strict (prose) rule, with "
        "the overwhelming majority of its credited hits coming from an entity NAME appearing in "
        "the context rather than any retrieved evidence. One dataset is an anecdote; this file "
        "asks whether the same split appears on 2WikiMultihopQA (Ho et al., COLING 2020) — a "
        "recognised multi-hop benchmark whose corpus and questions come from outside this "
        "project. **The numbers below are computed by the identical harness** "
        "(`run_hotpotqa.py`): same permissive/strict rules, same channel attribution, same "
        "effect-size floor, same bridge-vs-not machinery. Nothing here is re-implemented.",
        "",
        "This measures **retrieval**, not answer generation. 2Wiki's four reasoning types are "
        "`comparison` (both entities named up front), `inference` and `compositional` (a second "
        "entity hidden behind a bridge), and `bridge_comparison` (both, and it carries **four** "
        f"gold paragraphs, not two — {n_four_gold} of this sample). 'Both-gold' therefore reads "
        "as 'all-gold': every gold paragraph retrieved. This file is written by the harness, "
        "never edited by hand.",
        "",
    ]
    lines += _rebrand(summary_block(report))

    body = report.analysis()
    n_head = len(report.headlines())
    ti = next((i for i, ln in enumerate(body) if ln.startswith("Threats to validity")), len(body))
    support = body[n_head + 1 : ti]
    while support and not support[-1].strip():
        support.pop()

    lines += ["", "## What the numbers support", ""]
    lines += markdown_lines(_rebrand(support))
    lines += [""]
    lines += markdown_lines(bridge_vs_comparison(report))
    lines += [""]
    lines += markdown_lines(threats(report))
    lines += [
        "",
        "## k sweep — pooled passage baseline (A)",
        "",
        "How the graph-free baseline behaves as it is allowed to read more, to the point where "
        "it has read every paragraph in the pool — so a budget match is always reachable.",
        "",
    ]
    lines += _sweep_table(report.sweep, len(corpus.paragraphs))
    lines += [
        "",
        "## Per-question breakdown",
        "",
        "| # | Type | Question | Gold paragraphs | A | A_d | C | D | D strict | D missed |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for base, dist, graph, chunked in zip(
        report.baseline.results,
        report.distractor.results,
        report.graph.results,
        report.chunked.results,
        strict=True,
    ):
        lines.append(
            f"| {chunked.qid[:8]} | {chunked.qtype} | {chunked.question} | "
            f"{', '.join(chunked.gold)} | {base.gold_recall:.0%} | {dist.gold_recall:.0%} | "
            f"{graph.gold_recall:.0%} | {chunked.gold_recall:.0%} | "
            f"{chunked.strict_gold_recall:.0%} | {', '.join(chunked.gold_missing) or '—'} |"
        )
    lines += [
        "",
        "---",
        "",
        "Generated by `backend/benchmarks/public/twowiki.py`, scored by "
        "`run_hotpotqa.py` — Synapse, © 2026 Ahmed Maaloul, AGPL-3.0-or-later. "
        f"2WikiMultihopQA is © its authors, {TWOWIKI_LICENSE}.",
        "",
    ]
    return "\n".join(lines)


def print_report(report) -> None:
    """A compact terminal view. The report file is the authoritative artefact."""
    from benchmarks.public.run_hotpotqa import _row

    counts = _type_counts(report.corpus)
    print("\n" + "═" * 88)
    print(
        f"  2WikiMultihopQA (dev) — {report.chunked.agg['n']} questions, "
        f"{len(report.corpus.paragraphs)} paragraphs, "
        + ", ".join(f"{k} {v}" for k, v in counts.items())
    )
    print("═" * 88)
    for label, agg in report.rows():
        print(_row(label, agg))
    print("  " + "─" * 84)
    for line in _rebrand(report.headlines()):
        print(f"  {line}" if line else "")


def build_run_parser() -> argparse.ArgumentParser:
    from benchmarks.public.run_hotpotqa import (
        DEFAULT_BASELINE_K,
        DEFAULT_GRAPH_K,
        DEFAULT_MAX_USD,
        DEFAULT_QUESTIONS,
        DEFAULT_THEME,
    )

    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.public.twowiki run",
        description="Synapse GraphRAG vs. passage vector RAG on 2WikiMultihopQA — the HotpotQA "
        "harness, a second dataset. Spends real money on ingestion — always --dry-run first.",
    )
    parser.add_argument("--questions", type=int, default=DEFAULT_QUESTIONS,
                        help=f"HARD CAP on sampled questions (default: {DEFAULT_QUESTIONS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"sampling seed (default: {DEFAULT_SEED})")
    parser.add_argument("--reuse-graph", action="store_true",
                        help="skip ingestion and score the graph already in Neo4j — free re-runs")
    parser.add_argument("--dry-run", action="store_true",
                        help="estimate the cost and exit WITHOUT calling any model (free)")
    parser.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD,
                        help=f"abort ingestion once metered spend passes this (default: {DEFAULT_MAX_USD})")
    parser.add_argument("--baseline-k", type=int, default=DEFAULT_BASELINE_K,
                        help=f"passages the vector baseline retrieves (default: {DEFAULT_BASELINE_K})")
    parser.add_argument("--graph-k", type=int, default=DEFAULT_GRAPH_K,
                        help=f"seed entities for GraphRAG (default: {DEFAULT_GRAPH_K})")
    parser.add_argument("--theme", default=DEFAULT_THEME, help=f"extraction theme (default: {DEFAULT_THEME})")
    parser.add_argument("--ingest-concurrency", type=int, default=1,
                        help="documents ingested in parallel (default: 1 — deterministic)")
    parser.add_argument("--data", type=Path, default=None,
                        help="path to a 2Wiki dev json (default: the cache, downloading once)")
    parser.add_argument("--no-download", action="store_true",
                        help="never reach the network; fail if the dataset is not already cached")
    parser.add_argument("--model", default="",
                        help="price the run as this model instead of the configured one (estimates only)")
    parser.add_argument("--out", type=Path, default=RESULTS_PATH,
                        help=f"markdown report path (default: {RESULTS_PATH})")
    parser.add_argument("--no-write", action="store_true", help="do not write the markdown report")
    return parser


async def run_main(argv: Sequence[str] | None = None) -> int:
    """Benchmark entry point. Reuses ``run_hotpotqa`` for cost, ingestion, scoring
    and analysis; only dataset loading and report prose are 2Wiki's own."""
    import logging

    from benchmarks.public import cost
    from benchmarks.public.run_hotpotqa import (
        BudgetExceeded,
        ReuseError,
        _run_systems,
        chat_model_name,
        dry_run_lines,
        estimate_ingestion,
        loud_banner,
    )
    from benchmarks.run_benchmark import DatasetError, IntegrityError

    args = build_run_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    for line in loud_banner(args.questions):
        print(line)

    try:
        corpus = load_corpus(
            questions=args.questions, seed=args.seed,
            data_path=args.data, allow_download=not args.no_download,
        )
    except DatasetError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    counts = _type_counts(corpus)
    print(
        f"📚 2WikiMultihopQA dev: {len(corpus.questions)} questions "
        f"({', '.join(f'{k}={v}' for k, v in counts.items())}), "
        f"{len(corpus.paragraphs)} unique paragraphs, seed {args.seed}"
    )

    if args.dry_run:
        from app.config import get_settings

        model = args.model or chat_model_name(get_settings())
        for line in dry_run_lines(corpus, model=model, theme=args.theme, max_usd=args.max_usd):
            print(line)
        print(
            "\n  To run it for real:\n"
            "    cd backend && EMBEDDING_PROVIDER=fastembed NEO4J_URI=bolt://localhost:7688 \\\n"
            "        NEO4J_PASSWORD=research_secret python -m benchmarks.public.twowiki run "
            f"--questions {args.questions} --seed {args.seed}"
        )
        return 0

    from app import neo4j_driver
    from app.config import get_settings

    settings = get_settings()
    if not await neo4j_driver.verify_connectivity():
        print(
            f"❌ Cannot reach Neo4j at {settings.neo4j_uri}.\n"
            "   Start it first:  docker compose up -d neo4j\n"
            "   (override with NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD)",
            file=sys.stderr,
        )
        return 2

    if not args.reuse_graph:
        projected = cost.usd(
            estimate_ingestion(corpus, theme=args.theme),
            cost.resolve_price(args.model or chat_model_name(settings)),
        )
        if projected is not None and projected > args.max_usd:
            print(
                f"❌ Estimated ingestion cost {cost.format_usd(projected)} exceeds --max-usd "
                f"{cost.format_usd(args.max_usd)}. Lower --questions, or raise the cap "
                "deliberately.",
                file=sys.stderr,
            )
            await neo4j_driver.close_driver()
            return 5
        print(
            f"⚠️  Ingestion REPLACES the :Entity/:Community/:Chunk data in {settings.neo4j_uri}. "
            f"Estimated cost {cost.format_usd(projected)}, cap {cost.format_usd(args.max_usd)}."
        )

    ledgers: list[cost.CostLedger] = []
    try:
        report = await _run_systems(corpus, args, ledgers)
    except ReuseError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 4
    except BudgetExceeded as e:
        print(f"⛔ {e}", file=sys.stderr)
        return 5
    except IntegrityError as e:
        print(f"❌ Refusing to report this run: {e}", file=sys.stderr)
        return 3
    finally:
        await neo4j_driver.close_driver()

    report.settings_line = (
        f"`--questions {args.questions} --seed {args.seed}` · "
        f"{len(report.corpus.paragraphs)} paragraphs · embeddings "
        f"`{settings.embedding_provider}` · extraction `{report.settings_summary['model']}` "
        f"(`{settings.llm_provider}`, temperature {settings.extraction_temperature}) · "
        f"graph seeds k={args.graph_k} · baseline top-{args.baseline_k} · "
        f"`chunk_top_k={settings.chunk_top_k}` · "
        f"`chunk_context_max_chars={settings.chunk_context_max_chars}` · "
        f"`retrieval_max_hops={settings.retrieval_max_hops}`. Reproduce: "
        f"`python -m benchmarks.public.twowiki run --questions {args.questions} "
        f"--seed {args.seed} --reuse-graph`"
    )

    print_report(report)

    if not args.no_write:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(results_markdown(report), encoding="utf-8")
        except OSError as e:
            print(f"❌ Could not write {args.out}: {e}", file=sys.stderr)
            return 3
        print(f"📄 Wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    _argv = sys.argv[1:]
    if _argv and _argv[0] == "run":
        raise SystemExit(asyncio.run(run_main(_argv[1:])))
    raise SystemExit(main(_argv))
