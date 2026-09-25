# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse Lab — the OpenAI Batch API runner: half price, 24-hour window, resumable.

The Batch API takes a JSONL file of ``chat.completions`` requests, runs it
within 24 hours and bills it at HALF the realtime rate
(``cost.BATCH_PRICE_MULTIPLIER``). The Lab sends both of its paid phases this
way: the reader (``runner.py``) and the corpus extraction (``ingest.py``).

LIFECYCLE (one ``run_dir``; its state lives in ``run_dir/batches.json``)

  submit   — validate the request lines (the OpenAI Batch INPUT format:
             ``{custom_id, method: "POST", url, body}``), split them into
             PARTS (one model per part; at most 50,000 requests and 200 MB per
             part — the limits the installed ``openai`` SDK documents for
             ``batches.create``), write each part to ``run_dir/batches/``,
             upload it (``purpose="batch"``), create its batch
             (``/v1/chat/completions``, ``completion_window="24h"``, metadata
             = the run id). ``batches.json`` is written BEFORE every network
             step and after it, so a crash anywhere is recoverable.
  poll     — refresh each open batch: status + request counts; submit any
             part still queued behind the enqueued-token gate (below).
  collect  — once every open batch is terminal: download the output and
             error files (kept under ``run_dir/batches/``), return one record
             per request in input order, and mark those parts collected.
             A request that never ran (the batch expired, was cancelled or
             failed validation) comes back as a synthesized error record, so
             nothing silently disappears.
  failed_requests / resubmit_failed
           — the request lines whose latest result is an error, and a new
             submission of exactly those (the partial-failure path).
  parse_results / latest_records / status / cancel
           — answers + usage per custom_id, the latest result per request
             across retries, a local (network-free) summary with the spend at
             batch prices, and cancellation of what is still open.

The runner reaches ``submit`` / ``poll`` / ``collect`` through
``runner.ModuleBatchBackend`` (``client=`` is passed through).

IDEMPOTENT. Submitting the same lines again while their part is still open
reuses it (matched by the sha256 of the part's bytes) — a crash between
creating a batch and recording it upstream can never double-bill. A request
already in flight in a DIFFERENT open part is refused rather than sent twice.
A part whose batch creation was attempted but not recorded is first looked up
among the account's recent batches by its input file id.

ENQUEUED-TOKEN GATE (optional). OpenAI caps the input tokens an organisation
may have enqueued per model (tier-dependent — see the organisation's limits
page). Set ``max_enqueued_tokens`` (or ``SYNAPSE_LAB_BATCH_MAX_ENQUEUED_TOKENS``)
and parts are sized under it and submitted one wave at a time: ``poll`` sends
the next part when the in-flight ones finish. Unset = no gating.

COST. Every collected part records its usage and its $ at the batch rate
(``cost.usd(..., batch=True)``). A request that comes back as an error carries
no usage and is priced at $0 here; the runner reads it as an error
(``done_with_errors``) and can resend it.

NEVER call this against the real API from a test: without an explicit
``client`` it builds :func:`app.lab.reader.default_client`, which refuses to
exist under pytest.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.lab import reader
from benchmarks.public import cost

logger = logging.getLogger(__name__)

ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"
STATE_FILE = "batches.json"
PARTS_DIR = "batches"
STATE_VERSION = 1

#: Per-batch limits as documented by the installed ``openai`` SDK (1.83,
#: ``batches.create``): "The file can contain up to 50,000 requests, and can be
#: up to 200 MB in size." 200 MB is read as 200,000,000 bytes — the stricter
#: of the two readings of "MB".
MAX_REQUESTS_PER_BATCH = 50_000
MAX_BATCH_FILE_BYTES = 200_000_000
#: Environment override for the per-model enqueued-token gate (unset = no gate).
ENQUEUED_TOKENS_ENV = "SYNAPSE_LAB_BATCH_MAX_ENQUEUED_TOKENS"
#: Metadata values are capped at 512 characters by the API.
_METADATA_VALUE_MAX = 512

#: Local part states (before OpenAI has a batch for the part).
LOCAL = "local"
UPLOADED = "uploaded"
#: OpenAI batch states.
TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})
IN_FLIGHT = frozenset({"validating", "in_progress", "finalizing", "cancelling"})


class BatchError(RuntimeError):
    """A batch operation that cannot proceed (bad input, duplicate in flight, …)."""


# ── Small helpers ────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute or key access — SDK objects and plain dicts (fakes) alike."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def encode_line(request: dict) -> bytes:
    """One request as the exact JSONL bytes uploaded (compact, UTF-8, newline)."""
    return (json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _read_jsonl_bytes(data: bytes) -> tuple[list[dict], int]:
    """``(records, malformed line count)`` of a JSONL payload.

    Split on ``\n`` ONLY: ``str.splitlines`` also breaks on U+2028/U+2029/NEL,
    which ``json.dumps(ensure_ascii=False)`` leaves raw inside strings — a
    Wikipedia paragraph holding one would otherwise tear its line in two.
    """
    records: list[dict] = []
    bad = 0
    for raw in data.decode("utf-8", errors="replace").split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            bad += 1
    return records, bad


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records, bad = _read_jsonl_bytes(path.read_bytes())
    if bad:
        logger.warning("skipped %s malformed line(s) in %s", bad, path.name)
    return records


# ── State ────────────────────────────────────────────────────────────────────
def state_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / STATE_FILE


def load_state(run_dir: Path | str) -> dict[str, Any]:
    """``batches.json`` (an empty state when there is none yet)."""
    path = state_path(run_dir)
    if not path.exists():
        return {"version": STATE_VERSION, "endpoint": ENDPOINT, "submissions": 0, "parts": []}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise BatchError(f"{path} is unreadable ({e}); it is the only record of the batches") from e
    if not isinstance(state, dict) or not isinstance(state.get("parts"), list):
        raise BatchError(f"{path} is not a Lab batch state file")
    return state


def _save_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    blob = json.dumps(state, indent=2, ensure_ascii=False, default=str) + "\n"
    _write_bytes_atomic(state_path(run_dir), blob.encode("utf-8"))


def _part_path(run_dir: Path, part: dict, kind: str) -> Path:
    return Path(run_dir) / PARTS_DIR / f"{part['name']}.{kind}.jsonl"


def _is_open(part: dict) -> bool:
    return not part.get("collected_at")


def _is_terminal(part: dict) -> bool:
    return part.get("status") in TERMINAL


def _enqueued_limit(explicit: int | None) -> int | None:
    if explicit is not None:
        return int(explicit) if int(explicit) > 0 else None
    raw = os.environ.get(ENQUEUED_TOKENS_ENV, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("ignoring %s=%r (not an integer)", ENQUEUED_TOKENS_ENV, raw)
        return None
    return value if value > 0 else None


# ── Input validation and splitting (pure) ────────────────────────────────────
def validate_requests(requests: Iterable[dict]) -> list[dict]:
    """The request lines, checked: unique ``custom_id``, POST, one endpoint, a body with a model."""
    lines = list(requests)
    seen: set[str] = set()
    for i, line in enumerate(lines):
        if not isinstance(line, dict):
            raise BatchError(f"request {i} is not an object")
        custom_id = line.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise BatchError(f"request {i} has no custom_id")
        if custom_id in seen:
            raise BatchError(f"duplicate custom_id {custom_id!r} — outputs could not be matched")
        seen.add(custom_id)
        if line.get("method") != "POST":
            raise BatchError(f"{custom_id}: method must be POST")
        if line.get("url") != ENDPOINT:
            raise BatchError(f"{custom_id}: url must be {ENDPOINT} (got {line.get('url')!r})")
        body = line.get("body")
        if not isinstance(body, dict) or not body.get("model"):
            raise BatchError(f"{custom_id}: body must be an object naming a model")
    return lines


def request_prompt_tokens(line: dict) -> int:
    """Real-tokenizer prompt tokens of one request (what the enqueued-token limit counts)."""
    body = line.get("body") or {}
    return reader.request_prompt_tokens(body, str(body.get("model") or ""))[0]


def plan_parts(
    lines: Sequence[dict],
    *,
    max_requests: int = MAX_REQUESTS_PER_BATCH,
    max_bytes: int = MAX_BATCH_FILE_BYTES,
    max_tokens: int | None = None,
    tokens_of: Callable[[dict], int] = request_prompt_tokens,
) -> list[list[dict]]:
    """Split request lines into parts: one model per part, each under every limit.

    Order is preserved within a model; models appear in first-seen order. A
    single request larger than ``max_bytes`` cannot be sent at all and is
    refused. ``max_tokens`` (the enqueued-token gate) sizes parts so each can
    be enqueued on its own; a single request above it gets a part of its own.
    Tokens are only counted (``tokens_of``) when that gate is set.
    """
    if max_requests <= 0 or max_bytes <= 0:
        raise BatchError("batch limits must be positive")
    by_model: dict[str, list[dict]] = {}
    for line in lines:
        by_model.setdefault(str(line["body"]["model"]), []).append(line)
    parts: list[list[dict]] = []
    for model_lines in by_model.values():
        current: list[dict] = []
        size = 0
        tokens = 0
        for line in model_lines:
            n_bytes = len(encode_line(line))
            if n_bytes > max_bytes:
                raise BatchError(
                    f"{line['custom_id']}: one request is {n_bytes:,} bytes, above the "
                    f"{max_bytes:,}-byte batch file limit"
                )
            n_tokens = tokens_of(line) if max_tokens else 0
            if current and (
                len(current) >= max_requests
                or size + n_bytes > max_bytes
                or (max_tokens and tokens + n_tokens > max_tokens)
            ):
                parts.append(current)
                current, size, tokens = [], 0, 0
            current.append(line)
            size += n_bytes
            tokens += n_tokens
        if current:
            parts.append(current)
    return parts


# ── Client ───────────────────────────────────────────────────────────────────
def _client(client: Any) -> Any:
    return client if client is not None else reader.default_client()


async def _content_bytes(content: Any) -> bytes:
    """The bytes of a ``files.content`` result (SDK binary response, bytes or str)."""
    if isinstance(content, bytes | bytearray):
        return bytes(content)
    if isinstance(content, str):
        return content.encode("utf-8")
    raw = getattr(content, "content", None)
    if isinstance(raw, bytes | bytearray):
        return bytes(raw)
    read = getattr(content, "read", None)
    if callable(read):
        data = await _maybe_await(read())
        if isinstance(data, bytes | bytearray):
            return bytes(data)
        if isinstance(data, str):
            return data.encode("utf-8")
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text.encode("utf-8")
    raise BatchError(f"cannot read a batch file from a {type(content).__name__}")


def _apply_batch(part: dict, batch: Any) -> None:
    """Copy what OpenAI reports about a batch onto its part record."""
    part["batch_id"] = _get(batch, "id") or part.get("batch_id")
    part["status"] = str(_get(batch, "status") or part.get("status") or "")
    counts = _get(batch, "request_counts")
    if counts is not None:
        part["request_counts"] = {
            "total": int(_get(counts, "total") or 0),
            "completed": int(_get(counts, "completed") or 0),
            "failed": int(_get(counts, "failed") or 0),
        }
    for key in ("output_file_id", "error_file_id"):
        value = _get(batch, key)
        if value:
            part[key] = value
    errors = _get(_get(batch, "errors"), "data") or []
    if errors:
        part["batch_errors"] = [
            {"code": _get(e, "code"), "message": _get(e, "message"), "line": _get(e, "line")}
            for e in errors
        ]
    for key in ("created_at", "in_progress_at", "finalizing_at", "completed_at", "failed_at",
                "expired_at", "expires_at", "cancelled_at"):
        value = _get(batch, key)
        if value is not None:
            part.setdefault("openai_times", {})[key] = value


# ── Submit ───────────────────────────────────────────────────────────────────
def _open_custom_ids(run_dir: Path, state: dict[str, Any]) -> dict[str, str]:
    """custom_id → name of the open part that holds it."""
    owners: dict[str, str] = {}
    for part in state["parts"]:
        if not _is_open(part):
            continue
        for record in _read_jsonl(_part_path(run_dir, part, "input")):
            if record.get("custom_id"):
                owners[str(record["custom_id"])] = part["name"]
    return owners


def _metadata(run_id: str, part: dict, extra: dict[str, str] | None) -> dict[str, str]:
    data = {
        "synapse_run": str(run_id),
        "synapse_part": part["name"],
        "synapse_sha256": part["sha256"],
    }
    for key, value in (extra or {}).items():
        data[str(key)[:64]] = str(value)
    if len(data) > 16:
        raise BatchError("batch metadata holds at most 16 keys")
    return {k: v[:_METADATA_VALUE_MAX] for k, v in data.items()}


async def _find_created_batch(client: Any, input_file_id: str) -> Any | None:
    """A batch already created from ``input_file_id`` (crash recovery), or ``None``."""
    lister = getattr(getattr(client, "batches", None), "list", None)
    if not callable(lister):
        return None
    try:
        page = await _maybe_await(lister(limit=100))
    except Exception as e:  # noqa: BLE001 - recovery is best-effort
        logger.warning("could not list recent batches for crash recovery: %s", e)
        return None
    for batch in _get(page, "data") or []:
        if _get(batch, "input_file_id") == input_file_id:
            return batch
    return None


def _in_flight_tokens(state: dict[str, Any], model: str) -> int:
    return sum(
        int(p.get("prompt_tokens") or 0)
        for p in state["parts"]
        if p.get("model") == model and p.get("status") in IN_FLIGHT
    )


async def _advance(run_dir: Path, state: dict[str, Any], client: Any) -> Any:
    """Upload and create every part that is still local, as the token gate allows.

    Returns the client it used (created lazily, only when something is sent).
    """
    limit = state.get("max_enqueued_tokens")
    for part in state["parts"]:
        if part.get("status") not in (LOCAL, UPLOADED):
            continue
        if limit:
            enqueued = _in_flight_tokens(state, part["model"])
            if enqueued and enqueued + int(part.get("prompt_tokens") or 0) > int(limit):
                part["queued_reason"] = (
                    f"waiting: {enqueued:,} tokens already enqueued for {part['model']} "
                    f"(gate {int(limit):,})"
                )
                continue
        client = _client(client)
        part.pop("queued_reason", None)
        if part["status"] == LOCAL:
            data = _part_path(run_dir, part, "input").read_bytes()
            if hashlib.sha256(data).hexdigest() != part["sha256"]:
                raise BatchError(f"{part['name']}: the local input file changed since it was planned")
            uploaded = await _maybe_await(
                client.files.create(file=(f"{part['name']}.jsonl", data), purpose="batch")
            )
            part["input_file_id"] = _get(uploaded, "id")
            part["status"] = UPLOADED
            part["uploaded_at"] = _now()
            _save_state(run_dir, state)
        batch = None
        if part.get("create_attempted_at"):
            batch = await _find_created_batch(client, part["input_file_id"])
            if batch is not None:
                logger.info("%s: recovered batch %s", part["name"], _get(batch, "id"))
        if batch is None:
            part["create_attempted_at"] = _now()
            _save_state(run_dir, state)
            batch = await _maybe_await(
                client.batches.create(
                    input_file_id=part["input_file_id"],
                    endpoint=ENDPOINT,
                    completion_window=COMPLETION_WINDOW,
                    metadata=_metadata(state.get("run_id") or "", part, state.get("metadata")),
                )
            )
        _apply_batch(part, batch)
        part["submitted_at"] = part.get("submitted_at") or _now()
        _save_state(run_dir, state)
        logger.info("%s: batch %s (%s requests) %s", part["name"], part["batch_id"],
                    part["requests"], part["status"])
    _save_state(run_dir, state)  # records every queued part's reason
    return client


def _part_summary(part: dict) -> dict[str, Any]:
    return {
        "part": part["name"],
        "batch_id": part.get("batch_id"),
        "model": part.get("model"),
        "status": part.get("status"),
        "requests": part.get("requests"),
        "request_counts": part.get("request_counts"),
        "submission": part.get("submission"),
        "collected": bool(part.get("collected_at")),
        **({"queued_reason": part["queued_reason"]} if part.get("queued_reason") else {}),
        **({"batch_errors": part["batch_errors"]} if part.get("batch_errors") else {}),
    }


async def submit(
    run_dir: Path | str,
    requests: Sequence[dict],
    *,
    run_id: str,
    client: Any = None,
    metadata: dict[str, str] | None = None,
    max_requests: int = MAX_REQUESTS_PER_BATCH,
    max_bytes: int = MAX_BATCH_FILE_BYTES,
    max_enqueued_tokens: int | None = None,
) -> dict[str, Any]:
    """Submit request lines as one or more batches. Idempotent while they are open.

    ``requests`` are OpenAI Batch input lines. Returns
    ``{"batch_ids", "parts", "requests", "queued", "submission"}``.
    """
    run_dir = Path(run_dir)
    lines = validate_requests(requests)
    state = load_state(run_dir)
    if not lines:
        return {"batch_ids": [], "parts": [], "requests": 0, "queued": 0,
                "submission": state.get("submissions", 0)}
    limit = _enqueued_limit(max_enqueued_tokens)
    if limit is not None:
        state["max_enqueued_tokens"] = limit
    state["run_id"] = str(run_id)
    if metadata:
        state["metadata"] = {str(k): str(v) for k, v in metadata.items()}

    gate = state.get("max_enqueued_tokens")
    token_cache: dict[str, int] = {}

    def tokens_of(line: dict) -> int:
        if line["custom_id"] not in token_cache:
            token_cache[line["custom_id"]] = request_prompt_tokens(line)
        return token_cache[line["custom_id"]]

    groups = plan_parts(lines, max_requests=max_requests, max_bytes=max_bytes,
                        max_tokens=gate, tokens_of=tokens_of)
    owners = _open_custom_ids(run_dir, state)
    by_sha = {p["sha256"]: p for p in state["parts"] if _is_open(p)}
    planned: list[tuple[list[dict], bytes, str]] = []
    for group in groups:
        data = b"".join(encode_line(line) for line in group)
        planned.append((group, data, hashlib.sha256(data).hexdigest()))
    reused = {sha for _g, _d, sha in planned if sha in by_sha}
    for group, _data, sha in planned:
        if sha in reused:
            continue
        clash = [line["custom_id"] for line in group if line["custom_id"] in owners]
        if clash:
            raise BatchError(
                f"{len(clash)} request(s) are already in flight in an open batch part "
                f"(e.g. {clash[0]!r} in {owners[clash[0]]}); poll and collect it first, "
                "or cancel it — the same request is never sent twice"
            )

    submission = int(state.get("submissions", 0)) + 1
    parts: list[dict] = []
    for group, data, sha in planned:
        if sha in reused:
            parts.append(by_sha[sha])
            continue
        part = {
            "name": f"part-{len(state['parts']):04d}",
            "submission": submission,
            "model": str(group[0]["body"]["model"]),
            "sha256": sha,
            "requests": len(group),
            "bytes": len(data),
            # Counted only for the enqueued-token gate (tokenising every request
            # is not free, and the runner has already priced them).
            "prompt_tokens": sum(tokens_of(line) for line in group) if gate else None,
            "status": LOCAL,
            "planned_at": _now(),
        }
        _write_bytes_atomic(_part_path(run_dir, part, "input"), data)
        state["parts"].append(part)
        parts.append(part)
    if any(p.get("submission") == submission for p in parts):
        state["submissions"] = submission
    _save_state(run_dir, state)

    await _advance(run_dir, state, client)
    return {
        "batch_ids": [p["batch_id"] for p in parts if p.get("batch_id")],
        "parts": [_part_summary(p) for p in parts],
        "requests": sum(int(p["requests"]) for p in parts),
        "queued": sum(1 for p in parts if p.get("status") in (LOCAL, UPLOADED)),
        "submission": state.get("submissions", 0),
        "reused": sum(1 for p in parts if p["sha256"] in reused),
    }


# ── Poll ─────────────────────────────────────────────────────────────────────
def _aggregate_status(open_parts: list[dict]) -> str:
    if not open_parts:
        return "empty"
    statuses = {p.get("status") for p in open_parts}
    if not all(s in TERMINAL for s in statuses):
        if statuses & {LOCAL, UPLOADED} and not statuses & IN_FLIGHT:
            return "queued"
        return "in_progress"
    if statuses == {"completed"}:
        return "completed"
    return "partial" if "completed" in statuses else sorted(statuses)[0]


def _counts(parts: list[dict]) -> dict[str, int]:
    total = {"total": 0, "completed": 0, "failed": 0}
    for part in parts:
        counts = part.get("request_counts") or {}
        total["total"] += int(counts.get("total") or part.get("requests") or 0)
        total["completed"] += int(counts.get("completed") or 0)
        total["failed"] += int(counts.get("failed") or 0)
    return total


async def poll(run_dir: Path | str, *, client: Any = None) -> dict[str, Any]:
    """Refresh every open batch; send queued parts the gate now allows.

    Returns ``{"done", "status", "parts", "request_counts", "batch_ids"}`` —
    ``done`` once every open part is terminal (completed, failed, expired or
    cancelled). Makes no network call when nothing is open.
    """
    run_dir = Path(run_dir)
    state = load_state(run_dir)
    open_parts = [p for p in state["parts"] if _is_open(p)]
    for part in open_parts:
        if part.get("batch_id") and not _is_terminal(part):
            client = _client(client)
            _apply_batch(part, await _maybe_await(client.batches.retrieve(part["batch_id"])))
    state["polled_at"] = _now()
    _save_state(run_dir, state)
    if any(p.get("status") in (LOCAL, UPLOADED) for p in open_parts):
        client = await _advance(run_dir, state, client)
    done = all(_is_terminal(p) for p in open_parts)
    return {
        "done": done,
        "status": _aggregate_status(open_parts),
        "parts": [_part_summary(p) for p in open_parts],
        "request_counts": _counts(open_parts),
        "batch_ids": [p["batch_id"] for p in open_parts if p.get("batch_id")],
        "polled_at": state["polled_at"],
    }


# ── Collect ──────────────────────────────────────────────────────────────────
def record_ok(record: dict) -> bool:
    """True when a Batch output line carries a successful completion."""
    if record.get("error"):
        return False
    response = record.get("response") or {}
    return int(response.get("status_code") or 0) == 200 and bool(response.get("body"))


def record_body(record: dict) -> dict | None:
    """The ``chat.completion`` body of a successful output line, else ``None``."""
    return (record.get("response") or {}).get("body") if record_ok(record) else None


def record_error(record: dict) -> str | None:
    """A one-line description of a failed output line, else ``None``."""
    if record_ok(record):
        return None
    response = record.get("response") or {}
    err = record.get("error") or (response.get("body") or {}).get("error")
    if not err:
        err = f"status {response.get('status_code') or 'missing'}"
    if isinstance(err, dict):
        code, message = err.get("code"), err.get("message")
        return f"{code}: {message}" if code and message else str(message or code or err)
    return str(err)


async def _download(run_dir: Path, part: dict, client: Any) -> Any:
    """Fetch the part's output and error files into ``run_dir/batches/`` (once)."""
    for key, kind in (("output_file_id", "output"), ("error_file_id", "errors")):
        file_id = part.get(key)
        path = _part_path(run_dir, part, kind)
        if not file_id or path.exists():
            continue
        client = _client(client)
        content = await _maybe_await(client.files.content(file_id))
        _write_bytes_atomic(path, await _content_bytes(content))
        part[f"{kind}_downloaded_at"] = _now()
    return client


def _part_records(run_dir: Path, part: dict) -> list[dict]:
    """One record per input request, in input order; a request with no result is an error."""
    expected = [str(r["custom_id"]) for r in _read_jsonl(_part_path(run_dir, part, "input"))
                if r.get("custom_id")]
    wanted = set(expected)
    got: dict[str, dict] = {}
    unknown = 0
    for kind in ("output", "errors"):
        for record in _read_jsonl(_part_path(run_dir, part, kind)):
            custom_id = str(record.get("custom_id") or "")
            if custom_id not in wanted:
                unknown += 1
                continue
            if custom_id in got and record_ok(got[custom_id]) and not record_ok(record):
                continue  # a success is never overwritten by an error line
            got[custom_id] = record
    if unknown:
        logger.warning("%s: ignored %s output line(s) with an unknown custom_id",
                       part["name"], unknown)
    status = part.get("status") or "unknown"
    reason = "; ".join(
        f"{e.get('code')}: {e.get('message')}" for e in part.get("batch_errors") or []
    )
    for custom_id in expected:
        if custom_id not in got:
            got[custom_id] = {
                "custom_id": custom_id,
                "response": None,
                "error": {
                    "code": f"batch_{status}",
                    "message": f"no result: batch {part.get('batch_id') or part['name']} ended "
                    f"'{status}' before this request ran" + (f" ({reason})" if reason else ""),
                },
                "synthesized": True,
            }
    return [got[c] for c in expected]


def _part_cost(part: dict, records: list[dict]) -> dict[str, Any]:
    usage = cost.Usage()
    for record in records:
        body = record_body(record)
        if body is None:
            continue
        result = reader.parse_completion(body)
        usage = usage + cost.Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            reasoning_tokens=result.reasoning_tokens,
            calls=1,
        )
    model = str(part.get("model") or "")
    return {
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "calls": usage.calls,
        },
        "usd": cost.usd(usage, cost.resolve_price(model), batch=True),
        "succeeded": sum(1 for r in records if record_ok(r)),
        "failed": sum(1 for r in records if not record_ok(r)),
    }


async def collect(
    run_dir: Path | str, *, client: Any = None, include_collected: bool = False
) -> list[dict]:
    """Every result of the finished, not-yet-collected parts — then mark them collected.

    Records are OpenAI Batch OUTPUT lines (``{custom_id, response: {status_code,
    body}, error}``), one per submitted request in input order; a request that
    never ran is a synthesized error line (``"synthesized": true``). Output
    files are kept locally, so ``include_collected=True`` re-reads earlier
    parts without a network call. Refuses while an open part is still running.
    """
    run_dir = Path(run_dir)
    state = load_state(run_dir)
    running = [p["name"] for p in state["parts"] if _is_open(p) and not _is_terminal(p)]
    if running:
        raise BatchError(
            f"{len(running)} batch part(s) are not finished ({', '.join(running[:5])}); "
            "poll until done before collecting"
        )
    records: list[dict] = []
    for part in state["parts"]:
        if not _is_open(part):
            if include_collected:
                records.extend(_part_records(run_dir, part))
            continue
        client = await _download(run_dir, part, client)
        part_records = _part_records(run_dir, part)
        part.update(_part_cost(part, part_records))
        part["collected_at"] = _now()
        _save_state(run_dir, state)
        records.extend(part_records)
    return records


def parse_results(records: Iterable[dict]) -> dict[str, reader.ReaderResult]:
    """custom_id → answer + usage (a :class:`reader.ReaderResult`; ``error`` set on failure)."""
    out: dict[str, reader.ReaderResult] = {}
    for record in records:
        body = record_body(record)
        if body is not None:
            result = reader.parse_completion(body)
        else:
            result = reader.ReaderResult(error=(record_error(record) or "failed")[:500])
        out[str(record.get("custom_id") or "")] = result
    return out


def latest_records(run_dir: Path | str) -> dict[str, dict]:
    """custom_id → its latest collected result, a success always beating an error.

    Local only (reads the downloaded files); parts are read in submission order,
    so a successful retry supersedes the error it retried.
    """
    run_dir = Path(run_dir)
    latest: dict[str, dict] = {}
    for part in load_state(run_dir)["parts"]:
        if _is_open(part):
            continue
        for record in _part_records(run_dir, part):
            custom_id = record["custom_id"]
            if custom_id in latest and record_ok(latest[custom_id]) and not record_ok(record):
                continue
            latest[custom_id] = record
    return latest


def failed_requests(run_dir: Path | str) -> list[dict]:
    """The request lines whose latest collected result is an error, ready to resubmit.

    A request already being retried in an open part is not listed again.
    """
    run_dir = Path(run_dir)
    state = load_state(run_dir)
    latest = latest_records(run_dir)
    in_flight = _open_custom_ids(run_dir, state)
    lines: dict[str, dict] = {}
    for part in state["parts"]:
        if _is_open(part):
            continue
        for line in _read_jsonl(_part_path(run_dir, part, "input")):
            custom_id = str(line.get("custom_id") or "")
            record = latest.get(custom_id)
            if record is not None and not record_ok(record) and custom_id not in in_flight:
                lines[custom_id] = line
    return list(lines.values())


async def resubmit_failed(
    run_dir: Path | str, *, run_id: str | None = None, client: Any = None
) -> dict[str, Any]:
    """Submit exactly the failed requests again, as a new batch submission."""
    run_dir = Path(run_dir)
    lines = failed_requests(run_dir)
    state = load_state(run_dir)
    return await submit(run_dir, lines, run_id=run_id or state.get("run_id") or "",
                        client=client)


async def cancel(run_dir: Path | str, *, client: Any = None) -> dict[str, Any]:
    """Cancel every open batch; parts never sent are cancelled locally (nothing billed)."""
    run_dir = Path(run_dir)
    state = load_state(run_dir)
    cancelled = []
    for part in state["parts"]:
        if not _is_open(part) or _is_terminal(part):
            continue
        if part.get("status") in (LOCAL, UPLOADED) and not part.get("create_attempted_at"):
            part["status"] = "cancelled"
            part["cancelled_locally_at"] = _now()
        elif part.get("batch_id"):
            client = _client(client)
            _apply_batch(part, await _maybe_await(client.batches.cancel(part["batch_id"])))
        else:
            continue  # creation attempted, outcome unknown: poll/submit recovers it first
        cancelled.append(part["name"])
        _save_state(run_dir, state)
    return {"cancelled": cancelled}


def status(run_dir: Path | str) -> dict[str, Any]:
    """A local summary of ``batches.json`` — parts, counts and batch-rate spend. No network."""
    state = load_state(run_dir)
    parts = state["parts"]
    open_parts = [p for p in parts if _is_open(p)]
    spent = [p.get("usd") for p in parts if p.get("collected_at")]
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "calls": 0}
    for part in parts:
        for key in usage:
            usage[key] += int((part.get("usage") or {}).get(key) or 0)
    return {
        "run_id": state.get("run_id"),
        "submissions": state.get("submissions", 0),
        "status": _aggregate_status(open_parts) if open_parts else (
            "collected" if parts else "empty"
        ),
        "open": [_part_summary(p) for p in open_parts],
        "parts": [_part_summary(p) for p in parts],
        "request_counts": _counts(parts),
        "usage": usage,
        "usd": None if any(s is None for s in spent) else sum(spent),
        "max_enqueued_tokens": state.get("max_enqueued_tokens"),
        "batch_multiplier": cost.BATCH_PRICE_MULTIPLIER,
    }
