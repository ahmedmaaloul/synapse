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
             A part OpenAI REJECTED at validation (below) is reported apart,
             under ``"rejected"``.
  collect  — once every open batch is terminal: download the output and
             error files (kept under ``run_dir/batches/``), return one record
             per request in input order, and mark those parts collected.
             A request that never ran (the batch expired or was cancelled)
             comes back as a synthesized error record, so nothing silently
             disappears. A REJECTED part is never collected: it has no
             records at all.
  failed_requests / resubmit_failed
           — the request lines whose latest result is an error, or that a
             rejected part never ran, and a new submission of exactly those
             (the partial-failure path).
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

ENQUEUED-TOKEN GATE. OpenAI caps the input tokens an organisation may have
enqueued per model (tier-dependent — see the organisation's limits page). Set
``max_enqueued_tokens`` (or ``SYNAPSE_LAB_BATCH_MAX_ENQUEUED_TOKENS``) and
parts are sized under it and submitted one wave at a time: a part is created
only when the model's in-flight tokens plus its own fit under the gate, and
``poll`` sends the next part when the in-flight ones finish. Unset = no gate
until OpenAI refuses a batch for it (below). When the gate TIGHTENS, the parts
still queued under the looser one (never sent: no batch was created for them,
nothing was billed) are re-planned under the new gate before anything else is
sent — their part becomes ``replanned`` (never collected, $0) and its requests,
in order, go to new parts — so no stale part is sent only to be refused again.

REJECTED AT VALIDATION. A batch that ends ``failed`` with batch-level errors
and no output file never ran a single request, and nothing was billed. Its
part becomes ``rejected`` (never ``collected``; $0; the error code and message
kept under ``rejection``), and ``poll`` lists it under ``"rejected"`` until
its requests are submitted again — so a caller can tell a refusal of the whole
file from per-request failures. When the code is ``token_limit_exceeded``
("… Limit: 2,000,000 enqueued tokens …") the limit is parsed and the gate is
set AUTOMATICALLY to ``floor(0.9 × limit)`` in ``batches.json`` (recorded
under ``enqueued_gate``), unless the gate was set explicitly (the argument or
the environment variable win). A part the gate let through that is still
refused tightens the auto gate to 90% of that part. Resubmitting the rejected
requests (``resubmit_failed``, or ``submit`` of ``rejected_requests``) splits
them under the gate and sends them one part at a time. A rejected part's
requests are owed until each is in flight again or has a SUCCESSFUL result —
the same rule as ``failed_requests``, so a refused resend of failed requests
stays owed (and listed) instead of silently resolving.

COST. Every collected part records its usage and its $ at the batch rate
(``cost.usd(..., batch=True)``). A request that comes back as an error carries
no usage and is priced at $0 here; the runner reads it as an error
(``done_with_errors``) and can resend it. A rejected part is $0.

NEVER call this against the real API from a test: without an explicit
``client`` it builds :func:`app.lab.reader.default_client`, which refuses to
exist under pytest.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import re
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
#: A part whose batch OpenAI refused at validation (``failed``, batch-level
#: errors, no output file): nothing ran, nothing was billed, never collected.
REJECTED = "rejected"
#: A queued part (never sent: no batch was created for it) whose requests were
#: split again under a tighter enqueued-token gate: never collected, $0.
REPLANNED = "replanned"
#: OpenAI batch states.
TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})
IN_FLIGHT = frozenset({"validating", "in_progress", "finalizing", "cancelling"})
#: The batch-level error code of an organisation's enqueued-token limit.
TOKEN_LIMIT_CODE = "token_limit_exceeded"
#: The auto gate keeps this fraction of the limit OpenAI reports (our tokenizer
#: count and OpenAI's differ a little; other batches may hold some of the quota).
AUTO_GATE_FRACTION = 0.9
_LIMIT_RE = re.compile(r"limit:\s*([0-9][0-9,_ ]*)\s*(?:enqueued\s+)?tokens", re.IGNORECASE)
_LIMIT_MODEL_RE = re.compile(r"limit reached for\s+(\S+?)\s+in organization", re.IGNORECASE)
_ORG_RE = re.compile(r"\borg-[A-Za-z0-9]+")


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


def _is_rejected(part: dict) -> bool:
    return part.get("status") == REJECTED


def _never_runs(part: dict) -> bool:
    """Rejected or re-planned: none of its requests ran (or will), nothing to collect."""
    return part.get("status") in (REJECTED, REPLANNED)


def _is_open(part: dict) -> bool:
    """Not collected yet and not void: its requests are (or will be) in flight."""
    return not part.get("collected_at") and not _never_runs(part)


def _is_terminal(part: dict) -> bool:
    return part.get("status") in TERMINAL or _never_runs(part)


def _outstanding_rejections(state: dict[str, Any]) -> list[dict]:
    """Rejected parts whose requests have not been submitted again yet."""
    return [p for p in state["parts"] if _is_rejected(p) and not p.get("resubmitted_at")]


def _enqueued_limit(explicit: int | None) -> int | None:
    if explicit is not None:
        return int(explicit) if int(explicit) > 0 else None
    return _env_enqueued_limit()


def _env_enqueued_limit() -> int | None:
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
            {"code": _get(e, "code"), "message": redact(_get(e, "message")),
             "line": _get(e, "line")}
            for e in errors
        ]
    for key in ("created_at", "in_progress_at", "finalizing_at", "completed_at", "failed_at",
                "expired_at", "expires_at", "cancelled_at"):
        value = _get(batch, key)
        if value is not None:
            part.setdefault("openai_times", {})[key] = value


# ── Rejection at validation, and the automatic enqueued-token gate ───────────
def redact(message: Any) -> Any:
    """``message`` with OpenAI organisation ids masked (they end up in manifests)."""
    return _ORG_RE.sub("org-…", message) if isinstance(message, str) else message


def parse_enqueued_limit(message: str | None) -> int | None:
    """The limit in a ``token_limit_exceeded`` message, e.g. 2,000,000 → ``2000000``."""
    match = _LIMIT_RE.search(message or "")
    if not match:
        return None
    digits = re.sub(r"\D", "", match.group(1))
    return int(digits) if digits and int(digits) > 0 else None


def _rejection_of(part: dict) -> dict[str, Any] | None:
    """What OpenAI refused a part for — ``None`` unless it was rejected at validation.

    Rejected = the batch is ``failed`` (OpenAI's state for an input file that
    failed validation) with batch-level errors and no output file: no request
    ran, so nothing was billed and there is nothing to collect.
    """
    if part.get("status") != "failed" or part.get("output_file_id"):
        return None
    errors = [e for e in part.get("batch_errors") or [] if isinstance(e, dict)]
    if not errors:
        return None
    first = next((e for e in errors if e.get("code") == TOKEN_LIMIT_CODE), errors[0])
    message = str(redact(first.get("message")) or "")
    code = str(first.get("code") or "")
    token_limit = code == TOKEN_LIMIT_CODE or "enqueued token limit" in message.lower()
    model_match = _LIMIT_MODEL_RE.search(message)
    return {
        "code": code or None,
        "message": message,
        "limit": parse_enqueued_limit(message) if token_limit else None,
        "model": model_match.group(1) if model_match else part.get("model"),
        # A capacity refusal clears once the quota frees up; any other validation
        # error (a malformed file) would be refused again as it is.
        "retryable": token_limit,
        "errors": errors[:5],
    }


def _mark_rejected(state: dict[str, Any], part: dict) -> bool:
    """Turn a refused part into ``rejected`` ($0, never collected). True when it was."""
    if _is_rejected(part):
        return False
    rejection = _rejection_of(part)
    if rejection is None:
        return False
    part["openai_status"] = part.get("status")
    part["status"] = REJECTED
    part["rejected_at"] = _now()
    part["rejection"] = rejection
    part["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
                     "calls": 0}
    part["usd"] = 0.0
    part["succeeded"] = 0
    part["failed"] = 0
    logger.warning("%s: batch %s rejected at validation (%s: %s) — nothing ran, $0",
                   part["name"], part.get("batch_id"), rejection["code"], rejection["message"])
    if rejection["limit"]:
        _auto_gate(state, part, rejection)
    return True


def _auto_gate(state: dict[str, Any], part: dict, rejection: dict[str, Any]) -> None:
    """Set the enqueued-token gate from the limit a rejection reported (explicit wins)."""
    limit = int(rejection["limit"])
    gate = math.floor(AUTO_GATE_FRACTION * limit)
    current = state.get("max_enqueued_tokens")
    tokens = part.get("prompt_tokens")
    if current and tokens and int(tokens) <= int(current):
        # The gate already let this part through and it was still refused: OpenAI
        # counts more than we do, or other batches hold part of the quota. Tighten.
        gate = min(gate, math.floor(AUTO_GATE_FRACTION * int(tokens)))
    source = state.get("max_enqueued_tokens_source") or ("explicit" if current else None)
    env = _env_enqueued_limit()
    record: dict[str, Any] = {
        "limit_reported": limit,
        "model": rejection.get("model"),
        "fraction": AUTO_GATE_FRACTION,
        "from_part": part["name"],
        "batch_id": part.get("batch_id"),
        "message": rejection.get("message"),
        "at": _now(),
    }
    if env is not None or source == "explicit":
        kept = env if env is not None else int(current)
        record.update(applied=False, value=kept, source="env" if env is not None else "explicit")
        if kept > limit:
            record["warning"] = (
                f"the explicit gate {kept:,} is above the {limit:,} OpenAI reported; parts "
                "sized under it can be refused again"
            )
            logger.warning("%s", record["warning"])
        if env is not None:
            state["max_enqueued_tokens"] = env
            state["max_enqueued_tokens_source"] = "env"
    else:
        if current and source == "auto":
            gate = min(gate, int(current))
        state["max_enqueued_tokens"] = gate
        state["max_enqueued_tokens_source"] = "auto"
        record.update(applied=True, value=gate, source="auto")
        logger.warning("enqueued-token gate set to %s (%.0f%% of OpenAI's reported %s)",
                       f"{gate:,}", AUTO_GATE_FRACTION * 100, f"{limit:,}")
    state["enqueued_gate"] = record
    state.setdefault("gate_history", []).append(record)


def _rejection_summary(part: dict) -> dict[str, Any]:
    rejection = part.get("rejection") or {}
    return {
        "part": part["name"],
        "batch_id": part.get("batch_id"),
        "model": part.get("model"),
        "requests": part.get("requests"),
        "prompt_tokens": part.get("prompt_tokens"),
        "code": rejection.get("code"),
        "message": rejection.get("message"),
        "limit": rejection.get("limit"),
        "retryable": bool(rejection.get("retryable")),
        "rejected_at": part.get("rejected_at"),
    }


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
    _replan_queued(run_dir, state)
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
        _mark_rejected(state, part)
        part["submitted_at"] = part.get("submitted_at") or _now()
        _save_state(run_dir, state)
        logger.info("%s: batch %s (%s requests) %s", part["name"], part["batch_id"],
                    part["requests"], part["status"])
    _save_state(run_dir, state)  # records every queued part's reason
    return client


def _replan_queued(run_dir: Path, state: dict[str, Any]) -> None:
    """Split again, under the CURRENT gate, the queued parts planned under a looser one.

    A queued part was sized under the gate of its time. Once a rejection has
    tightened the gate, sending it as it is would only get it refused again ($0,
    but one owner re-run per stale part). Only parts no batch was ever created
    for are touched (``local``/``uploaded`` with no creation attempt: nothing ran,
    nothing was billed). The queued parts of one model and submission are
    re-planned together, their requests in order, each new part no larger (in
    requests and bytes) than the largest part it replaces; the old parts become
    ``replanned`` and point at their successors (``replanned_into``).
    """
    gate = state.get("max_enqueued_tokens")
    if not gate:
        return
    gate = int(gate)
    queued: dict[tuple[str, Any], list[dict]] = {}
    for part in state["parts"]:
        if part.get("status") in (LOCAL, UPLOADED) and not part.get("create_attempted_at"):
            queued.setdefault((str(part.get("model")), part.get("submission")), []).append(part)
    token_cache: dict[str, int] = {}

    def tokens_of(line: dict) -> int:
        if line["custom_id"] not in token_cache:
            token_cache[line["custom_id"]] = request_prompt_tokens(line)
        return token_cache[line["custom_id"]]

    replaced: dict[str, list[str]] = {}
    for (model, submission), parts in queued.items():
        for part in parts:
            if part.get("prompt_tokens") is None:  # planned before any gate existed
                part["prompt_tokens"] = sum(
                    tokens_of(x) for x in _read_jsonl(_part_path(run_dir, part, "input")))
        if not any(int(p["prompt_tokens"]) > gate and int(p.get("requests") or 0) > 1
                   for p in parts):
            continue  # the common case: no input file is read
        lines = [x for p in parts for x in _read_jsonl(_part_path(run_dir, p, "input"))]
        groups = plan_parts(lines, max_tokens=gate, tokens_of=tokens_of,
                            max_requests=max(int(p.get("requests") or 1) for p in parts),
                            max_bytes=max(int(p.get("bytes") or 1) for p in parts))
        names = []
        for group in groups:
            data = b"".join(encode_line(x) for x in group)
            new = {
                "name": f"part-{len(state['parts']):04d}",
                "submission": submission,
                "model": model,
                "sha256": hashlib.sha256(data).hexdigest(),
                "requests": len(group),
                "bytes": len(data),
                "prompt_tokens": sum(tokens_of(x) for x in group),
                "status": LOCAL,
                "planned_at": _now(),
                "replanned_from": [p["name"] for p in parts],
            }
            _write_bytes_atomic(_part_path(run_dir, new, "input"), data)
            state["parts"].append(new)
            names.append(new["name"])
        for part in parts:
            part.pop("queued_reason", None)
            part.update(status=REPLANNED, replanned_at=_now(), replanned_into=names,
                        replanned_gate=gate, usd=0.0, succeeded=0, failed=0)
            replaced[part["name"]] = names
        logger.warning("re-planned %s queued part(s) (%s request(s)) of %s into %s part(s) "
                       "under the tightened gate %s", len(parts), len(lines), model,
                       len(names), f"{gate:,}")
    if not replaced:
        return
    for part in state["parts"]:  # a rejection re-sent in a re-planned part: its successors
        if part.get("resubmitted_in"):
            part["resubmitted_in"] = list(dict.fromkeys(
                n for x in part["resubmitted_in"] for n in replaced.get(x, [x])))
    _save_state(run_dir, state)


def _part_summary(part: dict) -> dict[str, Any]:
    return {
        "part": part["name"],
        "batch_id": part.get("batch_id"),
        "model": part.get("model"),
        "status": part.get("status"),
        "requests": part.get("requests"),
        "prompt_tokens": part.get("prompt_tokens"),
        "request_counts": part.get("request_counts"),
        "submission": part.get("submission"),
        "collected": bool(part.get("collected_at")),
        **({"queued_reason": part["queued_reason"]} if part.get("queued_reason") else {}),
        **({"batch_errors": part["batch_errors"]} if part.get("batch_errors") else {}),
        **({"rejection": part["rejection"]} if part.get("rejection") else {}),
        **({"resubmitted_in": part["resubmitted_in"]} if part.get("resubmitted_in") else {}),
        **({"replanned_into": part["replanned_into"]} if part.get("replanned_into") else {}),
    }


def _answered_ids(run_dir: Path, state: dict[str, Any]) -> set[str]:
    """custom_ids whose latest collected result is a success (local files only)."""
    return {cid for cid, record in _latest(run_dir, state).items() if record_ok(record)}


def _resolve_rejections(run_dir: Path, state: dict[str, Any]) -> bool:
    """Mark rejected parts whose every request is now in an open part or answered.

    "Answered" = a SUCCESSFUL latest result: an id whose latest result is an
    error is still owed (see :func:`rejected_requests`). True when one changed.
    """
    outstanding = _outstanding_rejections(state)
    if not outstanding:
        return False
    owners = _open_custom_ids(run_dir, state)
    answered = _answered_ids(run_dir, state)
    changed = False
    for part in outstanding:
        ids = [str(r["custom_id"]) for r in _read_jsonl(_part_path(run_dir, part, "input"))
               if r.get("custom_id")]
        if all(cid in owners or cid in answered for cid in ids):
            part["resubmitted_at"] = _now()
            part["resubmitted_in"] = sorted({owners[cid] for cid in ids if cid in owners})
            changed = True
    return changed


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

    ``requests`` are OpenAI Batch input lines. A line whose latest collected
    result is already a success is never sent again (it is dropped and counted
    under ``already_answered``). Returns
    ``{"batch_ids", "parts", "requests", "queued", "submission", "max_enqueued_tokens"}``.
    """
    run_dir = Path(run_dir)
    lines = validate_requests(requests)
    state = load_state(run_dir)
    answered = _answered_ids(run_dir, state) if lines and state["parts"] else set()
    already_answered = sum(1 for line in lines if line["custom_id"] in answered)
    if already_answered:
        logger.warning("dropping %s request(s) that already have a successful answer",
                       already_answered)
        lines = [line for line in lines if line["custom_id"] not in answered]
    if not lines:
        if state["parts"] and _resolve_rejections(run_dir, state):
            _save_state(run_dir, state)  # a rejection whose requests all have an answer
        return {"batch_ids": [], "parts": [], "requests": 0, "queued": 0,
                "submission": state.get("submissions", 0),
                "already_answered": already_answered}
    limit = _enqueued_limit(max_enqueued_tokens)
    if limit is not None:
        state["max_enqueued_tokens"] = limit
        state["max_enqueued_tokens_source"] = (
            "explicit" if max_enqueued_tokens is not None else "env"
        )
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
    _resolve_rejections(run_dir, state)
    _save_state(run_dir, state)

    await _advance(run_dir, state, client)
    return {
        "batch_ids": [p["batch_id"] for p in parts if p.get("batch_id")],
        "parts": [_part_summary(p) for p in parts],
        "requests": sum(int(p["requests"]) for p in parts),
        "queued": sum(1 for p in parts if p.get("status") in (LOCAL, UPLOADED)),
        "submission": state.get("submissions", 0),
        "reused": sum(1 for p in parts if p["sha256"] in reused),
        "already_answered": already_answered,
        "max_enqueued_tokens": state.get("max_enqueued_tokens"),
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

    Returns ``{"done", "status", "parts", "request_counts", "batch_ids",
    "rejected", "max_enqueued_tokens", "enqueued_gate"}`` — ``done`` once every
    open part is terminal (completed, failed, expired or cancelled). A part
    OpenAI refused at validation is NOT among the open parts: it is listed under
    ``rejected`` (with its error code and message) until its requests are
    submitted again, so ``{"done": True, "rejected": [...]}`` means "nothing is
    running, and these requests never ran" — not per-request failures. Makes no
    network call when nothing is open.
    """
    run_dir = Path(run_dir)
    state = load_state(run_dir)
    for part in state["parts"]:
        if _is_open(part) and part.get("batch_id") and not _is_terminal(part):
            client = _client(client)
            _apply_batch(part, await _maybe_await(client.batches.retrieve(part["batch_id"])))
            _mark_rejected(state, part)
    state["polled_at"] = _now()
    _save_state(run_dir, state)
    if any(_is_open(p) and p.get("status") in (LOCAL, UPLOADED) for p in state["parts"]):
        client = await _advance(run_dir, state, client)
    if all(_is_terminal(p) for p in state["parts"] if _is_open(p)) and _resolve_rejections(
        run_dir, state
    ):  # local, and only once nothing runs: a rejection whose requests are all answered
        _save_state(run_dir, state)
    open_parts = [p for p in state["parts"] if _is_open(p)]
    rejected = [_rejection_summary(p) for p in _outstanding_rejections(state)]
    done = all(_is_terminal(p) for p in open_parts)
    return {
        "done": done,
        "status": _aggregate_status(open_parts) if open_parts or not rejected else REJECTED,
        "parts": [_part_summary(p) for p in open_parts],
        "request_counts": _counts(open_parts),
        "batch_ids": [p["batch_id"] for p in open_parts if p.get("batch_id")],
        "rejected": rejected,
        "max_enqueued_tokens": state.get("max_enqueued_tokens"),
        "enqueued_gate": state.get("enqueued_gate"),
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
        if _never_runs(part):
            continue  # refused at validation or re-planned: nothing ran, nothing to hand over
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
    so a successful retry supersedes the error it retried. A rejected or
    re-planned part has no results at all (its requests never ran there), so it
    contributes nothing.
    """
    run_dir = Path(run_dir)
    return _latest(run_dir, load_state(run_dir))


def _latest(run_dir: Path, state: dict[str, Any]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for part in state["parts"]:
        if _is_open(part) or _never_runs(part):
            continue
        for record in _part_records(run_dir, part):
            custom_id = record["custom_id"]
            if custom_id in latest and record_ok(latest[custom_id]) and not record_ok(record):
                continue
            latest[custom_id] = record
    return latest


def failed_requests(run_dir: Path | str) -> list[dict]:
    """The request lines to resubmit: latest collected result an error, or never run.

    "Never run" = held by a part OpenAI rejected at validation, with no result
    since. A request already in flight in an open part is not listed again, and
    one with a successful answer never is.
    """
    run_dir = Path(run_dir)
    state = load_state(run_dir)
    latest = _latest(run_dir, state)
    in_flight = _open_custom_ids(run_dir, state)
    lines: dict[str, dict] = {}
    for part in state["parts"]:
        if _is_open(part) or part.get("status") == REPLANNED:
            continue  # a re-planned part's requests live in its successors
        rejected = _is_rejected(part)
        for line in _read_jsonl(_part_path(run_dir, part, "input")):
            custom_id = str(line.get("custom_id") or "")
            if custom_id in in_flight:
                continue
            record = latest.get(custom_id)
            never_ran = record is None and rejected
            if never_ran or (record is not None and not record_ok(record)):
                lines[custom_id] = line
    return list(lines.values())


def rejected_requests(run_dir: Path | str) -> list[dict]:
    """The request lines rejected parts still owe: no SUCCESSFUL result, not in flight.

    The rule of :func:`failed_requests` (and of :func:`_resolve_rejections`): an
    id whose latest result is an error is owed too — a resend of failed requests
    can itself be refused at validation, and those requests never ran again.
    """
    run_dir = Path(run_dir)
    state = load_state(run_dir)
    latest = _latest(run_dir, state)
    in_flight = _open_custom_ids(run_dir, state)
    lines: dict[str, dict] = {}
    for part in state["parts"]:
        if not _is_rejected(part):
            continue
        for line in _read_jsonl(_part_path(run_dir, part, "input")):
            custom_id = str(line.get("custom_id") or "")
            if custom_id not in in_flight and not record_ok(latest.get(custom_id) or {}):
                lines[custom_id] = line
    return list(lines.values())


async def resubmit_failed(
    run_dir: Path | str, *, run_id: str | None = None, client: Any = None
) -> dict[str, Any]:
    """Submit exactly the failed requests again, as a new batch submission.

    Includes the requests of parts OpenAI rejected at validation (they never ran),
    split under the enqueued-token gate — the auto gate when a rejection set one.
    """
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
    ran = [p for p in parts if not _never_runs(p)]
    rejected = _outstanding_rejections(state)
    spent = [p.get("usd") for p in parts if p.get("collected_at")]
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "calls": 0}
    for part in parts:
        for key in usage:
            usage[key] += int((part.get("usage") or {}).get(key) or 0)
    if open_parts:
        overall = _aggregate_status(open_parts)
    elif rejected:
        overall = REJECTED
    else:
        overall = "collected" if parts else "empty"
    return {
        "run_id": state.get("run_id"),
        "submissions": state.get("submissions", 0),
        "status": overall,
        "open": [_part_summary(p) for p in open_parts],
        "parts": [_part_summary(p) for p in parts],
        # Rejected and re-planned parts ran nothing ($0): their requests are
        # counted where they are sent again, never twice.
        "request_counts": _counts(ran),
        "rejected": [_rejection_summary(p) for p in rejected],
        "rejected_parts": sum(1 for p in parts if _is_rejected(p)),
        "replanned_parts": sum(1 for p in parts if p.get("status") == REPLANNED),
        "usage": usage,
        "usd": None if any(s is None for s in spent) else sum(spent),
        "max_enqueued_tokens": state.get("max_enqueued_tokens"),
        "max_enqueued_tokens_source": state.get("max_enqueued_tokens_source"),
        "enqueued_gate": state.get("enqueued_gate"),
        "batch_multiplier": cost.BATCH_PRICE_MULTIPLIER,
    }
