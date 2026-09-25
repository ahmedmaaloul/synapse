# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse Lab — token counting with a real tokenizer, and honesty when there isn't one.

A token budget is only a budget if it is measured in the reader's own units.
``chat_engine`` budgets in characters because it hands context to somebody
else's model; the Lab knows the reader, so it counts with ``tiktoken``:

  • ``o200k_base`` for gpt-4o*, gpt-4.1*, gpt-5* and the o-series (o1, o3, …);
  • ``cl100k_base`` for everything else.

When ``tiktoken`` is not importable, or the encoding file cannot be loaded
(``tiktoken`` downloads it once and caches it), the count falls back to
``ceil(len(text) / 4)`` and says so: :func:`count_tokens` returns
``(count, estimated)`` and the flag travels into every report that uses it.

Offline safety: ``tiktoken`` fetches an encoding it has not cached over the
network, without a timeout. Setting ``SYNAPSE_TOKENIZER_OFFLINE=1`` — and
running under pytest, which sets ``PYTEST_CURRENT_TEST`` — restricts loading to
encodings already in tiktoken's cache, so a test suite never downloads
anything; a missing file then means the estimated fallback, not a hang.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from functools import lru_cache
from typing import Any

CHARS_PER_TOKEN = 4

O200K = "o200k_base"
CL100K = "cl100k_base"

#: Models tokenised with ``o200k_base``. Everything else gets ``cl100k_base``.
_O200K_MODELS = re.compile(r"^(?:chatgpt-4o|gpt-4o|gpt-4\.1|gpt-5|o[1-9])")

#: Where tiktoken publishes its encodings (used only to locate the cache file).
_ENCODING_URL = "https://openaipublic.blob.core.windows.net/encodings/{name}.tiktoken"


def encoding_name_for(model: str | None) -> str:
    """The tiktoken encoding name for ``model`` (see module docstring)."""
    name = (model or "").strip().lower()
    return O200K if _O200K_MODELS.match(name) else CL100K


def _offline() -> bool:
    flag = os.environ.get("SYNAPSE_TOKENIZER_OFFLINE", "").strip().lower()
    return flag in {"1", "true", "yes"} or bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _cache_path(name: str) -> str:
    """tiktoken's own cache location for encoding ``name`` (mirrors ``read_file_cached``)."""
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
    elif "DATA_GYM_CACHE_DIR" in os.environ:
        cache_dir = os.environ["DATA_GYM_CACHE_DIR"]
    else:
        cache_dir = os.path.join(tempfile.gettempdir(), "data-gym-cache")
    url = _ENCODING_URL.format(name=name)
    return os.path.join(cache_dir, hashlib.sha1(url.encode()).hexdigest())


def encoding_cached(name: str) -> bool:
    """True when tiktoken can load ``name`` without touching the network."""
    return os.path.exists(_cache_path(name))


@lru_cache(maxsize=8)
def _load_encoding(name: str, offline: bool) -> Any | None:
    """The tiktoken ``Encoding`` for ``name``, or ``None`` when it cannot be had.

    Cached — failures included — so a missing tokenizer costs one attempt per
    process, not one per string.
    """
    try:
        import tiktoken
    except Exception:  # noqa: BLE001 - tiktoken is optional; the fallback is flagged
        return None
    if offline and not encoding_cached(name):
        return None
    try:
        return tiktoken.get_encoding(name)
    except Exception:  # noqa: BLE001 - no network, corrupt cache, unknown name …
        return None


def get_encoding(model: str | None) -> Any | None:
    """The encoding used for ``model``, or ``None`` (→ the estimated fallback)."""
    return _load_encoding(encoding_name_for(model), _offline())


def estimate_tokens(text: str) -> int:
    """``ceil(len(text) / 4)`` — the flagged fallback, never negative."""
    n = len(text or "")
    return math.ceil(n / CHARS_PER_TOKEN) if n else 0


def count_tokens(text: str, model: str | None) -> tuple[int, bool]:
    """``(tokens, estimated)`` for ``text`` as ``model`` would tokenise it.

    ``estimated`` is True when the real tokenizer was unavailable and the count
    is the ``len/4`` fallback. Special-token markers in the text are counted as
    ordinary text (``disallowed_special=()``) — user documents can contain
    ``<|endoftext|>`` and must not crash a budget computation.
    """
    if not text:
        return 0, False  # nothing to count: exact under any tokenizer
    encoding = get_encoding(model)
    if encoding is None:
        return estimate_tokens(text), True
    try:
        return len(encoding.encode(text, disallowed_special=())), False
    except Exception:  # noqa: BLE001 - a tokenizer bug must not break a run
        return estimate_tokens(text), True


def tokenizer_label(model: str | None) -> str:
    """What the manifest records: ``tiktoken:<encoding>`` or the fallback rule."""
    if get_encoding(model) is None:
        return f"estimate:len/{CHARS_PER_TOKEN}"
    return f"tiktoken:{encoding_name_for(model)}"


def reset_cache() -> None:
    """Forget loaded encodings (tests flip the offline switch between cases)."""
    _load_encoding.cache_clear()
