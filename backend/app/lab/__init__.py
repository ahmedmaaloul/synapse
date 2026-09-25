# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse Lab — compare retrieval approaches side by side on your own data.

FinOps first: every arm is scored on answer quality AND on what it costs, next
to three evidence floors (closed-book, vocabulary null, random context) so a
gain is only ever reported above the floor it has to clear.

Modules (import them directly; this package deliberately imports nothing, so
the pure accounting pieces stay importable without Neo4j or a provider SDK):

  • ``evidence`` — the ranked-unit shape every arm returns.
  • ``tokens``   — real-tokenizer counts (tiktoken), flagged when estimated.
  • ``packer``   — the ONE shared packer that fills a token budget.
  • ``arms``     — the arms and their registry (``ARMS``).
  • ``reader``   — the ONE short-answer prompt + the realtime OpenAI reader.
  • ``metrics``  — EM/F1, cost-of-pass, floors, graph premium, Pareto, bootstrap.
  • ``estimate`` — the free dry-run estimator (point + upper bound + refusal).
  • ``runner``   — ``LabRun``: retrieve → read → score, resumable, capped.
"""

from __future__ import annotations
