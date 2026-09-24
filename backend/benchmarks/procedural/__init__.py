# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Procedural Graphs benchmark: does procedural memory make the Navigator better or cheaper?

The retrieval benchmarks next door (``benchmarks/``, ``benchmarks/public/``)
score what reaches the model. This package scores the *agent*: the GraphRAG
Navigator (``app.services.graph_agent``) answering questions with and without a
Procedural Graph (Lu, Chen, Wu, Arık, arXiv:2609.09153), by EM / F1, steps,
LLM calls, tokens and latency, the columns of the paper's Table 9.

``demo_qa.json`` is 30 questions over the zero-key demo graph, each checked
against the fixture; ``run_procedural.py`` is the harness. It spends real LLM
calls, so ``--dry-run`` first. See ``README.md``.
"""
