# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Evaluation against *public* benchmarks — corpora and questions from outside Synapse.

``benchmarks/`` (the sibling package) is honest but self-authored: 34 passages and
14 questions written by the same process that built the graph, a limitation its own
"Threats to validity" section concedes. Nothing in here is authored by this project.

Currently: **HotpotQA** (Yang et al., EMNLP 2018), dev / distractor split — a
recognised multi-hop QA benchmark with published baselines. See ``hotpotqa.py``.
"""
