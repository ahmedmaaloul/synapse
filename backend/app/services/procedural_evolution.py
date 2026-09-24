# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Procedural Graph self-evolution (offline; Algorithm 1)

A hand-written procedure encodes one person's intuition about how to use the
graph. This module learns the procedure from outcomes instead. It implements
Algorithm 1 of Lu, Chen, Wu, Arık, "Procedural Graphs: Self-Evolving Execution
Structures for LLM Agents" (arXiv:2609.09153, Appendix B.6), over the
GraphRAG Navigator environment (``graph_agent``), with QA pairs scored by
EM / F1 (``qa_metrics``):

    S0 ← Evaluate(G0, D_val);  H_rejected ← []
    for k = 1..K:
        E_k   ← Rollout(G_{k-1}, B_k)                  training traces + scores
        C_k   ← Tail_Lmax(concat(E_k))                 keep the END of the text
        ΔG_k  ← Refiner(G_{k-1}, C_k, scores, SerializeRejections(H_rejected))
        G_cand, d_k ← PrepareCandidate(G_{k-1}, ΔG_k, cycle policy)
        if d_k ≠ ∅: H_rejected += (ΔG_k, E_k, d_k); continue   NO validation rollout
        S_cand ← Evaluate(G_cand, D_val)
        if S_cand ≥ S_{k-1}: accept (ties included) → a new stored version
        else: H_rejected += (ΔG_k, G_cand, E_k, S_cand)

What is faithful, and what is Synapse's:

  • Structural checks, edit order and cycle repair are
    ``procedural_graph.prepare_candidate``, unchanged. A structurally invalid
    candidate never costs a validation rollout.
  • ``mode="static"`` starts from the stored graph (the paper's
    static_incremental, Mode 3). ``mode="scratch"`` starts from
    ``Start → End`` with the navigator's tool list (scratch_incremental,
    Mode 5). The stored graph is NOT touched until a candidate is accepted,
    and the skeleton's own S0 is measured first.
  • Scratch mode on a name that already holds a graph is refused
    (``ProceduralGraphExists``) unless ``replace=True``. With it, the stored
    graph is also evaluated ONCE on D_val, and the acceptance floor becomes
    max(S_skeleton, S_stored): a candidate grown from the skeleton can never
    overwrite a better stored graph. (Synapse's guard: the paper's scratch runs
    have no stored graph to lose.)
  • Training batches are sequential strides over ``train`` that wrap around
    (the paper strides; the wrap-around lets small sets run many rounds).
  • The traces shown to the refiner are grouped high-scoring first, then
    low-scoring. The paper tail-truncates the whole concatenation; Synapse
    splits ``evolution_trajectory_max_chars`` between the two partitions
    (half each, a partition's unused half going to the other) and
    tail-truncates EACH one, keeping its end. A long batch of failures
    therefore cannot push every success out, and in scratch mode the successes
    are the only material a first graph can be synthesized from. Each
    training trace includes its gold answer: these are training items, and
    rule 5 of the refiner prompt forbids copying specifics into the graph.
  • SerializeRejections shows the refiner, for each of the 8 most recent
    rejections, the candidate graph's structure (nodes and transitions) with
    its validation score, or the structural diagnostics, plus the edits
    (clipped). As in the paper, a structurally failed candidate has no graph
    (``prepare_candidate`` is strict), and the round's traces stay in the
    in-run record without being re-serialized.
  • Every accepted candidate is a new ``ProcedureVersion`` (rollback-able), and
    every round that ends without one (``structural``, ``score``,
    ``no_change``, ``refiner_error``) is a durable ``ProcedureRejection`` row:
    the paper's rejection memory made auditable.
  • A candidate the edits leave IDENTICAL to G_{k-1} is not validated (it
    would tie and be "accepted" unchanged after |val| paid rollouts): the round
    ends with reason ``no_change`` and the refiner is told so next round.
  • A hard LLM-call budget covers rollouts, guidance and refiner calls alike.
    A rollout's worst case is max_steps turns, doubled with generative
    guidance. ``evolve`` refuses BEFORE ANY CALL (``EvolutionBudgetTooSmall``,
    which names the smallest sufficient ``max_llm_calls``) unless the budget
    covers the baseline's worst case (|val| rollouts, plus |val| more for a
    stored graph that ``replace`` may overwrite) AND one whole round's
    (|B_1| + |val| rollouts + the refiner call): a baseline alone decides
    nothing. Every later round must fit whole before it starts too, since a
    round decides nothing until its validation ends; when one does not, the
    run stops with ``stopped="budget"``, never mid-save. An external spend cap
    (``SpendCapReached``, e.g. the benchmark's ``--max-usd``) refusing the
    refiner call stops the run the same way.

Read the output honestly. The gate compares two means over the same small
validation set, so its resolution is one question: ``effect_floor = 1/|val|``.
An accept/reject decision is a step in a search, not a significance test. The
paper says the same of its own gate.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.services import procedural_store as store
from app.services.graph_agent import TASK_DESCRIPTION, TOOL_NAMES, TOOL_SPECS, run_agent
from app.services.llm_provider import ProviderConfigError, get_chat_llm
from app.services.procedural_graph import (
    EDIT_KEYS,
    NODE_TYPES,
    RELATIONS,
    ProceduralGraph,
    graph_diff,
    prepare_candidate,
    skeleton,
    summarize_diff,
)
from app.services.procedural_guidance import format_step_action, llm_usage, response_text
from app.services.procedural_store import ProceduralGraphNotFound
from app.services.qa_metrics import METRICS
from app.services.qa_metrics import score as qa_score

logger = logging.getLogger(__name__)

EVOLUTION_MODES: tuple[str, ...] = ("static", "scratch")
EVOLUTION_GUIDANCE: tuple[str, ...] = ("raw", "generative")

#: A training trace scoring at least this is shown as "high-scoring".
HIGH_SCORE_THRESHOLD = 0.5
#: Characters of each observation kept in the refiner's copy of a trace.
TRACE_OBSERVATION_CHARS = 400
#: Most recent rejections serialized into the refiner prompt.
MAX_REJECTIONS_SHOWN = 8
#: Characters of one rejected candidate's edits JSON shown to the refiner.
REJECTED_EDITS_CHARS = 2500
#: Characters of one rejected candidate graph's structure shown to the refiner.
REJECTED_GRAPH_CHARS = 2500

SIGNIFICANCE_NOTE = (
    "Accept/reject compares two means over one small validation set; its resolution is "
    "effect_floor = 1/|val|. Treat the rounds as a search trace, not as significance."
)

RunFn = Callable[..., Awaitable[Any]]
ProgressFn = Callable[[dict], Any]

# Adapted from the paper's refiner prompt (Appendix B.5): same slots, same
# seven rules, same four-array JSON contract. The wording is Synapse's. Rule 5
# is made explicit about names and answers because the traces carry gold answers.
REFINER_PROMPT = """You are an expert cognitive architect optimizing a Procedural Graph for an \
intelligent agent. The Procedural Graph encodes structured procedural guidance: nodes are tool \
ACTIONs, REASONING steps or STATUS markers, and every directed edge is a transition carrying a \
condition, guidance and pitfalls.

Task context: {task_description}
Refinement mode: {mode}
Cycle policy: {cycle_policy} (cycles are {cycles_allowed})
Available Tool Actions (the agent can only execute these actions):
{tools}

Training batch scores ({metric}, 0 to 1): mean {train_mean:.3f} over {n} tasks; {n_high} \
high-scoring (>= {threshold}), {n_low} low-scoring.

Recent execution trajectories (high-scoring first, then low-scoring; when a group is too long, \
its OLDEST text was cut and its end kept):
{attempts_block}

Current Procedural Graph representation:
{graph_json}

Previously rejected candidates (do not propose them again):
{rejected_block}

Your job is to refine the Procedural Graph. Follow the guideline for your mode:
- static_incremental: prune edges/nodes that lead to loops, dead ends or failures, and add the \
missing nodes and edges that would fix the failures and improve performance on future tasks.
- scratch_incremental: if the graph contains only Start -> End, synthesize a brand new, complete \
Procedural Graph from the Available Tool Actions, status nodes and the successful patterns in the \
trajectories. Otherwise prune edges/nodes that lead to loops, dead ends or failures, and add \
missing nodes and edges on top of the given graph.

Rules for nodes and edges:
1. Action nodes: every node of type ACTION must be named exactly after one of the Available Tool \
Actions.
2. Transition conditions: give each edge a natural-language precondition under which the \
transition should fire, or null when the transition is unconditional.
3. Execution guidance: every edge in add_edges MUST carry a guidance string saying exactly what to \
do next and why.
4. Pitfalls: give a pitfalls string warning about premature actions, forbidden moves and common \
formatting mistakes at that step.
5. Generality and leak prevention: the graph must help on UNSEEN questions. Use high-level, \
reusable descriptions; never copy a specific question, entity name or answer from the \
trajectories into the graph.
6. Node ID compatibility: keep existing node IDs (Start, End and the tool names) exactly as they \
are; do not rename them.
7. Graph structure: follow the cycle policy. Every edge must reference existing nodes (or nodes \
you add), and every node must have a directed path to a terminal node (a node with no outgoing \
edges). Node types: {node_types}. Relations: {relations}.

Output your edits as ONE JSON object with exactly these four arrays:
{{"add_nodes": [{{"id": "...", "type": "ACTION", "description": "..."}}],
 "delete_nodes": ["node_id"],
 "add_edges": [{{"source": "...", "target": "...", "relation": "LEADS_TO", "condition": null, \
"guidance": "...", "pitfalls": "..."}}],
 "delete_edges": [{{"source": "...", "target": "..."}}]}}
Edits are applied in this order: delete_edges, delete_nodes, add_nodes, add_edges. A delete_edges \
entry removes ALL edges between that source and target whatever their relation; to keep one of \
them, re-add it in add_edges. An add_nodes entry whose id already exists updates that node. \
Output ONLY the raw JSON object."""


class SpendCapReached(RuntimeError):
    """A caller's own spend cap refused a model call before it was made.

    A metered model (the benchmark's ``GuardedModel``) raises a subclass of
    this when its cap binds. ``evolve`` then stops the run as if its own
    budget were spent (``stopped="budget"``) instead of reporting a refiner
    failure and carrying on, and it does not count the refused call.
    """


class EvolutionBudgetTooSmall(ValueError):
    """``max_llm_calls`` cannot pay for the baseline and one whole round.

    Raised by ``evolve`` before a single call is made. ``minimum`` is the
    smallest cap that runs one round (``minimum_llm_calls``), so the API and
    the CLI can say exactly what to ask for instead of "too small".
    """

    def __init__(self, max_llm_calls: int, minimum: int, worst_case: str) -> None:
        self.max_llm_calls = int(max_llm_calls)
        self.minimum = int(minimum)
        super().__init__(
            f"max_llm_calls={self.max_llm_calls} cannot pay for the baseline and one full "
            f"round, whose worst case is {worst_case} = {self.minimum} calls; set "
            f"max_llm_calls to at least {self.minimum} (or use fewer validation questions "
            "or a smaller batch)"
        )


class ProceduralGraphExists(RuntimeError):
    """``mode="scratch"`` would overwrite a stored graph and ``replace`` was not given."""

    def __init__(self, name: str, version: int | None = None) -> None:
        self.name = name
        self.version = version
        at = f" (v{version})" if version is not None else ""
        super().__init__(
            f"Procedural graph '{name}' already exists{at}; mode=scratch would replace it. "
            "Pass replace=true to evaluate it once on the validation set and overwrite it "
            "only with a candidate that scores at least as well, or choose another name."
        )


class _BudgetExhausted(Exception):
    """The next unit of work could exceed ``max_llm_calls``; stop cleanly."""


class _Budget:
    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.used = 0

    def require(self, calls: int) -> None:
        if self.used + calls > self.limit:
            raise _BudgetExhausted

    def spend(self, calls: int) -> None:
        self.used += int(calls)


# ── Pure helpers ─────────────────────────────────────
TRUNCATION_MARKER = "…[earlier trajectory text truncated]\n"


def tail_truncate(text: str, max_chars: int) -> str:
    """Tail_Lmax: keep the last ``max_chars`` characters, drop from the START.

    Shorter input is returned unchanged, and so is any input when
    ``max_chars <= 0`` (no limit). A marker states that the beginning was cut,
    so the refiner does not read a mid-sentence start as a whole trace.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return TRUNCATION_MARKER + text[-max_chars:]


def _batch_len(train_size: int, batch_size: int) -> int:
    """|B_k|: the batch size, capped by the training set (a stride never repeats an item)."""
    return max(1, min(int(batch_size), int(train_size)))


def training_batch(train: Sequence[dict], round_number: int, batch_size: int) -> list[dict]:
    """B_k: the k-th sequential stride over ``train``, wrapping around at the end."""
    n = len(train)
    size = _batch_len(n, batch_size)
    start = ((round_number - 1) * size) % n
    return [train[(start + i) % n] for i in range(size)]


def rollout_worst_case(max_steps: int, guidance: str) -> int:
    """The most LLM calls one rollout can make.

    One solver call per step, doubled when every step also asks for generative
    guidance (the guidance cache can only lower this).
    """
    return max(1, int(max_steps)) * (2 if guidance == "generative" else 1)


def minimum_llm_calls(
    *,
    train_size: int,
    val_size: int,
    batch_size: int,
    per_rollout: int,
    stored_eval: bool = False,
) -> int:
    """The smallest ``max_llm_calls`` that runs the baseline and ONE full round.

    Worst case, in rollouts of ``per_rollout`` calls: S0 on D_val, the stored
    graph on D_val when scratch mode may replace one (``stored_eval``), round
    1's batch B_1 and its validation on D_val, plus the one refiner call. The
    router and the CLI refuse below this number before anything is spent.
    """
    evaluations = int(val_size) * (2 if stored_eval else 1)
    rollouts = evaluations + _batch_len(train_size, batch_size) + int(val_size)
    return rollouts * int(per_rollout) + 1


def check_budget(
    max_llm_calls: int,
    *,
    train_size: int,
    val_size: int,
    batch_size: int,
    per_rollout: int,
    stored_eval: bool = False,
) -> int:
    """Raise ``EvolutionBudgetTooSmall`` unless the cap runs one round; return the minimum."""
    minimum = minimum_llm_calls(
        train_size=train_size,
        val_size=val_size,
        batch_size=batch_size,
        per_rollout=per_rollout,
        stored_eval=stored_eval,
    )
    if int(max_llm_calls) < minimum:
        parts = [f"{val_size} baseline"]
        if stored_eval:
            parts.append(f"{val_size} stored-graph")
        parts += [f"{_batch_len(train_size, batch_size)} train", f"{val_size} val"]
        worst_case = f"({' + '.join(parts)} rollouts) × {per_rollout} calls + 1 refiner call"
        raise EvolutionBudgetTooSmall(max_llm_calls, minimum, worst_case)
    return minimum


_FENCE = re.compile(r"```[A-Za-z]*\s*(.*?)```", re.DOTALL)


def extract_edits(text: str) -> tuple[dict | None, str | None]:
    """The refiner's JSON edits, from a fenced block or bare JSON inside prose.

    Returns ``(edits, None)`` or ``(None, diagnostic)``. An object that parses
    but carries unknown keys is still returned: ``prepare_candidate`` reports
    those precisely.
    """
    text = text or ""
    decoder = json.JSONDecoder()
    for chunk in [m.group(1) for m in _FENCE.finditer(text)] + [text]:
        chunk = chunk.strip()
        try:
            value = json.loads(chunk)
        except ValueError:
            value = None
        if isinstance(value, dict):
            return value, None
        for position, char in enumerate(chunk):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(chunk, position)
            except ValueError:
                continue
            if isinstance(value, dict) and any(key in value for key in EDIT_KEYS):
                return value, None
    preview = " ".join(text.split())[:200]
    return None, (
        f"refiner output is not a JSON object with {', '.join(EDIT_KEYS)} (got: {preview!r})"
    )


def _flat(text: Any) -> str:
    return " ".join(str(text or "").split())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def format_trace(index: int, rollout: Mapping[str, Any]) -> str:
    """One training trace as the refiner reads it."""
    result = rollout["result"]
    lines = [
        f"--- Trajectory {index} | score {rollout['score']:.2f} ---",
        f"Question: {_flat(rollout['question'])}",
        f"Gold answer (training item): {_flat(rollout['gold'])}",
    ]
    for step in result.get("steps") or []:
        if step.get("thought"):
            lines.append(f"Thought: {_flat(step['thought'])}")
        action = format_step_action(step) if step.get("action") else step.get("raw_action")
        lines.append(f"Action: {action or '(unparseable)'}")
        if step.get("action") != "answer":
            observation = _clip(_flat(step.get("observation")), TRACE_OBSERVATION_CHARS)
            lines.append(f"Observation: {observation}")
    lines.append(
        f"Final answer: {result.get('answer')!r} (stopped: {result.get('stopped')}, "
        f"parse failures: {result.get('parse_failures', 0)})"
    )
    return "\n".join(lines)


def split_trace_budget(high_chars: int, low_chars: int, max_chars: int) -> tuple[int, int]:
    """Share C_k's character budget between the high- and low-scoring partitions.

    Half each; a partition that needs less than its half gives what it leaves
    unused to the other. Returns ``(high_budget, low_budget)``.
    """
    high_share = max_chars // 2
    low_share = max_chars - high_share
    if high_chars < high_share:
        return high_chars, max_chars - high_chars
    if low_chars < low_share:
        return max_chars - low_chars, low_chars
    return high_share, low_share


def _keep_end(text: str, budget: int) -> str:
    """``tail_truncate`` for a partition, where a zero budget means "nothing fits"."""
    if budget <= 0 and text:
        return TRUNCATION_MARKER.rstrip("\n")
    return tail_truncate(text, budget)


def attempts_block(rollouts: Sequence[Mapping[str, Any]], max_chars: int) -> str:
    """C_k: high-scoring traces, then low-scoring ones, each partition tail-truncated.

    The paper keeps the end of the WHOLE concatenation (Tail_Lmax). With the
    successes first, a batch of long failures would then push every success
    out, and in scratch mode the successes are what a first graph is
    synthesized from. So ``max_chars`` is shared by ``split_trace_budget``
    (half per partition, the unused part of one half going to the other) and
    each partition is tail-truncated to its share separately: the paper's
    keep-the-end rule, within each partition. The budget counts trace text;
    the two section headers and truncation markers come on top of it.
    ``max_chars <= 0`` means no limit.
    """
    high = [r for r in rollouts if r["score"] >= HIGH_SCORE_THRESHOLD]
    low = [r for r in rollouts if r["score"] < HIGH_SCORE_THRESHOLD]
    high_text = "\n\n".join(format_trace(i, r) for i, r in enumerate(high, start=1))
    low_text = "\n\n".join(format_trace(i, r) for i, r in enumerate(low, start=len(high) + 1))
    if max_chars > 0:
        high_budget, low_budget = split_trace_budget(len(high_text), len(low_text), max_chars)
        high_text, low_text = _keep_end(high_text, high_budget), _keep_end(low_text, low_budget)
    sections = []
    if high:
        sections.append("## High-scoring trajectories\n\n" + high_text)
    if low:
        sections.append("## Low-scoring trajectories\n\n" + low_text)
    return "\n\n".join(sections) or "(no trajectories)"


def compact_graph(graph: ProceduralGraph) -> str:
    """A candidate's structure in two lines: its nodes and its transitions.

    The guidance and pitfall texts are left out: the ones the candidate changed
    are in its edits, and repeating the rest for every rejection would crowd
    out the traces.
    """
    nodes = ", ".join(f"{node.id} ({node.type})" for node in graph.nodes.values())
    edges = "; ".join(f"{e.source} -{e.relation}-> {e.target}" for e in graph.edges)
    return f"Nodes: {nodes or '(none)'}\nTransitions: {edges or '(none)'}"


def serialize_rejections(rejected: Sequence[Mapping[str, Any]]) -> str:
    """SerializeRejections(H_rejected), after the paper's Appendix B.6.

    Each rejected candidate graph with its validation score, or the structural
    diagnostics when there is no graph, plus the edits that produced it.
    """
    if not rejected:
        return "(none)"
    shown = list(rejected)[-MAX_REJECTIONS_SHOWN:]
    omitted = len(rejected) - len(shown)
    blocks = []
    if omitted:
        blocks.append(f"({omitted} older rejected candidate(s) not shown)")
    for entry in shown:
        if entry["reason"] == "structural":
            header = f"Rejected in round {entry['round']}: structural failure"
            detail = "Diagnostics:\n" + "\n".join(f"- {d}" for d in entry["diagnostics"])
        elif entry["reason"] == "no_change":
            header = f"Round {entry['round']}: the edits left the graph unchanged"
            detail = "Propose a concrete change that addresses the low-scoring trajectories."
        else:
            # The bar was the retained graph's score, or (scratch mode with
            # replace) the higher score of the stored graph it would overwrite.
            bar = "the stored graph's" if entry.get("floor") == "stored" else "retained"
            header = (
                f"Rejected in round {entry['round']}: validation score "
                f"{entry['score']:.3f} < {bar} {entry['retained_score']:.3f}"
            )
            detail = f"Change: {entry.get('summary') or 'n/a'}"
            candidate = entry.get("candidate")
            if candidate is not None:
                structure = _clip(compact_graph(candidate), REJECTED_GRAPH_CHARS)
                detail = f"Candidate graph:\n{structure}\n{detail}"
        edits = json.dumps(entry.get("edits"), ensure_ascii=False, default=str)
        blocks.append(f"{header}\nEdits: {_clip(edits, REJECTED_EDITS_CHARS)}\n{detail}")
    return "\n\n".join(blocks)


def build_refiner_prompt(
    graph: ProceduralGraph,
    *,
    mode: str,
    metric: str,
    rollouts: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    max_chars: int,
) -> str:
    """Refiner(G_{k-1}, C_k, scores, R_k), with the prompt adapted from Appendix B.5."""
    scores = [r["score"] for r in rollouts]
    n_high = sum(1 for s in scores if s >= HIGH_SCORE_THRESHOLD)
    tools = "\n".join(
        f"- {name}: {TOOL_SPECS[name].description}" if name in TOOL_SPECS else f"- {name}"
        for name in (graph.tools or TOOL_NAMES)
    )
    return REFINER_PROMPT.format(
        task_description=TASK_DESCRIPTION,
        mode=f"{mode}_incremental",
        cycle_policy=graph.cycle_policy,
        cycles_allowed="allowed" if graph.cycle_policy == "allow" else "NOT allowed",
        tools=tools,
        metric=metric,
        train_mean=sum(scores) / len(scores) if scores else 0.0,
        n=len(scores),
        n_high=n_high,
        n_low=len(scores) - n_high,
        threshold=HIGH_SCORE_THRESHOLD,
        attempts_block=attempts_block(rollouts, max_chars),
        graph_json=json.dumps(graph.to_dict(), indent=1, ensure_ascii=False),
        rejected_block=serialize_rejections(rejected),
        node_types=", ".join(NODE_TYPES),
        relations=", ".join(RELATIONS),
    )


# ── The loop ─────────────────────────────────────────
def _validate_inputs(
    train: Sequence,
    val: Sequence,
    rounds: int,
    batch_size: int,
    mode: str,
    metric: str,
    guidance: str,
    max_llm_calls: int,
) -> None:
    if mode not in EVOLUTION_MODES:
        raise ValueError(f"mode must be one of {', '.join(EVOLUTION_MODES)}, got {mode!r}")
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {', '.join(METRICS)}, got {metric!r}")
    if guidance not in EVOLUTION_GUIDANCE:
        raise ValueError(
            f"guidance must be one of {', '.join(EVOLUTION_GUIDANCE)}, got {guidance!r}"
        )
    if int(rounds) < 1 or int(batch_size) < 1 or int(max_llm_calls) < 1:
        raise ValueError("rounds, batch_size and max_llm_calls must all be >= 1")
    for label, items in (("train", train), ("val", val)):
        if not items:
            raise ValueError(f"{label} must contain at least one {{question, answer}} item")
        for i, item in enumerate(items):
            if not isinstance(item, Mapping) or not str(item.get("question") or "").strip():
                raise ValueError(f"{label}[{i}] needs a non-empty 'question'")
            gold = item.get("answer")
            golds = gold if isinstance(gold, list) else [gold]
            if not any(str(g or "").strip() for g in golds):
                # A blank gold scores 0 whatever the agent says: refuse it
                # before a single rollout is paid for.
                raise ValueError(f"{label}[{i}] needs a non-empty 'answer'")


async def _emit(on_progress: ProgressFn | None, event: dict) -> None:
    """Publish a progress event; a failing listener must not stop a paid run."""
    if on_progress is None:
        return
    try:
        outcome = on_progress({"type": "progress", **event})
        if inspect.isawaitable(outcome):
            await outcome
    except Exception as e:  # noqa: BLE001
        logger.warning("⚠️ Evolution progress listener failed: %s", e)


def _as_dict(result: Any) -> dict:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return dict(result)


async def evolve(
    name: str,
    *,
    train: Sequence[Mapping[str, Any]],
    val: Sequence[Mapping[str, Any]],
    rounds: int,
    batch_size: int,
    mode: str = "static",
    metric: str = "f1",
    guidance: str = "raw",
    max_llm_calls: int,
    on_progress: ProgressFn | None = None,
    run: RunFn = run_agent,
    refiner_llm: Any = None,
    max_steps: int | None = None,
    initial: ProceduralGraph | None = None,
    replace: bool = False,
) -> dict:
    """Run Algorithm 1 on graph ``name``; return the report.

    QA items are ``{"question", "answer"}`` (``answer`` may be a list of
    acceptable answers). ``run`` is called as ``run(question, graph_name=,
    graph=, graph_version=, guidance=, max_steps=)`` and must return an
    ``AgentResult`` or its dict; tests inject a fake. ``refiner_llm``
    defaults to ``get_chat_llm(temperature=0, json_mode=True)``. ``initial``
    replaces the stored graph as G0 in static mode; the bench uses it to
    evolve a copy of the prior under another name. ``replace`` only matters
    in scratch mode: it allows a run on a name that already holds a graph,
    which is then scored once on ``val`` and sets the acceptance floor.

    Raises, before any LLM call: ``ValueError`` for invalid inputs,
    ``EvolutionBudgetTooSmall`` (a ``ValueError``) when ``max_llm_calls``
    cannot pay for the baseline and one full round, ``ProceduralGraphNotFound``
    in static mode without a stored graph, ``ProceduralGraphExists`` in
    scratch mode on a stored graph without ``replace``.

    Report: ``{graph, mode, metric, guidance, replace, rounds_requested,
    rounds_run, accepted_rounds, stopped ("completed" | "budget"),
    baseline_score, stored_score, stored_version, final_score, final_version,
    rounds: [{round, train_mean, candidate_score, floor, accepted, reason,
    diagnostics, notes, diff, version}], llm_calls, max_llm_calls,
    min_llm_calls, usage, effect_floor, note}``. ``stored_score`` is the
    replaced graph's validation score (scratch + replace), else ``None``;
    ``floor`` is the score a round's candidate had to reach. A round's
    ``reason`` is ``accepted``, ``score``, ``structural``, ``no_change``,
    ``refiner_error`` or ``budget``.
    Only a round with a decision counts in ``rounds_run``.
    """
    settings = get_settings()
    _validate_inputs(train, val, rounds, batch_size, mode, metric, guidance, max_llm_calls)
    steps_cap = settings.agent_max_steps if max_steps is None else max(1, int(max_steps))
    per_rollout = rollout_worst_case(steps_cap, guidance)
    budget = _Budget(max_llm_calls)
    usage = {"input_tokens": 0, "output_tokens": 0, "estimated": False}
    trace_chars = int(settings.evolution_trajectory_max_chars)

    # The graph scratch mode would overwrite, when ``replace`` allows it.
    replaced: tuple[ProceduralGraph, dict] | None = None
    if mode == "scratch":
        existing = await store.load_graph_with_meta(name)
        if existing is not None and not replace:
            raise ProceduralGraphExists(name, existing[1].get("version"))
        replaced = existing
        current = skeleton(name, tools=TOOL_NAMES, description=TASK_DESCRIPTION)
        version, stored = None, False
    elif initial is not None:
        current = initial.copy()
        current.name = name
        version, stored = None, False
    else:
        loaded = await store.load_graph_with_meta(name)
        if loaded is None:
            raise ProceduralGraphNotFound(name)
        current, meta = loaded
        version, stored = meta.get("version"), True

    # Refuse before the first call: a baseline alone decides nothing, so the
    # cap must pay for it (and the replaced graph's score) plus one whole round.
    min_calls = check_budget(
        max_llm_calls,
        train_size=len(train),
        val_size=len(val),
        batch_size=batch_size,
        per_rollout=per_rollout,
        stored_eval=replaced is not None,
    )

    report: dict[str, Any] = {
        "graph": name,
        "mode": mode,
        "metric": metric,
        "guidance": guidance,
        "replace": bool(replace),
        "rounds_requested": int(rounds),
        "rounds_run": 0,
        "accepted_rounds": 0,
        "stopped": "completed",
        "baseline_score": None,
        "stored_score": None,
        "stored_version": None,
        "final_score": None,
        "final_version": version if stored else None,
        "rounds": [],
        "llm_calls": 0,
        "max_llm_calls": int(max_llm_calls),
        "min_llm_calls": min_calls,
        "usage": usage,
        "effect_floor": 1.0 / len(val),
        "note": SIGNIFICANCE_NOTE,
    }

    async def evaluate(
        graph: ProceduralGraph,
        graph_version: int | None,
        items: Sequence[Mapping[str, Any]],
        *,
        stage: str,
        round_number: int,
        source: str,
    ) -> tuple[float, list[dict]]:
        """Roll the agent out on ``items`` with ``graph`` frozen; mean score + traces."""
        rollouts = []
        for position, item in enumerate(items, start=1):
            budget.require(per_rollout)
            result = _as_dict(
                await run(
                    item["question"],
                    graph_name=name,
                    graph=graph,
                    graph_version=graph_version,
                    guidance=guidance,
                    max_steps=steps_cap,
                )
            )
            run_usage = result.get("usage") or {}
            budget.spend(int(run_usage.get("llm_calls", 0)))
            usage["input_tokens"] += int(run_usage.get("input_tokens", 0))
            usage["output_tokens"] += int(run_usage.get("output_tokens", 0))
            usage["estimated"] = usage["estimated"] or bool(run_usage.get("estimated"))
            value = qa_score(result.get("answer"), item["answer"], metric)
            rollouts.append(
                {
                    "question": item["question"],
                    "gold": item["answer"],
                    "score": value,
                    "result": result,
                }
            )
            try:
                await store.record_trajectory(
                    name,
                    version=graph_version,
                    query=item["question"],
                    steps=list(result.get("steps") or []),
                    score=value,
                    source=source,
                )
            except Exception as e:  # noqa: BLE001 - an audit write must not sink the run
                logger.warning("⚠️ Could not record an evolution trajectory: %s", e)
            await _emit(
                on_progress,
                {
                    "stage": stage,
                    "round": round_number,
                    "processed": position,
                    "total": len(items),
                    "llm_calls": budget.used,
                },
            )
        mean = sum(r["score"] for r in rollouts) / len(rollouts)
        return mean, rollouts

    # Line 1: S0 ← Evaluate(G0, D_val). ``check_budget`` covered it; the guard
    # below only matters for a ``run`` reporting more calls than its bound.
    stored_score: float | None = None
    try:
        budget.require(len(val) * per_rollout)
        retained_score, _ = await evaluate(
            current, version, val, stage="baseline", round_number=0, source="evolution-val"
        )
        report["baseline_score"] = report["final_score"] = retained_score
        await _emit(on_progress, {"stage": "baseline", "round": 0, "score": retained_score})
        if replaced is not None:
            # Synapse's guard: score the graph a scratch run may overwrite, ONCE,
            # on the same D_val. Its score joins the acceptance floor below.
            stored_graph, stored_meta = replaced
            stored_version = stored_meta.get("version")
            budget.require(len(val) * per_rollout)
            stored_score, _ = await evaluate(
                stored_graph,
                stored_version,
                val,
                stage="stored",
                round_number=0,
                source="evolution-val",
            )
            report.update(stored_score=stored_score, stored_version=stored_version)
            await _emit(
                on_progress,
                {"stage": "stored", "round": 0, "score": stored_score, "version": stored_version},
            )
    except _BudgetExhausted:
        report.update(stopped="budget", llm_calls=budget.used)
        return report

    rejected: list[dict] = []
    model = refiner_llm
    for k in range(1, int(rounds) + 1):
        entry: dict[str, Any] = {
            "round": k,
            "train_mean": None,
            "candidate_score": None,
            "floor": None,
            "accepted": False,
            "reason": None,
            "diagnostics": [],
            "notes": [],
            "diff": None,
            "version": None,
        }
        report["rounds"].append(entry)
        try:
            # Lines 5-6: roll the retained graph out on the next stride of D_train.
            batch = training_batch(train, k, batch_size)
            # The round's worst case must fit before its first rollout: a round
            # decides nothing until its validation ends, so a partial one would
            # pay for a batch, a refiner call and some validation for nothing.
            budget.require((len(batch) + len(val)) * per_rollout + 1)
            train_mean, rollouts = await evaluate(
                current, version, batch, stage="rollout", round_number=k, source="evolution"
            )
            entry["train_mean"] = train_mean
            # Lines 7-9: the refiner call.
            budget.require(1)
            await _emit(on_progress, {"stage": "refine", "round": k, "llm_calls": budget.used})
            prompt = build_refiner_prompt(
                current,
                mode=mode,
                metric=metric,
                rollouts=rollouts,
                rejected=rejected,
                max_chars=trace_chars,
            )
            if model is None:
                model = get_chat_llm(temperature=0, json_mode=True)
            try:
                response = await model.ainvoke([HumanMessage(content=prompt)])
            except ProviderConfigError:
                raise
            except SpendCapReached as e:
                # The caller's cap refused the call before it was made: that is
                # the budget running out, not a refiner failure to retry.
                logger.info("Refiner call refused by a spend cap in round %s: %s", k, e)
                raise _BudgetExhausted from e
            except Exception as e:  # noqa: BLE001 - not a candidate; try again next round
                budget.spend(1)  # it was attempted, and may have been billed
                logger.warning("⚠️ Refiner call failed in round %s: %s", k, e)
                entry.update(reason="refiner_error", diagnostics=[f"{type(e).__name__}: {e}"])
                await _record_rejection(name, k, "refiner_error", None, entry["diagnostics"], None)
                report["rounds_run"] += 1
                await _emit(
                    on_progress,
                    {
                        "stage": "rejected",
                        "round": k,
                        "reason": "refiner_error",
                        "diagnostics": entry["diagnostics"],
                    },
                )
                continue
            budget.spend(1)
            call_usage = llm_usage(response, prompt)
            usage["input_tokens"] += call_usage["input_tokens"]
            usage["output_tokens"] += call_usage["output_tokens"]
            usage["estimated"] = usage["estimated"] or call_usage["estimated"]

            # Line 10: PrepareCandidate.
            edits, parse_error = extract_edits(response_text(response))
            if parse_error:
                candidate, diagnostics, notes = None, [parse_error], []
            else:
                candidate, diagnostics, notes = prepare_candidate(current, edits)
            entry["notes"] = notes

            # Lines 11-13: structural failure → rejection memory, no validation rollout.
            if diagnostics or candidate is None:
                entry.update(reason="structural", diagnostics=list(diagnostics))
                rejected.append(
                    {
                        "round": k,
                        "reason": "structural",
                        "edits": edits,
                        "diagnostics": list(diagnostics),
                        # E_k stays in the record (Algorithm 1 line 12); the
                        # strict PrepareCandidate leaves no G_cand to keep.
                        "traces": rollouts,
                    }
                )
                await _record_rejection(name, k, "structural", None, diagnostics, edits)
                report["rounds_run"] += 1
                await _emit(
                    on_progress,
                    {
                        "stage": "rejected",
                        "round": k,
                        "reason": "structural",
                        "diagnostics": list(diagnostics),
                    },
                )
                continue

            diff = graph_diff(current, candidate)
            entry["diff"] = diff
            if not any(diff.values()):
                # Synapse addition: a candidate identical to G_{k-1} would tie
                # and be "accepted" as an unchanged graph, after |val| paid
                # rollouts. Skip the rollout, keep G_{k-1}, and tell the refiner.
                entry["reason"] = "no_change"
                rejected.append({"round": k, "reason": "no_change", "edits": edits})
                await _record_rejection(name, k, "no_change", None, [], edits)
                report["rounds_run"] += 1
                await _emit(on_progress, {"stage": "rejected", "round": k, "reason": "no_change"})
                continue

            # Line 15: S_cand ← Evaluate(G_cand, D_val).
            candidate_score, _ = await evaluate(
                candidate, None, val, stage="validate", round_number=k, source="evolution-val"
            )
        except _BudgetExhausted:
            entry["reason"] = "budget"
            report["stopped"] = "budget"
            break
        entry["candidate_score"] = candidate_score
        report["rounds_run"] += 1

        # Lines 16-20: accept iff S_cand ≥ S_{k-1} (ties accepted). When scratch
        # mode may overwrite a stored graph, the floor is max(S_{k-1}, S_stored):
        # never replace a better graph. After an acceptance S_{k-1} ≥ S_stored,
        # so from then on the floor is the paper's again.
        floor, floor_source = retained_score, "retained"
        if stored_score is not None and stored_score > retained_score:
            floor, floor_source = stored_score, "stored"
        entry["floor"] = floor
        if candidate_score >= floor:
            version = await store.save_graph(
                candidate,
                score=candidate_score,
                note=f"evolution round {k}",
                edits=edits,
                previous=current if stored else None,
            )
            previous_score = retained_score
            current, retained_score, stored = candidate, candidate_score, True
            entry.update(accepted=True, reason="accepted", version=version)
            report["accepted_rounds"] += 1
            report["final_version"] = version
            await _emit(
                on_progress,
                {
                    "stage": "accepted",
                    "round": k,
                    "score": candidate_score,
                    "previous_score": previous_score,
                    "version": version,
                    "change": summarize_diff(diff),
                },
            )
        else:
            entry["reason"] = "score"
            rejected.append(
                {
                    "round": k,
                    "reason": "score",
                    "edits": edits,
                    "score": candidate_score,
                    "retained_score": floor,
                    "floor": floor_source,
                    "summary": summarize_diff(diff),
                    # (ΔG_k, G_cand, E_k, S_cand): Algorithm 1 line 19.
                    "candidate": candidate,
                    "traces": rollouts,
                }
            )
            await _record_rejection(name, k, "score", candidate_score, [], edits)
            event = {
                "stage": "rejected",
                "round": k,
                "reason": "score",
                "score": candidate_score,
                "retained_score": floor,
            }
            if floor_source == "stored":
                event["floor"] = "stored"
            await _emit(on_progress, event)

    report["final_score"] = retained_score
    report["llm_calls"] = budget.used
    return report


async def _record_rejection(
    name: str,
    round_number: int,
    reason: str,
    score: float | None,
    diagnostics: Sequence[str],
    edits: Any,
) -> None:
    try:
        await store.record_rejection(
            name,
            round=round_number,
            reason=reason,
            score=score,
            diagnostics=list(diagnostics),
            edits=edits,
        )
    except Exception as e:  # noqa: BLE001 - the in-memory rejection memory still has it
        logger.warning("⚠️ Could not record a rejected candidate: %s", e)
