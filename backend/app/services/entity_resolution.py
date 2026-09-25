# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Synapse — Entity Resolution

LLM extraction is *locally* consistent but *globally* sloppy: chunk 1 says
"Ahmed", chunk 7 says "Ahmed Maaloul"; one paragraph writes "Postgres", the next
"PostgreSQL". Left alone the graph fills with near-duplicate nodes, which
fragments every neighbourhood the retriever walks and is the single most visible
quality flaw in the product.

This module collapses those duplicates into one canonical node.

The decision rule is deliberately simple and fully explainable — two entities are
the same only when **both** independent signals agree:

  1. **semantic** — cosine similarity of their embeddings is
     ``>= entity_resolution_threshold`` (default 0.93, deliberately strict: a
     wrong merge destroys information, a missed merge merely leaves a duplicate).
  2. **lexical**  — a normalized string similarity is
     ``>= entity_resolution_name_threshold`` (default 0.87), computed with the
     stdlib :class:`difflib.SequenceMatcher`, where one name being a strict
     *prefix* or *token subsequence* of the other ("Ahmed" ⊂ "Ahmed Maaloul",
     "Postgres" ⊂ "PostgreSQL") counts as a strong match.

Two hard guards keep precision high:

  * entities of **different types never merge** — a PERSON is never a TOOL;
  * an entity with **no embedding never merges** — one signal is not a quorum.

Matching pairs are grouped transitively with union-find (A~B, B~C ⇒ {A, B, C}),
then :func:`choose_canonical` picks the most complete name, deterministically, so
repeated runs over the same corpus produce byte-identical graphs.

Candidate pairs are generated with classic **blocking** (see :func:`_block_keys`)
rather than a full O(n²) sweep, which is what keeps a whole-graph consolidation
pass affordable as the corpus grows.

Cross-document resolution: nearest neighbours, not the whole graph
------------------------------------------------------------------
Each ingest also resolves its *fresh* entities against the ones earlier
documents left in Neo4j. Reading the whole graph for that
(:func:`fetch_graph_entities`) makes every document cost O(graph size), which is
quadratic over a corpus: at 5,458 entities a one-paragraph document spent
16-35 s there, about 80% of it building Python dicts from the full result set.
:func:`fetch_candidate_entities` asks the entity vector index for the fresh
entities' nearest neighbours instead, so a document costs O(fresh × k),
whatever the size of the graph. The one thing that can raise it is a crowded
neighbourhood, where more than k entities clear the merge threshold around one
new entity (see 2 below). That probe then costs as many rows as there are
entities in its neighbourhood, up to :data:`MAX_CANDIDATE_K`.

**Why the result is the same as the full scan.** Call a pair the rule above
accepts an *edge*. The reference is the *uncapped* full scan: it clusters every
entity in the graph, with this document's vectors overriding the stored ones,
and keeps the clusters that contain a fresh entity. (The scan the code
actually ships with, :func:`fetch_graph_entities`, reads at most
:data:`MAX_GRAPH_CANDIDATES` entities. Below that size the two are the same,
and above it the capped scan is the one that misses merges; see the end of this
section.) The candidate path returns the same clusters because:

  1. *Edges are pairwise.* Whether (a, b) is an edge depends on a and b alone
     (types, block keys, names, vectors). So clustering any subset S of the
     graph finds exactly the full graph's edges inside S, and a cluster of S
     never reaches outside the full cluster that contains it.
  2. *An edge needs cosine >= threshold.* A cosine vector index scores a
     neighbour (1 + cos) / 2 (see :func:`vector_index_score_floor`). An exact
     top-k query with the matching score floor therefore returns every entity
     that could form an edge with the probe, except when its k answers all
     clear the floor, since more may lie beyond them. k counts *every* node
     above the floor, including this document's own entities and entities an
     earlier round already fetched. The query drops those only after ranking,
     so they take up places too. The query reports each probe whose answers
     all cleared the floor, counted before that filter, and
     :func:`fetch_candidate_entities` asks those probes again with twice the
     k until an answer ends below the floor. So an exact index returns every
     such entity, however crowded the neighbourhood, up to
     :data:`MAX_CANDIDATE_K` (see (a) below). This holds for the vectors the
     index stores; 4 covers the rest.
  3. *The candidates are grown breadth-first until nothing changes.* The fresh
     entities probe the index. Every entity that joins a cluster touching a
     fresh entity then probes it too, until a round adds nobody. Take any x in
     a full-scan cluster of a fresh f, and a path of edges f = p0, ..., pm = x.
     Suppose some p_i were never fetched, and take the first one. Then
     p_0 ... p_(i-1) were fetched, so by 1 the path edges between them were
     found, so p_(i-1) sits in a cluster touching f. At the fixpoint it has
     therefore probed, with a usable vector (it has an edge). By 2 its probe
     returned p_i, or by 4 p_i was fetched anyway. That is a contradiction.
     Every full-scan cluster that touches a fresh entity is therefore found
     whole, and by 1 it is found with nothing extra. The outputs are identical.
  4. *No entity whose vector here differs from the index's copy is left to
     the index.* The fresh entities are always candidates. An alias this
     document folded in memory takes this document's vector in the full scan,
     but the index may hold an older node of that name under another vector,
     so aliases are fetched by exact name. So is a fresh name this document
     has no vector for, because it keeps its stored vector, as it did in the
     full scan.

Existing entities that match each other were normally merged by the ingest
that wrote the second of them. So in practice step 3 stops after one extra
query (and only on a document that merged something). Step 3 is what keeps the
result exact when that merge did not happen, for example a merge held back by
an unsafe relationship type (see :func:`_plain_merge`), resolution switched off
at the time, or a threshold lowered since.

**Where the two can still differ.** Step 2 needs the index to return *every*
neighbour above the floor. It may not, in two cases:

  (a) More than :data:`MAX_CANDIDATE_K` (5,000, or k if that is larger)
      nodes clear cosine >= 0.93 against one probe. The doubling stops there,
      so one probe never reads more rows than the whole-graph scan does, and
      :func:`fetch_candidate_entities` logs a warning that names the probe.
      The crowd need not be duplicates: model versions with near-identical
      descriptions can fill a neighbourhood without passing the name gate.
      Between k and the ceiling a crowded probe is exact but costs more, one
      extra query per doubling, each returning only the new rows.
  (b) The HNSW index's recall is approximate (Neo4j's docs say so for
      ``db.index.vector.queryNodes``). HNSW misses sit mostly deep in the
      ranking, and a duplicate at cosine >= 0.93 sits at the top of it.

Either case can only leave an entity *out* of a cluster. By 1, every cluster
the candidate path finds lies inside a full-scan cluster, and the merge
decision is still the Python rule above, run on fetched vectors. So the
candidate path never merges two entities that the full scan would keep apart.
When only part of a cluster is found, though, the surviving name can differ,
since it is chosen among fewer members, and the missed members stay behind
as duplicates.

The shipped full scan errs in the opposite direction. It reads at most
:data:`MAX_GRAPH_CANDIDATES` entities, so on a larger graph it never compares
the rest (the 5,458-entity graph above was already past it), and it logs a
warning each time it stops short. That scan is what the fallback and k = 0
run. The candidate path has no such cap, so on a graph that size it finds
merges the capped scan misses.

Public API (other services code against these exact signatures):
    find_duplicate_clusters(entities, embeddings, settings=None) -> list[list[str]]
    choose_canonical(names) -> str
    await merge_entity_clusters(clusters) -> {"clusters": int, "merged": int}
    await fetch_graph_entities(limit) -> (entities, embeddings)
    await fetch_candidate_entities(fresh_names, fresh_embeddings, k, ...) -> (entities, embeddings)
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher

from app import neo4j_driver
from app.config import Settings, get_settings
from app.services.graph_schema import ENTITY_VECTOR_INDEX

logger = logging.getLogger(__name__)

#: Lexical score awarded when one name is a strict prefix / token subsequence of
#: the other. Above every sane ``entity_resolution_name_threshold`` yet below 1.0
#: so an exact match still ranks higher.
CONTAINMENT_SCORE = 0.95

#: A containment match needs at least this many characters, so "AI" does not
#: latch onto every name starting with those letters.
MIN_CONTAINMENT_CHARS = 3

#: For *single-token* prefixes ("postgres" ⊂ "postgresql") the shorter name must
#: cover at least this fraction of the longer one. Blocks the classic
#: "Java" / "JavaScript" false positive (0.4) while keeping
#: "Postgres" / "PostgreSQL" (0.8). Multi-token names are exempt: dropping a
#: surname ("Ahmed" for "Ahmed Maaloul") is a normal human abbreviation.
PREFIX_MIN_LENGTH_RATIO = 0.6

#: Upper bound on entities pulled from Neo4j for a whole-graph consolidation
#: pass, so ingestion time stays predictable on a large corpus.
MAX_GRAPH_CANDIDATES = 5000

#: Cosine slack below ``entity_resolution_threshold`` when it becomes a vector
#: index score floor. Lucene scores in float32, so a pair sitting exactly on the
#: threshold in Python's float64 can score a hair under it in the index. The
#: slack only widens the candidate set: the merge decision is still
#: :func:`find_duplicate_clusters` on the fetched vectors.
CANDIDATE_COSINE_MARGIN = 1e-4

#: How far :func:`fetch_candidate_entities` doubles k for a probe whose answers
#: all clear the score floor. The configured k raises it if k is larger. It
#: equals :data:`MAX_GRAPH_CANDIDATES`, so one crowded probe never reads more
#: rows than the whole-graph scan. Past it a warning is logged, and a duplicate
#: ranked beyond it can be missed (residual (a) in the module docstring).
MAX_CANDIDATE_K = MAX_GRAPH_CANDIDATES

#: Relationship types are interpolated into Cypher for the no-APOC fallback
#: (Neo4j < 5.26 cannot parameterize a type), so they are strictly validated.
_SAFE_REL_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


# ── Normalization & similarity ───────────────────────────
def _normalize(name: str) -> str:
    """Lowercase, strip accents, punctuation and *all* whitespace.

    ``"Ahmed  Maaloul!"`` and ``"ahmed-maaloul"`` both become ``"ahmedmaaloul"``.
    """
    if not name:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_ish = "".join(c for c in decomposed if not unicodedata.combining(c))
    stripped = _PUNCT.sub(" ", ascii_ish.lower())
    return _SPACE.sub("", stripped).strip()


def _tokens(name: str) -> list[str]:
    """Lowercased, punctuation-free word tokens (``"Ahmed Maaloul"`` -> two)."""
    if not name:
        return []
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_ish = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _SPACE.sub(" ", _PUNCT.sub(" ", ascii_ish.lower())).split()


def _is_token_subsequence(short: list[str], long: list[str]) -> bool:
    """True if every token of ``short`` appears in ``long``, in order."""
    if not short or len(short) >= len(long):
        return False
    it = iter(long)
    return all(token in it for token in short)


def _is_containment(a: str, b: str) -> bool:
    """True if one name is a strict prefix / token subsequence of the other."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb or na == nb:
        return False

    short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(short) < MIN_CONTAINMENT_CHARS:
        return False

    ta, tb = _tokens(a), _tokens(b)
    short_tokens, long_tokens = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if _is_token_subsequence(short_tokens, long_tokens):
        return True

    # Single-token prefix ("postgres" ⊂ "postgresql") needs enough coverage.
    return long.startswith(short) and len(short) / len(long) >= PREFIX_MIN_LENGTH_RATIO


def name_similarity(a: str, b: str) -> float:
    """Normalized lexical similarity of two entity names, in ``[0, 1]``.

    ``difflib.SequenceMatcher`` on the normalized forms, lifted to
    :data:`CONTAINMENT_SCORE` when one name is a prefix / token subsequence of
    the other. Symmetric and deterministic — no fuzzy-matching dependency.
    """
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    if _is_containment(a, b):
        return max(ratio, CONTAINMENT_SCORE)
    return ratio


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is degenerate).

    Degenerate covers a NaN or infinite component too. A NaN result would slip
    through the ``cosine < threshold`` gate in :func:`find_duplicate_clusters`
    (every comparison with NaN is false), so a broken vector would let the name
    signal merge on its own.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    similarity = dot / ((norm_a**0.5) * (norm_b**0.5))
    return similarity if math.isfinite(similarity) else 0.0


def _entity_type(entity: dict) -> str:
    """Normalized entity type; missing/blank types collapse to ``UNKNOWN``."""
    return str(entity.get("type") or "UNKNOWN").strip().upper() or "UNKNOWN"


def _block_keys(name: str) -> set[str]:
    """Blocking keys for a name: the initial of the whole name and of each token.

    ``"Ahmed Maaloul"`` -> ``{"a", "m"}``. Two names only become candidates when
    they share a key, which turns the O(n²) all-pairs sweep into a handful of
    small buckets while still catching reordered names ("Maaloul Ahmed").
    """
    keys = {t[0] for t in _tokens(name) if t}
    normalized = _normalize(name)
    if normalized:
        keys.add(normalized[0])
    return keys


# ── Union-find ───────────────────────────────────────────
class _UnionFind:
    """Minimal disjoint-set forest used to group matching pairs transitively."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self._parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # path compression
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            # Deterministic merge direction: lexicographically smaller root wins.
            low, high = sorted((root_a, root_b))
            self._parent[high] = low

    def groups(self) -> list[list[str]]:
        clusters: dict[str, list[str]] = {}
        for item in self._parent:
            clusters.setdefault(self.find(item), []).append(item)
        return [sorted(members) for members in clusters.values() if len(members) > 1]


# ── Public: clustering ───────────────────────────────────
def find_duplicate_clusters(
    entities: list[dict],
    embeddings: dict[str, list[float]],
    settings: Settings | None = None,
) -> list[list[str]]:
    """Group near-duplicate entity **names** into clusters. Pure and synchronous.

    Args:
        entities: dicts with at least ``name``; ``type`` is honoured as a hard
            barrier (entities of different types are never grouped).
        embeddings: map of entity name -> embedding vector. An entity missing
            from this map can never merge (the semantic signal is unavailable).
        settings: overrides the app settings, for tests and callers that already
            hold one.

    Returns:
        One list per cluster, each with **2 or more** names and the canonical
        name (see :func:`choose_canonical`) first, remaining names sorted
        alphabetically. Clusters themselves are sorted by canonical name, so the
        output is fully deterministic.
    """
    settings = settings or get_settings()
    vector_threshold = settings.entity_resolution_threshold
    name_threshold = settings.entity_resolution_name_threshold

    # Collapse to one record per name; keep the first type seen for stability.
    # Names are kept verbatim (not stripped) because they are the node keys the
    # merge queries and the embeddings map are addressed by.
    records: dict[str, str] = {}
    for entity in entities or []:
        name = str(entity.get("name", ""))
        if name.strip() and name not in records:
            records[name] = _entity_type(entity)

    # Block by (type, initial) so we only score plausible pairs.
    blocks: dict[tuple[str, str], list[str]] = {}
    for name, entity_type in records.items():
        if name not in embeddings:
            continue  # no vector => no quorum => never a candidate
        for key in _block_keys(name):
            blocks.setdefault((entity_type, key), []).append(name)

    union = _UnionFind()
    scored: set[tuple[str, str]] = set()
    for members in blocks.values():
        members.sort()
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                pair = (left, right)
                if pair in scored:
                    continue
                scored.add(pair)
                if cosine_similarity(embeddings[left], embeddings[right]) < vector_threshold:
                    continue
                if name_similarity(left, right) < name_threshold:
                    continue
                logger.debug("🔗 Duplicate candidate: %r ≈ %r", left, right)
                union.union(left, right)

    clusters = []
    for members in union.groups():
        canonical = choose_canonical(members)
        clusters.append([canonical] + sorted(n for n in members if n != canonical))
    clusters.sort(key=lambda c: c[0])
    return clusters


# ── Public: canonical name choice ────────────────────────
def choose_canonical(names: list[str]) -> str:
    """Pick the best canonical name from a cluster. Pure and deterministic.

    Prefers the most *informative* name: longest first ("Ahmed Maaloul" beats
    "Ahmed", "PostgreSQL" beats "Postgres"), then the one with more tokens, then
    alphabetical — so the result never depends on input order.
    """
    candidates = [str(n) for n in names or [] if str(n).strip()]
    if not candidates:
        return ""
    return min(candidates, key=lambda n: (-len(n.strip()), -len(_tokens(n)), n))


# ── Public: Neo4j merge ──────────────────────────────────
# Copies a description onto the canonical node if it lacks one, and records the
# absorbed names (plus any aliases the duplicates had already collected) as an
# `aliases` list — which keeps "Postgres" searchable after it becomes
# "PostgreSQL". Plain Cypher: no APOC required.
_ABSORB_PROPERTIES = """
MATCH (c:Entity {name: $canonical})
OPTIONAL MATCH (d:Entity) WHERE d.name IN $dups
WITH c, collect(d) AS dups
WITH c,
     [x IN dups WHERE coalesce(x.description, '') <> '' | x.description] AS descriptions,
     reduce(acc = [], x IN dups | acc + coalesce(x.aliases, [])) AS inherited
WITH c, descriptions, [a IN inherited + $dups WHERE a <> $canonical] AS aliases
SET c.description = CASE
        WHEN coalesce(c.description, '') = '' AND size(descriptions) > 0
        THEN descriptions[0] ELSE c.description END,
    c.aliases = [a IN coalesce(c.aliases, []) WHERE NOT a IN aliases] + aliases
RETURN c.name AS name
"""

# APOC fast path: one call re-points every relationship and drops the duplicates.
# `properties: 'discard'` keeps the canonical node's own values (including the
# description/aliases just written above); `produceSelfRel: false` stops a
# duplicate's edge to the canonical from becoming a self-loop.
_APOC_MERGE = """
MATCH (c:Entity {name: $canonical})
MATCH (d:Entity) WHERE d.name IN $dups
WITH c, collect(d) AS dups
CALL apoc.refactor.mergeNodes([c] + dups,
     {properties: 'discard', mergeRels: true, produceSelfRel: false}) YIELD node
RETURN node.name AS name
"""

_COLLECT_REL_TYPES = """
MATCH (d:Entity)-[r]-(other)
WHERE d.name IN $dups
RETURN DISTINCT type(r) AS rel_type
"""

# Deleting the duplicates is guarded on the canonical node *existing*.
# Every rewire query below starts with `MATCH (c:Entity {name: $canonical})`,
# which — when the canonical node is absent — matches zero rows and returns
# quietly instead of raising. An unguarded `DETACH DELETE` would then wipe the
# duplicates *and every relationship they hold* after nothing had been
# re-pointed, silently destroying the whole cluster. The leading MATCH makes the
# delete a no-op in exactly that case, so the duplicates survive to be merged on
# a later pass.
_DELETE_DUPS = """
MATCH (c:Entity {name: $canonical})
WITH c
MATCH (d:Entity) WHERE d.name IN $dups
DETACH DELETE d
"""


def _rewire_queries(rel_type: str) -> tuple[str, str]:
    """Build the (outgoing, incoming) re-point queries for one relationship type.

    ``MERGE`` on the canonical side guarantees no duplicate parallel edges, and
    excluding every cluster member from the far side guarantees no self-loops.

    **Why the far side is null-safe.** ``other`` is deliberately unlabeled: a
    duplicate can be wired to nodes that are not ``:Entity`` — most visibly the
    ``(:Community)`` nodes that ``communities.py`` attaches with
    ``(:Entity)-[:IN_COMMUNITY]->(:Community)``, which carry no ``name``
    property. In Cypher ``null IN [...]`` is ``null``, ``NOT null`` is ``null``,
    and ``WHERE`` keeps only rows that are *true* — so the naive predicate
    ``NOT other.name IN $dups`` dropped **every** edge whose far endpoint lacks
    a name, and the ``DETACH DELETE`` that follows then destroyed it for good.
    ``coalesce(other.name, '')`` restores the intended meaning: a nameless node
    can never be a cluster member (members are matched by name, and blank names
    are filtered out before we get here), so it always passes the filter.

    The membership check is also label-aware — only an ``:Entity`` can be a
    cluster member — and the canonical node is excluded by *identity*
    (``other <> c``) rather than by name, which cannot go null either.

    **IN_COMMUNITY edges are re-pointed, not dropped.** The canonical node is
    the same real-world entity as the duplicate, so it belongs to the
    duplicate's community; ``MERGE`` collapses the two memberships into a single
    edge, and community detection rewrites these edges wholesale on its next run
    anyway. Re-pointing is therefore both correct and free, whereas dropping
    them silently loses graph structure.
    """
    far_side_filter = (
        "other <> c AND NOT (other:Entity AND coalesce(other.name, '') IN $dups)"
    )
    outgoing = f"""
    MATCH (c:Entity {{name: $canonical}})
    MATCH (d:Entity)-[r:{rel_type}]->(other)
    WHERE d.name IN $dups AND {far_side_filter}
    MERGE (c)-[nr:{rel_type}]->(other)
    SET nr.description = coalesce(nr.description, r.description),
        nr.type = coalesce(nr.type, r.type)
    """
    incoming = f"""
    MATCH (c:Entity {{name: $canonical}})
    MATCH (other)-[r:{rel_type}]->(d:Entity)
    WHERE d.name IN $dups AND {far_side_filter}
    MERGE (other)-[nr:{rel_type}]->(c)
    SET nr.description = coalesce(nr.description, r.description),
        nr.type = coalesce(nr.type, r.type)
    """
    return outgoing, incoming


async def _try_apoc_merge(canonical: str, dups: list[str]) -> bool:
    """Attempt the APOC one-shot merge. Returns False if APOC is unavailable."""
    try:
        await neo4j_driver.execute_query(_APOC_MERGE, {"canonical": canonical, "dups": dups})
        return True
    except Exception as e:  # noqa: BLE001 - APOC is optional by design
        logger.debug("APOC merge unavailable (%s); using plain-Cypher fallback", e)
        return False


async def _plain_merge(canonical: str, dups: list[str]) -> bool:
    """APOC-free fallback: re-point relationships type by type, then delete.

    Returns ``True`` when the duplicates were deleted, ``False`` when the delete
    was deliberately skipped because at least one relationship type could not be
    re-pointed.

    **Why an un-rewirable type must block the delete.** Neo4j < 5.26 cannot
    parameterize a relationship type, so the type is interpolated into the rewire
    query and therefore has to clear :data:`_SAFE_REL_TYPE` first — that guard is
    correct and stays. But *silently dropping* the types it rejects and then
    running ``DETACH DELETE`` anyway destroyed those edges permanently: they were
    never re-pointed onto the canonical node, and their only other endpoint was
    about to be deleted. Refusing to delete trades an invisible, unrecoverable
    data loss for a visible, fixable one — a duplicate node that a later pass (or
    a human who reads the warning below) can still resolve once the offending
    type is renamed.
    """
    params = {"canonical": canonical, "dups": dups}
    rows = await neo4j_driver.execute_query(_COLLECT_REL_TYPES, {"dups": dups}) or []
    found = {str(row.get("rel_type") or "") for row in rows}
    safe = sorted(t for t in found if _SAFE_REL_TYPE.match(t))
    unsafe = sorted(found.difference(safe))

    for rel_type in safe:
        outgoing, incoming = _rewire_queries(rel_type)
        await neo4j_driver.execute_query(outgoing, params)
        await neo4j_driver.execute_query(incoming, params)

    if unsafe:
        logger.warning(
            "⚠️ Keeping duplicate(s) %s of '%s': relationship type(s) %s cannot be "
            "safely interpolated into Cypher, so their edges were not re-pointed. "
            "Deleting the duplicates would destroy those edges, so the merge is "
            "left incomplete — rename the relationship type and re-run resolution.",
            dups,
            canonical,
            unsafe,
        )
        return False

    # Carries $canonical so the delete can be gated on the canonical existing.
    await neo4j_driver.execute_query(_DELETE_DUPS, params)
    return True


async def merge_entity_clusters(clusters: list[list[str]]) -> dict:
    """Collapse each cluster onto its canonical node in Neo4j.

    For every cluster: keep the canonical node, fill in a description it lacks,
    record the absorbed names under ``aliases``, re-point every incoming *and*
    outgoing relationship of each duplicate onto the canonical node (no
    self-loops, no duplicate parallel edges), then delete the duplicates.

    Uses ``apoc.refactor.mergeNodes`` when available and falls back to plain
    Cypher when it is not. A cluster that fails is logged and skipped — a bad
    merge must never abort an ingest. A cluster the fallback cannot fully rewire
    (see :func:`_plain_merge`) keeps its duplicates and is *not* counted, because
    those nodes are still in the graph.

    Returns:
        ``{"clusters": <clusters merged>, "merged": <duplicate nodes absorbed>}``
    """
    clusters_merged = 0
    nodes_merged = 0

    for cluster in clusters or []:
        names = [n for n in dict.fromkeys(str(x) for x in cluster) if n.strip()]
        if len(names) < 2:
            continue
        canonical = choose_canonical(names)
        dups = [n for n in names if n != canonical]
        if not dups:
            continue

        try:
            await neo4j_driver.execute_query(
                _ABSORB_PROPERTIES, {"canonical": canonical, "dups": dups}
            )
            if await _try_apoc_merge(canonical, dups):
                absorbed = True
            else:
                absorbed = await _plain_merge(canonical, dups)
        except Exception as e:  # noqa: BLE001 - one bad cluster must not fail the doc
            logger.warning("⚠️ Entity merge failed for %r: %s", canonical, e)
            continue

        if not absorbed:
            # _plain_merge already logged exactly why. The duplicates are still
            # in the graph, so counting them as merged would be a lie — and the
            # ingest response reports this number to the user.
            continue

        logger.info("🧬 Merged %s into '%s'", dups, canonical)
        clusters_merged += 1
        nodes_merged += len(dups)

    return {"clusters": clusters_merged, "merged": nodes_merged}


# ── Whole-graph consolidation ────────────────────────────
_FETCH_GRAPH_ENTITIES = """
MATCH (n:Entity)
RETURN n.name AS name, n.type AS type, n.embedding AS embedding
LIMIT $limit
"""


def _entities_from_rows(rows: list[dict]) -> tuple[list[dict], dict[str, list[float]]]:
    """``(name, type, embedding)`` rows -> ``(entities, embeddings_by_name)``.

    Shared by both fetches, so the full scan and the candidate path hand
    :func:`find_duplicate_clusters` data in exactly the same shape.
    """
    entities: list[dict] = []
    embeddings: dict[str, list[float]] = {}
    for row in rows:
        name = str(row.get("name") or "")
        if not name.strip():
            continue
        entities.append({"name": name, "type": row.get("type")})
        vector = row.get("embedding")
        if isinstance(vector, list) and vector:
            embeddings[name] = [float(v) for v in vector]
    return entities, embeddings


async def fetch_graph_entities(
    limit: int = MAX_GRAPH_CANDIDATES,
) -> tuple[list[dict], dict[str, list[float]]]:
    """Load entities already in Neo4j as ``(entities, embeddings_by_name)``.

    Nodes stored without an embedding are still returned (their type is needed as
    a merge barrier) but are simply absent from the embeddings map, which makes
    them ineligible for merging.

    This is the whole-graph read, O(graph size). Cross-document resolution
    uses :func:`fetch_candidate_entities` and falls back to this only when the
    vector index cannot answer.

    At most ``limit`` entities come back. One more row is requested, so a
    graph larger than ``limit`` is detected and logged at WARNING: the rest of
    it is not compared, and duplicates there are missed.
    """
    try:
        rows = await neo4j_driver.execute_query(_FETCH_GRAPH_ENTITIES, {"limit": limit + 1}) or []
    except Exception as e:  # noqa: BLE001 - resolution is best-effort
        logger.warning("⚠️ Could not load existing entities for resolution: %s", e)
        return [], {}
    if len(rows) > limit:
        logger.warning(
            "⚠️ Entity resolution read only the first %s entities of a larger graph; "
            "duplicates among the rest are not compared. The vector-index path "
            "(ENTITY_RESOLUTION_CANDIDATE_K > 0) has no such cap.",
            limit,
        )
        rows = rows[:limit]
    return _entities_from_rows(rows)


# ── Cross-document candidates (vector index) ─────────────
class CandidateSearchUnavailable(RuntimeError):
    """The entity vector index could not answer. Use :func:`fetch_graph_entities`."""


def vector_index_score_floor(threshold: float, margin: float = CANDIDATE_COSINE_MARGIN) -> float:
    """The ``db.index.vector.queryNodes`` score that matches ``cosine >= threshold``.

    A cosine vector index does not return the raw cosine. Neo4j's Cypher
    manual ("Vector indexes", section "Cosine and Euclidean similarity
    functions") defines the cosine score as ``(1 + cos(v, u)) / 2``, which maps
    [-1, 1] onto [0, 1]. This is Lucene's ``VectorSimilarityFunction.COSINE``,
    and :data:`graph_schema.ENTITY_VECTOR_INDEX` is created with
    ``vector.similarity_function: 'cosine'``. So ``cos >= t`` holds exactly when
    ``score >= (1 + t) / 2``: the default 0.93 becomes 0.965, not 0.93. Using
    the raw threshold as the floor would admit neighbours down to cosine 0.86.
    Checked on Neo4j 5.26 with ``vector.similarity.cosine``, which the manual
    says uses the index's function: orthogonal vectors score 0.5, opposite
    vectors 0.0, and cosine 0.93 scores 0.9650000333786011, a float32 value.

    ``margin`` lowers the cosine side slightly (see
    :data:`CANDIDATE_COSINE_MARGIN`), so float32 rounding in the index can only
    add a candidate, never drop one.
    """
    return min(1.0, max(0.0, (1.0 + threshold - margin) / 2.0))


def is_probe_vector(vector: object) -> bool:
    """True for a vector the index can be probed with: non-empty, finite, not all zero.

    A vector that fails this has cosine 0.0 with everything (see
    :func:`cosine_similarity`), so it can never form an edge. Skipping it loses
    nothing, and Neo4j would reject it anyway.
    """
    if not isinstance(vector, list | tuple) or not vector:
        return False
    try:
        values = [float(v) for v in vector]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(v) for v in values) and any(v != 0.0 for v in values)


# One round trip, whatever the number of probes: every probe's top-k comes back
# in the same query, deduplicated. `k` is the caller's k + 1 because a probe's
# own node is in the index and ranks first (see fetch_candidate_entities).
# `saturated` lists the probes (by index into $probes) whose k answers all
# cleared $min_score, so more may lie beyond them. They are counted *before*
# `$known` is applied, because known nodes take up places in the top k too.
# `$known` then stops the query from sending back nodes the caller already
# holds, the probes' own nodes included. `$names` fetches by exact name the
# entities whose stored vector cannot be trusted (point 4 of the module
# docstring). The answer is always exactly one row, even when both lists are
# empty or every hit is known, because an aggregation with no grouping key over
# zero rows still yields one row. Checked on Neo4j 5.26: with no probes and no
# names it returns [{saturated: [], entities: []}].
_FETCH_CANDIDATE_ENTITIES = """
UNWIND range(0, size($probes) - 1) AS i
CALL db.index.vector.queryNodes($index, $k, $probes[i]) YIELD node, score
    WHERE score >= $min_score
WITH i, collect(node) AS hits
UNWIND hits AS node
WITH collect(DISTINCT CASE WHEN size(hits) >= $k THEN i END) AS saturated,
     collect(DISTINCT CASE WHEN NOT node.name IN $known THEN node END) AS near
OPTIONAL MATCH (named:Entity) WHERE named.name IN $names
WITH saturated, near, collect(DISTINCT named) AS pinned
RETURN saturated,
       [n IN near + [p IN pinned WHERE NOT p IN near] | n {.name, .type, .embedding}] AS entities
"""


async def fetch_candidate_entities(
    fresh_names: Iterable[str],
    fresh_embeddings: Mapping[str, list[float]],
    k: int,
    *,
    settings: Settings | None = None,
    pinned_names: Iterable[str] = (),
    known_names: Iterable[str] = (),
) -> tuple[list[dict], dict[str, list[float]]]:
    """The entities that could merge with ``fresh_names``, as ``(entities, embeddings_by_name)``.

    This is the same shape as :func:`fetch_graph_entities`, but only for the
    fresh entities' neighbourhoods. For each name in ``fresh_names`` that has a
    usable vector in ``fresh_embeddings`` (see :func:`is_probe_vector`), the
    entity vector index returns its ``k`` nearest neighbours that score at
    least :func:`vector_index_score_floor` of ``entity_resolution_threshold``.
    All probes go in one batched ``UNWIND`` query, so it is one round trip and
    at most ``len(fresh_names) × k`` rows, however large the graph is, unless
    a neighbourhood is crowded (see below).

    ``pinned_names`` are fetched by exact name as well (point 4 of the module
    docstring). ``known_names`` are left out of the vector results, because the
    caller already holds them. The index is asked for ``k + 1`` neighbours so
    that a probe's own node, which always ranks first, does not use up one of
    the ``k`` places.

    A probe whose answers all clear the floor may have more neighbours above it
    than ``k`` let through, and known nodes count towards that. Those probes
    are asked again together, in one query, with twice the ``k`` and the rows
    already fetched marked known. This repeats until every answer ends below
    the floor, so an exact index yields the whole neighbourhood. It stops at
    :data:`MAX_CANDIDATE_K` (or ``k`` if larger) and logs a warning, because a
    duplicate ranked past that point can be missed.

    Raises:
        CandidateSearchUnavailable: a query failed. Typical causes are no
            vector index, a Neo4j build without the vector procedures, or
            vectors whose dimension does not match the index. Callers fall
            back to :func:`fetch_graph_entities`.
    """
    settings = settings or get_settings()
    probe_names = [
        name
        for name in dict.fromkeys(str(n) for n in fresh_names)
        if is_probe_vector(fresh_embeddings.get(name))
    ]
    names = sorted({str(n) for n in pinned_names if str(n).strip()})
    if not probe_names and not names:
        return [], {}

    k = max(1, int(k))
    ceiling = max(k, MAX_CANDIDATE_K)
    min_score = vector_index_score_floor(settings.entity_resolution_threshold)
    known = {str(n) for n in known_names}
    entities: list[dict] = []
    embeddings: dict[str, list[float]] = {}
    while True:
        params = {
            "index": ENTITY_VECTOR_INDEX,
            "k": k + 1,
            "probes": [[float(v) for v in fresh_embeddings[name]] for name in probe_names],
            "min_score": min_score,
            "known": sorted(known),
            "names": names,
        }
        try:
            rows = await neo4j_driver.execute_query(_FETCH_CANDIDATE_ENTITIES, params) or []
        except Exception as e:  # noqa: BLE001 - the caller owns the fallback
            raise CandidateSearchUnavailable(str(e)) from e
        row = rows[0] if rows else {}
        found, vectors = _entities_from_rows(row.get("entities") or [])
        entities += found
        embeddings.update(vectors)
        known.update(entity["name"] for entity in found)
        saturated = [
            probe_names[i]
            for i in sorted({int(index) for index in row.get("saturated") or []})
            if 0 <= i < len(probe_names)
        ]
        if not saturated:
            return entities, embeddings
        if k >= ceiling:
            logger.warning(
                "⚠️ Entity resolution: more than %s entities clear the merge threshold "
                "around %s, so a duplicate ranked past them can be missed. Raise "
                "ENTITY_RESOLUTION_CANDIDATE_K to look further.",
                k,
                saturated[:5],
            )
            return entities, embeddings
        k = min(2 * k, ceiling)
        logger.debug("Crowded neighbourhoods around %s; asking again with k=%s", saturated, k)
        probe_names, names = saturated, []
