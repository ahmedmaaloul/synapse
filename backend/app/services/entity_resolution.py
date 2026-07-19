# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Ahmed Maaloul <maaloulahmed25@gmail.com>
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

Public API (other services code against these exact signatures):
    find_duplicate_clusters(entities, embeddings, settings=None) -> list[list[str]]
    choose_canonical(names) -> str
    await merge_entity_clusters(clusters) -> {"clusters": int, "merged": int}
"""

from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher

from app import neo4j_driver
from app.config import Settings, get_settings

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
    """Cosine similarity of two equal-length vectors (0.0 if either is degenerate)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))


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


async def fetch_graph_entities(
    limit: int = MAX_GRAPH_CANDIDATES,
) -> tuple[list[dict], dict[str, list[float]]]:
    """Load entities already in Neo4j as ``(entities, embeddings_by_name)``.

    Nodes stored without an embedding are still returned (their type is needed as
    a merge barrier) but are simply absent from the embeddings map, which makes
    them ineligible for merging.
    """
    try:
        rows = await neo4j_driver.execute_query(_FETCH_GRAPH_ENTITIES, {"limit": limit}) or []
    except Exception as e:  # noqa: BLE001 - resolution is best-effort
        logger.warning("⚠️ Could not load existing entities for resolution: %s", e)
        return [], {}

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
