# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""Tests for the Procedural Graph data structure (pure: no DB, no LLM).

The evolution loop's correctness rests on these rules — the paper's edit order,
"delete_edges removes every relation between two nodes", cycle repair before
validation, and the terminal-reachability check — so each is pinned here
against the behaviour described in Lu et al., arXiv:2609.09153, Appendix B.
"""

from __future__ import annotations

import dataclasses
import json
import re

import pytest

from app.services import procedural_graph as pg
from app.services.procedural_graph import (
    END,
    START,
    InvalidProceduralGraph,
    Localization,
    ProcEdge,
    ProceduralGraph,
    ProcNode,
)
from app.services.procedural_store import PRIORS_DIR


def _graph(edges, *, nodes=None, policy="forbid", tools=(), name="g") -> ProceduralGraph:
    """A graph from ``(source, target[, relation])`` tuples.

    Start/End are STATUS nodes; any other unlisted node is an ACTION.
    """
    node_types = dict(nodes or {})
    ordered = [START]
    for edge in edges:
        for node_id in edge[:2]:
            if node_id not in ordered:
                ordered.append(node_id)
    ordered += [n for n in node_types if n not in ordered]

    def node_type(node_id: str) -> str:
        return node_types.get(node_id, "STATUS" if node_id in (START, END) else "ACTION")

    return ProceduralGraph(
        name=name,
        nodes={n: ProcNode(n, node_type(n), f"{n} step") for n in ordered},
        edges=[
            ProcEdge(e[0], e[1], e[2] if len(e) > 2 else "LEADS_TO", guidance=f"go {e[1]}")
            for e in edges
        ],
        cycle_policy=policy,
        tools=tuple(tools),
    )


LINEAR = [(START, "search"), ("search", "read"), ("read", "answer"), ("answer", END)]


def _prior(name: str = "graphrag-navigator") -> ProceduralGraph:
    data = json.loads((PRIORS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return ProceduralGraph.from_dict(data)


# ── Vocabulary & model ───────────────────────────────
class TestVocabulary:
    def test_node_types_and_relations_are_the_papers(self):
        assert pg.NODE_TYPES == ("ACTION", "REASONING", "STATUS")
        assert pg.RELATIONS == ("LEADS_TO", "TRIGGERS", "PROVIDES_INPUT_FOR", "CONVERGES_TO")
        assert (START, END) == ("Start", "End")

    def test_nodes_and_edges_are_frozen_and_slotted(self):
        node = ProcNode("a", "ACTION")
        edge = ProcEdge("a", "b")
        with pytest.raises(dataclasses.FrozenInstanceError):
            node.id = "x"  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            edge.guidance = "x"  # type: ignore[misc]
        assert not hasattr(node, "__dict__") and not hasattr(edge, "__dict__")

    def test_edge_defaults(self):
        edge = ProcEdge("a", "b")
        assert (edge.relation, edge.condition, edge.guidance, edge.pitfalls) == (
            "LEADS_TO",
            None,
            "",
            "",
        )

    def test_graph_defaults_and_tools_become_a_tuple(self):
        graph = ProceduralGraph("g", {}, [], tools=["a", "b"])
        assert graph.cycle_policy == "forbid"
        assert graph.description == ""
        assert graph.tools == ("a", "b")


class TestGraphBasics:
    def test_out_edges_in_edge_order(self):
        graph = _graph([(START, "b"), ("x", "y"), (START, "a"), ("b", END), ("a", END), ("y", END)])
        assert [e.target for e in graph.out_edges(START)] == ["b", "a"]
        assert graph.out_edges("nope") == []

    def test_terminals_are_zero_out_degree_nodes_not_just_end(self):
        graph = _graph([(START, "a"), (START, "b")])
        assert graph.terminals() == ["a", "b"]

    def test_copy_is_independent(self):
        graph = _graph(LINEAR)
        clone = graph.copy()
        clone.nodes["new"] = ProcNode("new", "STATUS")
        clone.edges.append(ProcEdge("answer", "new"))
        assert "new" not in graph.nodes
        assert len(graph.edges) == len(LINEAR)
        assert clone != graph and graph.copy() == graph


class TestSkeleton:
    def test_is_start_to_end_only(self):
        graph = pg.skeleton("nav", tools=("search", "answer"), description="d")
        assert list(graph.nodes) == [START, END]
        assert all(n.type == "STATUS" for n in graph.nodes.values())
        assert graph.edges == [ProcEdge(START, END, "LEADS_TO")]
        assert graph.tools == ("search", "answer")
        assert graph.description == "d"

    def test_is_valid(self):
        assert pg.validate(pg.skeleton("nav")) == []


# ── JSON round trip ──────────────────────────────────
class TestJson:
    def test_to_dict_shape(self):
        graph = pg.skeleton("nav", tools=("a",))
        data = graph.to_dict()
        assert list(data) == ["name", "description", "cycle_policy", "tools", "nodes", "edges"]
        assert data["tools"] == ["a"]
        assert set(data["nodes"][0]) == {"id", "type", "description"}
        assert data["edges"][0] == {
            "source": START,
            "target": END,
            "relation": "LEADS_TO",
            "condition": None,
            "guidance": "",
            "pitfalls": "",
        }

    def test_round_trip_is_exact_and_json_safe(self):
        graph = _prior()
        again = ProceduralGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
        assert again == graph
        assert again.to_dict() == graph.to_dict()

    def test_defaults_for_missing_fields(self):
        graph = ProceduralGraph.from_dict(
            {"name": "g", "nodes": [{"id": "Start", "type": "STATUS"}], "edges": []}
        )
        assert graph.cycle_policy == "forbid"
        assert graph.tools == ()
        assert graph.nodes[START].description == ""

    def test_normalizes_case_and_whitespace(self):
        graph = ProceduralGraph.from_dict(
            {
                "name": " g ",
                "cycle_policy": " Allow ",
                "tools": ["a", " a ", "b"],
                "nodes": [{"id": " Start ", "type": "status"}, {"id": "a", "type": "Action"}],
                "edges": [
                    {"source": "Start", "target": "a", "relation": "provides input for"},
                    {"source": "Start", "target": "a", "relation": "converges-to", "condition": "null"},
                ],
            }
        )
        assert graph.name == "g"
        assert graph.cycle_policy == "allow"
        assert graph.tools == ("a", "b")
        assert graph.nodes[START].type == "STATUS" and graph.nodes["a"].type == "ACTION"
        assert [e.relation for e in graph.edges] == ["PROVIDES_INPUT_FOR", "CONVERGES_TO"]
        assert graph.edges[1].condition is None

    def test_missing_relation_defaults_to_leads_to(self):
        graph = ProceduralGraph.from_dict(
            {"name": "g", "nodes": [{"id": "Start", "type": "STATUS"}, {"id": "a", "type": "ACTION"}],
             "edges": [{"source": "Start", "target": "a"}]}
        )
        assert graph.edges[0].relation == "LEADS_TO"

    def test_name_override_wins_and_can_supply_a_missing_name(self):
        assert ProceduralGraph.from_dict({"name": "body"}, name="path").name == "path"
        assert ProceduralGraph.from_dict({"nodes": []}, name="path").name == "path"

    def test_unknown_top_level_keys_are_ignored(self):
        """GET returns to_dict() + version/score; that document must PUT back as-is."""
        data = pg.skeleton("nav").to_dict() | {"version": 3, "score": 0.5}
        assert ProceduralGraph.from_dict(data) == pg.skeleton("nav")

    def test_content_errors_are_left_to_validate(self):
        graph = ProceduralGraph.from_dict(
            {
                "name": "g",
                "nodes": [{"id": "Start", "type": "BOGUS"}],
                "edges": [{"source": "Start", "target": "ghost", "relation": "JUMPS_TO"}],
            }
        )
        problems = pg.validate(graph)
        assert any("invalid type 'BOGUS'" in p for p in problems)
        assert any("invalid relation 'JUMPS_TO'" in p for p in problems)
        assert any("missing node(s): ghost" in p for p in problems)

    @pytest.mark.parametrize(
        ("data", "fragment"),
        [
            ([], "must be an object"),
            ({"nodes": []}, "name' is missing"),
            ({"name": "g", "nodes": {}}, "'nodes' must be a list"),
            ({"name": "g", "edges": "x"}, "'edges' must be a list"),
            ({"name": "g", "nodes": ["Start"]}, "nodes[0] must be an object"),
            ({"name": "g", "nodes": [{"type": "STATUS"}]}, "nodes[0] has no 'id'"),
            ({"name": "g", "nodes": [{"id": "a"}, {"id": "a"}]}, "duplicate node id 'a'"),
            ({"name": "g", "edges": [{"source": "a"}]}, "needs non-empty 'source' and 'target'"),
            ({"name": "g", "edges": [{"source": "a", "target": "b", "guidance": {}}]}, "must be text"),
            ({"name": "g", "tools": "search"}, "'tools' must be a list"),
            ({"name": "g", "cycle_policy": 3}, "'cycle_policy' must be a string"),
        ],
    )
    def test_malformed_documents_raise_with_every_diagnostic(self, data, fragment):
        with pytest.raises(InvalidProceduralGraph) as err:
            ProceduralGraph.from_dict(data)
        assert any(fragment in d for d in err.value.diagnostics), err.value.diagnostics

    def test_invalid_graph_error_carries_all_diagnostics(self):
        err = InvalidProceduralGraph([f"p{i}" for i in range(7)])
        assert err.diagnostics == [f"p{i}" for i in range(7)]
        assert "+2 more" in str(err)
        assert isinstance(err, ValueError)


# ── Edits (PrepareCandidate's first half) ────────────
class TestApplyEdits:
    def test_input_graph_is_never_mutated(self):
        graph = _graph(LINEAR)
        before = graph.to_dict()
        pg.apply_edits(graph, {"delete_nodes": ["read"], "add_nodes": [{"id": "x", "type": "ACTION"}]})
        assert graph.to_dict() == before

    def test_delete_edges_removes_every_relation_between_the_endpoints(self):
        graph = _graph([(START, "a", "LEADS_TO"), (START, "a", "TRIGGERS"), ("a", END)])
        out, diags = pg.apply_edits(graph, {"delete_edges": [{"source": START, "target": "a", "relation": "TRIGGERS"}]})
        assert diags == []
        assert [e.key for e in out.edges] == [("a", "LEADS_TO", END)]

    def test_order_lets_a_refiner_keep_one_relation_by_re_adding_it(self):
        """delete_edges runs BEFORE add_edges (paper, Appendix B.5)."""
        graph = _graph([(START, "a", "LEADS_TO"), (START, "a", "TRIGGERS"), ("a", END)])
        edits = {
            "add_edges": [{"source": START, "target": "a", "relation": "TRIGGERS", "guidance": "kept"}],
            "delete_edges": [{"source": START, "target": "a"}],
        }
        out, diags = pg.apply_edits(graph, edits)
        assert diags == []
        start_edges = out.out_edges(START)
        assert [(e.relation, e.guidance) for e in start_edges] == [("TRIGGERS", "kept")]

    def test_nodes_are_deleted_before_they_are_added(self):
        """delete_nodes → add_nodes: a node can be re-created fresh in one response."""
        graph = _graph(LINEAR)
        out, diags = pg.apply_edits(
            graph,
            {
                "add_nodes": [{"id": "read", "type": "REASONING", "description": "new"}],
                "delete_nodes": ["read"],
                "add_edges": [
                    {"source": "search", "target": "read", "guidance": "g"},
                    {"source": "read", "target": "answer", "guidance": "g"},
                ],
            },
        )
        assert diags == []
        assert out.nodes["read"] == ProcNode("read", "REASONING", "new")
        assert pg.validate(out) == []

    def test_deleting_a_node_drops_its_incident_edges(self):
        out, diags = pg.apply_edits(_graph(LINEAR), {"delete_nodes": ["read"]})
        assert diags == []
        assert "read" not in out.nodes
        assert all("read" not in (e.source, e.target) for e in out.edges)

    def test_delete_nodes_accepts_objects_with_an_id(self):
        out, diags = pg.apply_edits(_graph(LINEAR), {"delete_nodes": [{"id": "read"}]})
        assert diags == [] and "read" not in out.nodes

    def test_add_nodes_upserts_an_existing_id(self):
        graph = _graph(LINEAR)
        out, diags = pg.apply_edits(graph, {"add_nodes": [{"id": "read", "description": "reworded"}]})
        assert diags == []
        assert out.nodes["read"] == ProcNode("read", "ACTION", "reworded")
        # Position (authoring order) is preserved by an upsert.
        assert list(out.nodes) == list(graph.nodes)

    def test_add_edges_with_an_existing_triplet_replaces_its_attributes(self):
        graph = _graph(LINEAR)
        out, diags = pg.apply_edits(
            graph, {"add_edges": [{"source": "search", "target": "read", "relation": "LEADS_TO", "guidance": "new"}]}
        )
        assert diags == []
        assert len(out.edges) == len(graph.edges)
        assert out.edges[1].guidance == "new"

    def test_new_edge_can_use_a_node_added_in_the_same_response(self):
        out, diags = pg.apply_edits(
            _graph(LINEAR),
            {
                "add_nodes": [{"id": "verify", "type": "reasoning"}],
                "add_edges": [
                    {"source": "read", "target": "verify", "relation": "leads_to", "guidance": "g"},
                    {"source": "verify", "target": "answer", "relation": "CONVERGES_TO", "guidance": "g"},
                ],
            },
        )
        assert diags == []
        assert out.nodes["verify"].type == "REASONING"
        assert pg.validate(out) == []

    def test_text_attributes_are_normalized(self):
        out, _ = pg.apply_edits(
            _graph(LINEAR),
            {
                "add_edges": [
                    {
                        "source": "search",
                        "target": "answer",
                        "relation": "TRIGGERS",
                        "condition": "  ",
                        "guidance": None,
                        "pitfalls": ["don't guess", "cite"],
                    }
                ]
            },
        )
        edge = out.edges[-1]
        assert (edge.condition, edge.guidance, edge.pitfalls) == (None, "", "don't guess; cite")

    @pytest.mark.parametrize(
        ("edits", "fragment"),
        [
            ("add a node", "edits must be a JSON object"),
            ({"add_nodes": "x"}, "'add_nodes' must be a list"),
            ({"rename_nodes": []}, "unknown edit key 'rename_nodes'"),
            ({"delete_edges": ["Start->a"]}, "delete_edges[0] must be an object"),
            ({"delete_edges": [{"source": "a"}]}, "delete_edges[0] needs non-empty"),
            ({"delete_nodes": [3]}, "delete_nodes[0] must be a node id"),
            ({"add_nodes": [["x"]]}, "add_nodes[0] must be an object"),
            ({"add_nodes": [{"type": "ACTION"}]}, "add_nodes[0] has no 'id'"),
            ({"add_nodes": [{"id": "x"}]}, "new node 'x' has no 'type'"),
            ({"add_nodes": [{"id": "x", "type": "TOOL"}]}, "invalid type 'TOOL'"),
            ({"add_nodes": [{"id": "x", "type": "ACTION", "description": {"a": 1}}]}, "description must be text"),
            ({"add_edges": [{"source": "search"}]}, "add_edges[0] needs non-empty"),
            ({"add_edges": [{"source": "search", "target": "read", "relation": "GOES"}]}, "invalid relation 'GOES'"),
            ({"add_edges": [{"source": "search", "target": "ghost"}]}, "missing node(s): ghost"),
        ],
    )
    def test_malformed_entries_are_reported_and_skipped(self, edits, fragment):
        graph = _graph(LINEAR)
        out, diags = pg.apply_edits(graph, edits)
        assert any(fragment in d for d in diags), diags
        assert out == graph  # the bad entry changed nothing

    def test_endpoint_deleted_in_the_same_response_is_missing(self):
        _, diags = pg.apply_edits(
            _graph(LINEAR),
            {"delete_nodes": ["read"], "add_edges": [{"source": "search", "target": "read"}]},
        )
        assert any("missing node(s): read" in d for d in diags)

    def test_good_entries_still_apply_next_to_bad_ones(self):
        out, diags = pg.apply_edits(
            _graph(LINEAR),
            {"add_nodes": [{"id": "ok", "type": "STATUS"}, {"id": "bad", "type": "NOPE"}]},
        )
        assert len(diags) == 1
        assert "ok" in out.nodes and "bad" not in out.nodes


# ── Cycles ───────────────────────────────────────────
class TestRepairCycles:
    def test_acyclic_graph_is_untouched(self):
        graph = _graph(LINEAR)
        repaired, removed = pg.repair_cycles(graph)
        assert removed == [] and repaired == graph

    def test_removes_the_edge_that_returns_toward_start(self):
        graph = _graph(LINEAR + [("answer", "search")])
        repaired, removed = pg.repair_cycles(graph)
        assert [(e.source, e.target) for e in removed] == [("answer", "search")]
        assert pg.find_cycle(repaired) is None
        assert pg.validate(repaired) == []

    def test_self_loop_is_a_back_edge(self):
        graph = _graph(LINEAR + [("read", "read")])
        _, removed = pg.repair_cycles(graph)
        assert [(e.source, e.target) for e in removed] == [("read", "read")]

    def test_dfs_is_rooted_at_start_regardless_of_edge_order(self):
        """Edge order must not decide WHICH edge is cut: DFS starts at Start."""
        edges = [("b", "a"), ("a", "b"), (START, "a"), ("b", END)]
        graph = _graph(edges)
        _, removed = pg.repair_cycles(graph)
        # From Start: Start→a→b, then b→a closes the cycle and is the one cut.
        assert [(e.source, e.target) for e in removed] == [("b", "a")]

    def test_cycles_unreachable_from_start_are_repaired_too(self):
        graph = _graph(LINEAR + [("x", "y"), ("y", "x"), ("y", END)])
        repaired, removed = pg.repair_cycles(graph)
        assert len(removed) == 1
        assert pg.find_cycle(repaired) is None

    def test_is_deterministic(self):
        graph = _graph(LINEAR + [("answer", "search"), ("read", START), ("answer", "read")])
        runs = {tuple(e.key for e in pg.repair_cycles(graph)[1]) for _ in range(5)}
        assert len(runs) == 1

    def test_find_cycle_returns_a_closed_path(self):
        cycle = pg.find_cycle(_graph(LINEAR + [("answer", "search")]))
        assert cycle == ["search", "read", "answer", "search"]


# ── Validation ───────────────────────────────────────
class TestValidate:
    def test_valid_graph_has_no_problems(self):
        assert pg.validate(_graph(LINEAR)) == []

    def test_start_is_required(self):
        graph = _graph(LINEAR)
        del graph.nodes[START]
        graph.edges = [e for e in graph.edges if e.source != START]
        assert any("missing the 'Start' node" in p for p in pg.validate(graph))

    def test_needs_a_terminal(self):
        graph = _graph([(START, "a"), ("a", START)], policy="allow")
        assert any("no terminal node" in p for p in pg.validate(graph))

    def test_every_node_must_reach_a_terminal(self):
        graph = _graph(LINEAR + [("x", "y"), ("y", "x")], policy="allow")
        problems = pg.validate(graph)
        assert any("no directed path to a terminal node from: x, y" in p for p in problems)

    def test_terminal_need_not_be_end(self):
        graph = _graph([(START, "a"), ("a", "done")], nodes={"done": "STATUS"})
        assert "End" not in graph.nodes
        assert pg.validate(graph) == []

    def test_forbid_rejects_any_cycle_allow_accepts_one_with_an_exit(self):
        edges = LINEAR + [("answer", "search")]
        assert any("cycle not allowed" in p for p in pg.validate(_graph(edges)))
        assert pg.validate(_graph(edges, policy="allow")) == []

    def test_dangling_endpoint(self):
        graph = _graph(LINEAR)
        graph.edges.append(ProcEdge("read", "ghost"))
        assert any("missing node(s): ghost" in p for p in pg.validate(graph))

    def test_invalid_types_relations_policy_and_key(self):
        graph = _graph(LINEAR)
        graph.nodes["read"] = ProcNode("read", "TOOL")
        graph.nodes["alias"] = ProcNode("other", "ACTION")
        graph.edges.append(ProcEdge("alias", END, "GOES_TO"))
        graph.cycle_policy = "sometimes"
        problems = " | ".join(pg.validate(graph))
        assert "invalid type 'TOOL'" in problems
        assert "invalid relation 'GOES_TO'" in problems
        assert "invalid cycle_policy 'sometimes'" in problems
        assert "key 'alias' has id 'other'" in problems

    @pytest.mark.parametrize("name", ["", "a/b", "a::b", "-lead", "x" * 65, "has space"])
    def test_graph_name_must_be_a_url_safe_slug(self, name):
        graph = _graph(LINEAR, name=name)
        assert any("invalid graph name" in p for p in pg.validate(graph))

    @pytest.mark.parametrize("name", ["graphrag-navigator", "nav.v2", "a_b-C9", "x" * 64])
    def test_accepted_graph_names(self, name):
        assert pg.validate(_graph(LINEAR, name=name)) == []


class TestWarnings:
    def test_clean_graph_has_none(self):
        assert pg.warnings(_graph(LINEAR, tools=("search", "read", "answer"))) == []

    def test_action_not_in_tools(self):
        notes = pg.warnings(_graph(LINEAR, tools=("search", "answer")))
        assert any("[read] is not one of the tools" in n for n in notes)

    def test_no_tool_catalog_means_no_tool_warning(self):
        assert pg.warnings(_graph(LINEAR)) == []

    def test_unreachable_from_start(self):
        graph = _graph(LINEAR + [("orphan", END)])
        assert any("unreachable from Start: orphan" in n for n in pg.warnings(graph))

    def test_empty_guidance(self):
        graph = pg.skeleton("nav")
        assert pg.warnings(graph) == ["transition [Start] -LEADS_TO→ [End] has no guidance"]

    def test_warnings_are_not_validation_failures(self):
        """Tool-catalog membership is a refiner-prompt rule, not structural (paper B.6)."""
        graph = _graph(LINEAR + [("orphan", END)], tools=("nope",))
        assert pg.warnings(graph) and pg.validate(graph) == []


# ── PrepareCandidate (Algorithm 1, line 10) ──────────
class TestPrepareCandidate:
    def test_success_returns_the_candidate_and_no_diagnostics(self):
        graph = _graph(LINEAR)
        cand, diags, notes = pg.prepare_candidate(
            graph, {"add_nodes": [{"id": "verify", "type": "REASONING"}],
                    "add_edges": [{"source": "read", "target": "verify", "guidance": "g"},
                                  {"source": "verify", "target": "answer", "guidance": "g"}]}
        )
        assert diags == []
        assert cand is not None and "verify" in cand.nodes
        assert cand is not graph and "verify" not in graph.nodes

    def test_malformed_edit_is_a_structural_failure_with_no_candidate(self):
        cand, diags, _ = pg.prepare_candidate(_graph(LINEAR), {"add_nodes": [{"id": "x", "type": "BAD"}]})
        assert cand is None
        assert any("invalid type 'BAD'" in d for d in diags)

    def test_edit_failures_and_validation_failures_are_reported_together(self):
        """The refiner's rejection memory should see every problem in one round."""
        cand, diags, _ = pg.prepare_candidate(
            _graph(LINEAR),
            {"add_nodes": [{"id": "dead_end", "type": "STATUS"}, {"id": "x", "type": "BAD"},
                           {"id": "stuck", "type": "ACTION"}],
             "add_edges": [{"source": "read", "target": "stuck"}, {"source": "stuck", "target": "ghost"}]},
        )
        assert cand is None
        joined = " | ".join(diags)
        assert "invalid type 'BAD'" in joined and "missing node(s): ghost" in joined

    def test_validation_failure_without_edit_errors(self):
        cand, diags, _ = pg.prepare_candidate(_graph(LINEAR), {"delete_nodes": [START]})
        assert cand is None
        assert any("missing the 'Start' node" in d for d in diags)

    def test_forbid_repairs_cycles_before_validation_and_notes_them(self):
        cand, diags, notes = pg.prepare_candidate(
            _graph(LINEAR), {"add_edges": [{"source": "answer", "target": "search", "relation": "TRIGGERS", "guidance": "retry"}]}
        )
        assert diags == []
        assert cand is not None and pg.find_cycle(cand) is None
        assert notes == ["removed cycle-closing edge [answer] -TRIGGERS→ [search]"]
        assert cand == _graph(LINEAR)

    def test_allow_keeps_the_cycle(self):
        graph = _graph(LINEAR, policy="allow")
        cand, diags, notes = pg.prepare_candidate(
            graph,
            {"add_edges": [{"source": "answer", "target": "search", "relation": "TRIGGERS",
                            "guidance": "retry"}]},
        )
        assert diags == [] and notes == []
        assert cand is not None and pg.find_cycle(cand) is not None

    def test_allow_still_requires_a_path_to_a_terminal(self):
        graph = _graph(LINEAR, policy="allow")
        cand, diags, _ = pg.prepare_candidate(
            graph,
            {"add_nodes": [{"id": "loop", "type": "ACTION"}],
             "add_edges": [{"source": "read", "target": "loop"}, {"source": "loop", "target": "read"}],
             "delete_edges": [{"source": "read", "target": "answer"}]},
        )
        assert cand is None
        assert any("no directed path to a terminal" in d for d in diags)

    def test_empty_edits_yield_an_identical_candidate_with_a_note(self):
        graph = _graph(LINEAR)
        cand, diags, notes = pg.prepare_candidate(graph, {"add_nodes": [], "delete_nodes": []})
        assert diags == [] and cand == graph
        assert any("no edits proposed" in n for n in notes)

    def test_no_op_deletes_are_notes_not_failures(self):
        cand, diags, notes = pg.prepare_candidate(
            _graph(LINEAR), {"delete_nodes": ["ghost"], "delete_edges": [{"source": "a", "target": "b"}]}
        )
        assert cand is not None and diags == []
        assert any("no node [ghost]" in n for n in notes)
        assert any("no edge [a]→[b]" in n for n in notes)


# ── Localisation ─────────────────────────────────────
class TestNormalizeAction:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('search_entities(query="x")', "search_entities"),
            ("search_entities", "search_entities"),
            ("Search Entities", "search_entities"),
            ("Search-Entities", "search_entities"),
            ("SEARCH_ENTITIES", "search_entities"),
            ("SearchEntities", "search_entities"),
            ("Action: FindPath(source='A', target='B')", "find_path"),
            ("  `answer`  (text=1)", "answer"),
            ("HTTPRequest", "http_request"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_cases(self, raw, expected):
        assert pg.normalize_action(raw) == expected


class TestLocalize:
    @pytest.fixture
    def graph(self):
        return _graph([(START, "search_entities"), ("search_entities", "find_path"),
                       ("find_path", END)])

    @pytest.mark.parametrize("last", [None, "", "   "])
    def test_empty_trajectory_localizes_on_start(self, graph, last):
        assert pg.localize(graph, last) == Localization(START, "start")

    def test_exact(self, graph):
        assert pg.localize(graph, "find_path") == Localization("find_path", "exact")
        assert pg.localize(graph, " find_path ") == Localization("find_path", "exact")

    @pytest.mark.parametrize(
        "last", ['search_entities(query="x")', "Search Entities", "SearchEntities", "searchentities"]
    )
    def test_normalized(self, graph, last):
        assert pg.localize(graph, last) == Localization("search_entities", "normalized")

    def test_none(self, graph):
        assert pg.localize(graph, "read_sources(entity='x')") == Localization(None, "none")

    def test_no_start_node_means_none(self):
        graph = ProceduralGraph("g", {"a": ProcNode("a", "ACTION")}, [])
        assert pg.localize(graph, None) == Localization(None, "none")

    def test_normalized_prefers_an_action_node(self):
        graph = _graph([(START, "Answer"), ("Answer", "answer"), ("answer", END)],
                       nodes={"Answer": "STATUS"})
        assert pg.localize(graph, "ANSWER") == Localization("answer", "normalized")

    def test_to_dict(self):
        assert Localization("a", "semantic", 0.8).to_dict() == {
            "node_id": "a", "method": "semantic", "score": 0.8
        }


# ── Neighbourhood & serialization ────────────────────
class TestNeighborhood:
    @pytest.fixture
    def graph(self):
        return _graph([(START, "a"), ("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"),
                       ("d", "e"), ("e", END)])

    def test_hops_are_bfs_layers_of_outgoing_edges(self, graph):
        layers = [(hop, e.source, e.target) for hop, e in pg.local_transitions(graph, "a", 2)]
        assert layers == [(1, "a", "b"), (1, "a", "c"), (2, "b", "d"), (2, "c", "d")]

    def test_only_outgoing_direction(self, graph):
        sub = pg.neighborhood(graph, "d", 2)
        assert [(e.source, e.target) for e in sub.edges] == [("d", "e"), ("e", END)]
        assert set(sub.nodes) == {"d", "e", END}

    def test_neighborhood_subgraph(self, graph):
        sub = pg.neighborhood(graph, "a", 2)
        assert set(sub.nodes) == {"a", "b", "c", "d"}
        assert list(sub.nodes) == ["a", "b", "c", "d"]  # original node order
        assert sub.name == graph.name and sub.tools == graph.tools

    def test_hops_zero_is_the_node_alone(self, graph):
        assert pg.local_transitions(graph, "a", 0) == []
        assert list(pg.neighborhood(graph, "a", 0).nodes) == ["a"]
        assert pg.local_transitions(graph, "a", -3) == []

    def test_cycles_terminate(self):
        graph = _graph(LINEAR + [("answer", "search")], policy="allow")
        transitions = pg.local_transitions(graph, "search", 10)
        assert len(transitions) == len(graph.edges) - 1  # Start→search is never outgoing here

    def test_unknown_node_raises(self, graph):
        with pytest.raises(KeyError):
            pg.local_transitions(graph, "ghost")


class TestSerialize:
    @pytest.fixture
    def graph(self):
        return ProceduralGraph(
            name="nav",
            description="demo",
            nodes={
                START: ProcNode(START, "STATUS", "Nothing done yet."),
                "search": ProcNode("search", "ACTION", "Find the entity."),
                "answer": ProcNode("answer", "ACTION", ""),
                END: ProcNode(END, "STATUS", "Done."),
            },
            edges=[
                ProcEdge(START, "search", "LEADS_TO", None, "Search the named entity.", "No full sentences."),
                ProcEdge("search", "answer", "TRIGGERS", "When the fact is\n found", "Answer   briefly.", ""),
                ProcEdge("answer", END, "LEADS_TO"),
            ],
        )

    def test_local_format_matches_the_papers_layout_plus_relation(self, graph):
        assert pg.serialize_local(graph, START, 2) == (
            "Active Cognitive Node: [Start] (Type: STATUS)\n"
            "Description: Nothing done yet.\n"
            "Immediate Transition Options (Hop 1):\n"
            "- Transition: [Start]→[search] (Relation: LEADS_TO; Condition: unconditional)\n"
            "  * Guidance: Search the named entity.\n"
            "  * Pitfalls to Avoid: No full sentences.\n"
            "Subsequent Horizon (Hop 2):\n"
            "- Transition: [search]→[answer] (Relation: TRIGGERS; Condition: When the fact is found)\n"
            "  * Guidance: Answer briefly."
        )

    def test_terminal_node(self, graph):
        assert pg.serialize_local(graph, END) == (
            "Active Cognitive Node: [End] (Type: STATUS)\n"
            "Description: Done.\n"
            "Immediate Transition Options (Hop 1):\n"
            "- none: this is a terminal node; the procedure ends here"
        )

    def test_empty_description_and_hop_sections_are_omitted(self, graph):
        text = pg.serialize_local(graph, "answer", 2)
        assert "Description" not in text
        assert "Subsequent Horizon" not in text  # End has no outgoing edges

    def test_further_hops_are_labelled(self, graph):
        assert "Further Horizon (Hop 3):" in pg.serialize_local(graph, START, 3)

    def test_is_deterministic(self, graph):
        assert pg.serialize_local(graph, START) == pg.serialize_local(graph.copy(), START)

    def test_full_lists_every_node_then_every_transition(self, graph):
        text = pg.serialize_full(graph)
        lines = text.splitlines()
        assert lines[:3] == ["Procedural Graph: [nav]", "Description: demo", "Nodes:"]
        assert "- [search] (Type: ACTION): Find the entity." in lines
        assert "- [answer] (Type: ACTION)" in lines
        assert lines.index("Transitions:") > lines.index("- [End] (Type: STATUS): Done.")
        assert text.count("- Transition:") == len(graph.edges)

    def test_full_of_an_edgeless_graph(self):
        graph = ProceduralGraph("g", {START: ProcNode(START, "STATUS")}, [])
        assert pg.serialize_full(graph).endswith("Transitions:\n- none")


# ── Diff ─────────────────────────────────────────────
class TestGraphDiff:
    def test_identical_graphs(self):
        diff = pg.graph_diff(_graph(LINEAR), _graph(LINEAR))
        assert all(v == [] for v in diff.values())
        assert set(diff) >= {"added_nodes", "removed_nodes", "added_edges", "removed_edges", "changed_edges"}
        assert pg.summarize_diff(diff) == "no structural change"

    def test_detects_every_kind_of_change(self):
        old = _graph(LINEAR)
        new, _ = pg.apply_edits(
            old,
            {
                "delete_nodes": ["read"],
                "add_nodes": [{"id": "verify", "type": "REASONING"}, {"id": "search", "description": "reworded"}],
                "add_edges": [
                    {"source": "search", "target": "verify", "guidance": "g"},
                    {"source": "verify", "target": "answer", "guidance": "g"},
                    {"source": "answer", "target": END, "guidance": "changed"},
                ],
            },
        )
        diff = pg.graph_diff(old, new)
        assert [n["id"] for n in diff["added_nodes"]] == ["verify"]
        assert [n["id"] for n in diff["removed_nodes"]] == ["read"]
        assert [n["id"] for n in diff["changed_nodes"]] == ["search"]
        assert {(e["source"], e["target"]) for e in diff["added_edges"]} == {
            ("search", "verify"), ("verify", "answer")}
        assert {(e["source"], e["target"]) for e in diff["removed_edges"]} == {
            ("search", "read"), ("read", "answer")}
        [changed] = diff["changed_edges"]
        assert (changed["source"], changed["target"]) == ("answer", END)
        assert changed["before"]["guidance"] == f"go {END}" and changed["after"]["guidance"] == "changed"
        assert pg.summarize_diff(diff) == (
            "+1 node, -1 node, 1 node changed, +2 edges, -2 edges, 1 edge changed"
        )

    def test_relation_change_is_a_remove_plus_add(self):
        old = _graph(LINEAR)
        new = old.copy()
        new.edges[0] = dataclasses.replace(new.edges[0], relation="TRIGGERS")
        diff = pg.graph_diff(old, new)
        assert len(diff["added_edges"]) == 1 and len(diff["removed_edges"]) == 1
        assert diff["changed_edges"] == []

    def test_is_json_serializable(self):
        json.dumps(pg.graph_diff(pg.skeleton("a"), _prior()))


# ── The bundled expert prior ─────────────────────────
class TestNavigatorPrior:
    TOOLS = ("search_entities", "neighbors", "read_sources", "search_passages", "find_path", "answer")

    def test_is_structurally_valid_with_no_warnings(self):
        graph = _prior()
        assert pg.validate(graph) == []
        assert pg.warnings(graph) == []

    def test_size_is_in_the_papers_range(self):
        graph = _prior()
        assert 9 <= len(graph.nodes) <= 12
        assert 7 <= len(graph.edges) <= 27  # paper Table 7: 7-27 triplets
        assert graph.name == "graphrag-navigator"
        assert graph.cycle_policy == "forbid"

    def test_action_nodes_are_exactly_the_navigator_tools(self):
        graph = _prior()
        assert graph.tools == self.TOOLS
        assert {n.id for n in graph.nodes.values() if n.type == "ACTION"} == set(self.TOOLS)

    def test_matches_the_navigator_tool_names_when_present(self):
        agent = pytest.importorskip("app.services.graph_agent")
        assert tuple(agent.TOOL_NAMES) == _prior().tools

    def test_uses_all_three_node_types_and_starts_at_start(self):
        graph = _prior()
        assert {n.type for n in graph.nodes.values()} == set(pg.NODE_TYPES)
        assert list(graph.nodes)[0] == START and graph.terminals() == [END]
        assert all(pg.localize(graph, tool).method == "exact" for tool in self.TOOLS)

    def test_every_edge_has_guidance_and_pitfalls(self):
        for edge in _prior().edges:
            assert edge.guidance.strip() and edge.pitfalls.strip(), edge
            assert edge.relation in pg.RELATIONS

    def test_encodes_the_multi_hop_strategy(self):
        graph = _prior()
        start_view = pg.serialize_local(graph, START)
        assert "search_entities" in start_view  # hop 2 from Start
        after_search = {e.target for e in graph.out_edges("search_entities")}
        assert after_search == {"neighbors", "read_sources", "search_passages"}
        text = pg.serialize_full(graph).lower()
        assert "bridge" in text
        assert "do not answer from the entity name alone" in text


class TestMcpHostPrior:
    """The prior an MCP host (Claude, Cursor…) is steered by.

    Localisation matches the caller's last action to an ACTION node id, so a host
    can only be placed in a graph whose ACTION ids are the tools IT calls — the
    synapse_* MCP tools, not the navigator's internal ones.
    """

    TOOLS = (
        "synapse_retrieve",
        "synapse_find_entities",
        "synapse_communities",
        "synapse_graph_stats",
        "synapse_status",
        "synapse_ask",
    )
    MCP_SERVER = (
        PRIORS_DIR.parents[3]
        / "packages"
        / "synapse-graphrag"
        / "src"
        / "synapse_graphrag"
        / "mcp_server.py"
    )

    def test_is_structurally_valid_with_no_warnings(self):
        graph = _prior("mcp-host")
        assert pg.validate(graph) == []
        assert pg.warnings(graph) == []

    def test_size_is_in_the_papers_range(self):
        graph = _prior("mcp-host")
        assert 7 <= len(graph.nodes) <= 17  # paper: 7-17 nodes
        assert 7 <= len(graph.edges) <= 27  # paper Table 7: 7-27 triplets
        assert graph.name == "mcp-host"
        assert graph.cycle_policy == "forbid"

    def test_action_nodes_are_exactly_the_mcp_tools(self):
        graph = _prior("mcp-host")
        assert graph.tools == self.TOOLS
        assert {n.id for n in graph.nodes.values() if n.type == "ACTION"} == set(self.TOOLS)

    def test_matches_the_mcp_servers_tool_names_when_present(self):
        """Drift guard against the package: every ACTION id is a registered MCP tool."""
        if not self.MCP_SERVER.is_file():
            pytest.skip("the synapse-graphrag package is not in this checkout")
        source = self.MCP_SERVER.read_text(encoding="utf-8")
        registered = set(re.findall(r'@server\.tool\(\s*name="(\w+)"', source))
        assert "synapse_retrieve" in registered  # the pattern still finds the tools
        assert set(_prior("mcp-host").tools) <= registered
        assert re.search(r'^DEFAULT_PROCEDURE = "mcp-host"$', source, re.MULTILINE)

    def test_uses_all_three_node_types_and_starts_at_start(self):
        graph = _prior("mcp-host")
        assert {n.type for n in graph.nodes.values()} == set(pg.NODE_TYPES)
        assert list(graph.nodes)[0] == START and graph.terminals() == [END]
        assert all(pg.localize(graph, tool).method == "exact" for tool in self.TOOLS)
        call = pg.localize(graph, 'synapse_retrieve(query="Who founded it?", k=8)')
        assert (call.node_id, call.method) == ("synapse_retrieve", "normalized")

    def test_every_edge_has_guidance_and_pitfalls(self):
        for edge in _prior("mcp-host").edges:
            assert edge.guidance.strip() and edge.pitfalls.strip(), edge
            assert edge.relation in pg.RELATIONS

    def test_a_host_is_localised_here_but_never_on_the_navigator(self):
        """The bug this prior fixes: the navigator graph gives a host the full graph every call."""
        host, navigator = _prior("mcp-host"), _prior()
        for tool in self.TOOLS:
            assert pg.localize(navigator, tool).method == "none", tool
            assert pg.localize(host, tool).node_id == tool
        local = pg.serialize_local(host, "synapse_retrieve")
        assert len(local) < len(pg.serialize_full(host)) / 3

    def test_encodes_the_host_strategy(self):
        graph = _prior("mcp-host")
        start_view = pg.serialize_local(graph, START)
        # Hop 2 from Start already offers every first tool, each with its condition.
        assert all(f"→[{tool}]" in start_view for tool in self.TOOLS)
        [plan] = [e.target for e in graph.out_edges(START)]
        first = {e.target: e for e in graph.out_edges(plan)}
        assert set(first) == set(self.TOOLS)
        # synapse_retrieve is the default path; synapse_ask only on an explicit request.
        assert first["synapse_retrieve"].relation == "LEADS_TO"
        assert "budget" in first["synapse_retrieve"].guidance
        assert first["synapse_ask"].relation == "TRIGGERS"
        assert "explicitly" in (first["synapse_ask"].condition or "")
        assert "second llm" in first["synapse_ask"].pitfalls.lower() + first["synapse_retrieve"].pitfalls.lower()
        # Retrieval feeds verification against the sources before any answer.
        [verify] = [e.target for e in graph.out_edges("synapse_retrieve")]
        assert graph.nodes[verify].type == "REASONING"
        assert "source" in graph.nodes[verify].description.lower()
        # The run is recorded at the end, with an honest score.
        [last] = [e for e in graph.edges if e.target == END and e.source != "synapse_status"]
        assert "synapse_record_trajectory" in last.guidance and "honest" in last.guidance

    async def test_guidance_for_a_host_call_is_local_and_free(self, monkeypatch):
        """guide() on the prior: an exact match, the local subgraph, no LLM and no embedding."""
        from app.services import procedural_guidance

        def no_embeddings():
            raise AssertionError("an exact match must not embed anything")

        monkeypatch.setattr(procedural_guidance, "get_embeddings", no_embeddings)
        graph = _prior("mcp-host")
        result = await procedural_guidance.guide(
            "mcp-host",
            query="Who founded the lab?",
            trajectory=[
                {"action": "synapse_find_entities", "observation": "Ada Lovelace (PERSON)"},
                {"action": "synapse_retrieve", "observation": "context …"},
            ],
            mode="raw",
            hops=2,
            graph=graph,
        )
        assert (result["active_node"], result["localization"], result["scope"]) == (
            "synapse_retrieve",
            "exact",
            "local",
        )
        assert result["context"] == pg.serialize_local(graph, "synapse_retrieve", 2)
        assert result["next_actions"] == ["Verify_Against_Sources"]
        assert result["usage"]["llm_calls"] == 0
