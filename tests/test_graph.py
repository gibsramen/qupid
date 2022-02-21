import networkx as nx
import pandas as pd
import pytest

import qupid
import qupid.graph as graph


@pytest.fixture
def one_to_many_mock():
    cm_dict = {
        "S0A": {"S1B", "S2B", "S3B"},
        "S1A": {"S2B", "S3B", "S5B"},
        "S2A": {"S0B", "S1B", "S4B"},
        "S3A": {"S6B", "S7B"},
        "S4A": {"S4B", "S1B", "S6B"},
        "S5A": {"S7B", "S8B", "S9B"}
    }
    return qupid.CaseMatchOneToMany(cm_dict)


def test_to_networkx_graph(one_to_many_mock):
    G = one_to_many_mock.to_networkx_graph()
    exp_nodes = [f"S{x}A" for x in range(6)]
    exp_nodes += [f"S{x}B" for x in range(10)]
    assert isinstance(G, nx.DiGraph)
    assert set(exp_nodes) == set(G.nodes)

    for node1, node2, attr_dict in G.edges.data():
        assert node2 in one_to_many_mock[node1]
        assert attr_dict["capacity"] == 1
        assert G.nodes[node1]["sample_type"] == "case"
        assert G.nodes[node2]["sample_type"] == "control"


def test_edmonds_karp(one_to_many_mock):
    G = one_to_many_mock.to_networkx_graph()
    match_dict = graph.edmonds_karp(G)
    assert len(match_dict) == len(one_to_many_mock.cases)
    for case, ctrl in match_dict.items():
        assert case in one_to_many_mock.cases
        assert len(ctrl) == 1
        assert ctrl.issubset(one_to_many_mock.controls)
