import networkx as nx
import pytest

import qupid
import qupid.algorithms as algo


all_algos = list(algo.MATCH_ALGORITHM_FUNCS.keys())


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


@pytest.mark.parametrize("algo", all_algos)
def test_match_algos(one_to_many_mock, algo):
    cm_one_to_one = one_to_many_mock.greedy_match(algo)
    assert isinstance(cm_one_to_one, qupid.CaseMatchOneToOne)

    for case, ctrl in cm_one_to_one.case_control_map.items():
        assert len(ctrl) == 1
        assert ctrl.issubset(cm_one_to_one[case])


def test_get_matching_algorithms():
    match_algos = set(algo.get_matching_algorithms())
    exp_match_algos = {
        "edmonds_karp", "ek", "dinitz", "preflow_push", "pp",
        "boykov_kolmogorov", "bk", "shortest_augmenting_path", "sap"
    }
    assert match_algos == exp_match_algos
