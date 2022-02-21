import networkx as nx
import pandas as pd
import pytest

import qupid


@pytest.fixture
def one_to_many_mock():
    focus_cat_1 = ["A", "B", "C", "B", "C"]
    focus_cat_2 = [1.0, 2.0, 3.0, 2.5, 4.0]
    bg_cat_1 = ["A", "B", "B", "C", "D", "C", "A"]
    bg_cat_2 = [2.0, 1.0, 2.5, 2.5, 3.5, 4.0, 3.0]

    focus_index = [f"S{x}A" for x in range(5)]
    bg_index = [f"S{x}B" for x in range(7)]

    focus = pd.DataFrame({"cat_1": focus_cat_1, "cat_2": focus_cat_2},
                         index=focus_index)
    bg = pd.DataFrame({"cat_1": bg_cat_1, "cat_2": bg_cat_2},
                      index=bg_index)

    cat_type_map = {"cat_1": "discrete", "cat_2": "continuous"}
    tol_map = {"cat_2": 1.0}

    cm = qupid.match_by_multiple(focus, bg, cat_type_map, tol_map)
    return cm


def test_to_networkx_graph(one_to_many_mock):
    G = one_to_many_mock.to_networkx_graph()
    exp_nodes = [f"S{x}A" for x in range(5)]
    exp_nodes += [f"S{x}B" for x in (0, 1, 2, 3, 5)]
    assert isinstance(G, nx.DiGraph)
    assert set(exp_nodes) == set(G.nodes)

    for node1, node2, attr_dict in G.edges.data():
        assert node2 in one_to_many_mock[node1]
        assert attr_dict["capacity"] == 1
        assert G.nodes[node1]["sample_type"] == "case"
        assert G.nodes[node2]["sample_type"] == "control"
