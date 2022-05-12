import networkx as nx
import pytest

from qupid.matching import hopcroft_karp_matching


@pytest.fixture
def simple_graph():
    G = nx.Graph()

    G.add_node("1")
    G.add_node("2")
    G.add_node("3")

    G.add_node("A")
    G.add_node("B")
    G.add_node("C")

    G.add_edge("1", "A")
    G.add_edge("1", "C")

    G.add_edge("2", "B")
    G.add_edge("2", "C")

    G.add_edge("3", "A")
    G.add_edge("3", "B")

    return G


def test_hk(simple_graph):
    all_matches = set()
    for i in range(20):
        M = hopcroft_karp_matching(simple_graph)
        match = frozenset(tuple((x, M[x]) for x in M))
        all_matches.add(match)
    print(all_matches)

    exp_matches = {
        frozenset((("1", "A"), ("2", "C"), ("3", "B"))),
        frozenset((("1", "C"), ("2", "B"), ("3", "A"))),
    }
    assert all_matches == exp_matches
