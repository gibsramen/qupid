from functools import wraps
from typing import Callable

import networkx as nx
from networkx.algorithms import flow

from qupid import CaseMatchOneToMany


def add_flow_nodes(func: Callable):
    """Add source and target nodes to graph for flow analysis."""
    @wraps(func)
    def inner(G: nx.DiGraph, **kwargs):
        _G = G.copy()
        _G.add_node("source")
        _G.add_node("target")
        for node, attr_dict in _G.nodes(data=True):
            if not attr_dict:
                continue
            if attr_dict["sample_type"] == "case":
                _G.add_edge("source", node, capacity=1.0)
            else:
                _G.add_edge(node, "target", capacity=1.0)
        return func(_G, **kwargs)
    return inner


@add_flow_nodes
def edmonds_karp(G: nx.DiGraph):
    """Return one-to-one matches using Edmonds-Karp algorithm."""
    R = flow.edmonds_karp(G, "source", "target")
    return _get_match_dict(R)


def _get_match_dict(R: nx.DiGraph):
    _R = R.copy()
    _R.remove_nodes_from(["source", "target"])
    ccm = dict()
    for node1, node2, attr_dict in _R.edges(data=True):
        if attr_dict["flow"] == 1.0:
            assert node1 not in ccm
            ccm[node1] = {node2}
    return ccm
