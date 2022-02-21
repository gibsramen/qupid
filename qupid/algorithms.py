from typing import Callable

import networkx as nx
from networkx.algorithms import flow

import qupid
from qupid import _exceptions as exc


MATCH_ALGORITHM_FUNCS = {
    "edmonds_karp": flow.edmonds_karp,
    "ek": flow.edmonds_karp,
    "dinitz": flow.dinitz,
    "preflow_push": flow.preflow_push,
    "pp": flow.preflow_push,
    "boykov_kolmogorov": flow.boykov_kolmogorov,
    "bk": flow.boykov_kolmogorov,
    "shortest_augmenting_path": flow.shortest_augmenting_path,
    "sap": flow.shortest_augmenting_path,
}


def get_matching_algorithms() -> None:
    """Get all valid matching algorithms."""
    return list(MATCH_ALGORITHM_FUNCS.keys())


def match_one_to_one(
    case_match: "qupid.CaseMatchOneToMany",
    match_func: Callable,
) -> dict:
    """Match each case to a single control given a particular algorithm.

    :param case_match: All possible controls for each case
    :type case_match: qupid.CaseMatchOneToMany

    :param match_func: Function to use for flow maximization
    :type match_func: Callable
    """
    G = _add_flow_nodes(case_match.to_networkx_graph())
    _, match_graph_dict = flow.maximum_flow(G, "source", "target",
                                            flow_func=match_func)
    one_to_one_map = dict()
    for case, ctrls in match_graph_dict.items():
        if case not in case_match.cases:
            continue
        try:
            match_ctrl = next(
                ctrl_id for ctrl_id, flow in ctrls.items() if flow == 1
            )
        except StopIteration:
            remaining = case_match.cases.difference(one_to_one_map.keys())
            raise exc.NoMoreControlsError(remaining)
        one_to_one_map[case] = {match_ctrl}

    return one_to_one_map


def _add_flow_nodes(G: nx.DiGraph) -> nx.DiGraph:
    """Add source and target nodes to graph for flow analysis."""
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
    return _G
