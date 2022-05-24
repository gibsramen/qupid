import collections
import random

from networkx.algorithms.bipartite import sets as bipartite_sets

INFINITY = float("inf")


def hopcroft_karp_matching(G, top_nodes=None):
    """Returns a maximum cardinality matching of the bipartite graph G.

    NOTE: This function is modified from the NetworkX implementation:
        https://tinyurl.com/akcfy6r4

    We have modified the DFS algorithm to move to a *random* neighbor from all
    possible neighbors. This avoids the issue where the original NetworkX
    implementation was deterministic. We have also updated the names of some
    variables for readability in case-control matching context.

    :param G: Undirected bipartite graph
    :type G: networkx.Graph

    :param top_nodes: Container of nodes
    :type top_nodes: Iterable

    :returns: Dictionary matching each case to a single control
    """
    def breadth_first_search():
        for v in case:
            if casematches[v] is None:
                distances[v] = 0
                queue.append(v)
            else:
                distances[v] = INFINITY
        distances[None] = INFINITY
        while queue:
            v = queue.popleft()
            if distances[v] < distances[None]:
                for u in G[v]:
                    if distances[controlmatches[u]] is INFINITY:
                        distances[controlmatches[u]] = distances[v] + 1
                        queue.append(controlmatches[u])
        return distances[None] is not INFINITY

    def depth_first_search(v):
        if v is not None:
            # The following 3 lines have been modified from the original code
            connections = list(G[v])
            random.shuffle(connections)
            for u in connections:
                if distances[controlmatches[u]] == distances[v] + 1:
                    if depth_first_search(controlmatches[u]):
                        controlmatches[u] = v
                        casematches[v] = u
                        return True
            distances[v] = INFINITY
            return False
        return True

    # Initialize the "global" variables that maintain state during the search.
    case, control = bipartite_sets(G, top_nodes)
    casematches = {v: None for v in case}
    controlmatches = {v: None for v in control}
    distances = {}
    queue = collections.deque()

    # Implementation note: this counter is incremented as pairs are matched but
    # it is currently not used elsewhere in the computation.
    num_matched_pairs = 0
    while breadth_first_search():
        for v in case:
            if casematches[v] is None:
                if depth_first_search(v):
                    num_matched_pairs += 1

    # Strip the entries matched to None
    casematches = {k: v for k, v in casematches.items() if v is not None}

    return casematches
