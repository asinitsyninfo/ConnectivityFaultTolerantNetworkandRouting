__all__ = ["umst"]
import itertools
import random
from typing import Optional
import networkx as nx
import sympy


def umst(graph: nx.Graph, k: int) -> nx.Graph:
    """Basic sync implementation based on A. Sinitsyn's algorithm"""
    n = len(graph.nodes)
    vertices_enumeration = range(0, n)
    combinations = itertools.combinations(vertices_enumeration, k)

    dummy = graph.copy()
    vertices = next(combinations)
    for v in vertices:
        dummy.remove_node(v)
    span_tree = nx.minimum_spanning_tree(dummy)
    for e in span_tree.edges():
        span_tree[e[0]][e[1]]["weight"] = 1

    for vertices in combinations:
        dummy = graph.copy()
        for v in vertices:
            dummy.remove_node(v)

        cur_span_tree = nx.minimum_spanning_tree(dummy)
        # TODO: smallest loop first optimization (ensure that the outcome is similar to the original algorithm)
        for e1 in cur_span_tree.edges():
            for e2 in span_tree.edges():
                if eq(e1, e2) == 1:
                    span_tree[e2[0]][e2[1]]["weight"] += 1
                else:
                    cur_span_tree[e1[0]][e1[1]]["weight"] = 1

        # Why nx.compose(span_tree, cur_span_tree) is not working?
        # Because: Attribute values from H take precedent over attribute values from G.
        span_tree = nx.compose(cur_span_tree, span_tree)
    return span_tree


def eq(e1, e2):
    result = 0
    if e1[0] == e2[0] and e1[1] == e2[1]:
        result = 1
    if e1[1] == e2[0] and e1[0] == e2[1]:
        result = 1
    if e1[0] == e2[1] and e1[1] == e2[0]:
        result = 1
    if e1[1] == e2[1] and e1[0] == e2[0]:
        result = 1
    return result


def get_route_priority_list(initial_graph: nx.Graph, umst_graph: nx.Graph) -> str:
    node_labels = {}
    n = len(initial_graph.nodes)
    for i in range(n):
        node_labels[i] = nx.get_node_attributes(initial_graph, "capital")[i]
    edges_sorted = sorted(
        umst_graph.edges(data=True),
        key=lambda edge: edge[2].get("weight", 1),
        reverse=True,
    )
    result = "Route Priority List\nfrom\t\t to\t\t priority\n"
    for i in range(n):
        result += str(node_labels[i]) + "\n"
        for j in range(len(edges_sorted)):
            if i == edges_sorted[j][0]:
                result += (
                    "\t\t"
                    + str(node_labels[edges_sorted[j][1]])
                    + "\t\t"
                    + str(
                        umst_graph.edges[edges_sorted[j][0], edges_sorted[j][1]][
                            "weight"
                        ]
                    )
                    + "\n"
                )
            if i == edges_sorted[j][1]:
                result += (
                    "\t\t"
                    + str(node_labels[edges_sorted[j][0]])
                    + "\t\t"
                    + str(
                        umst_graph.edges[edges_sorted[j][0], edges_sorted[j][1]][
                            "weight"
                        ]
                    )
                    + "\n"
                )
    return result


def get_umst_with_probs(
    span_tree: nx.Graph, broken_nodes: int, probs: list[float]
) -> nx.Graph:
    nodes_count = len(span_tree.nodes)
    span_tree = span_tree.copy()
    for i, j in span_tree.edges():
        span_tree[i][j]["weight"] = round(
            span_tree[i][j]["weight"]
            * probs[i]
            * probs[j]
            / sympy.binomial(nodes_count, broken_nodes),
            2,
        )
    return span_tree


def get_probs(
    graph: nx.Graph, ones: bool = False, seed: Optional[int] = None
) -> list[float]:
    nodes_count = len(graph.nodes)
    if seed:
        random.seed(seed)
    if ones:
        probs = [1 for _ in range(nodes_count + 1)]
    else:
        probs = [
            1 - 1.5 * random.randint(0, 100) / 1000 for i in range(nodes_count + 1)
        ]
    return probs
