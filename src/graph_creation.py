"""
graph_creation.py
-----------------
Utility functions for creating various types of graphs using NetworkX.
"""

import networkx as nx


def create_simple_graph():
    """
    Create a simple undirected graph with 6 nodes.
    Represents a small social network.
    """
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5, 6])
    G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (5, 6), (3, 6)])
    return G


def create_directed_graph():
    """
    Create a directed graph representing information flow.
    Arrows show direction of influence.
    """
    DG = nx.DiGraph()
    DG.add_edges_from([
        ("A", "B"), ("A", "C"),
        ("B", "D"), ("C", "D"),
        ("D", "E"), ("B", "E")
    ])
    return DG


def create_weighted_graph():
    """
    Create a weighted undirected graph.
    Weights could represent distances, costs, or connection strength.
    """
    WG = nx.Graph()
    WG.add_weighted_edges_from([
        (1, 2, 4), (1, 3, 2), (2, 3, 5),
        (2, 4, 10), (3, 4, 3), (4, 5, 7), (3, 5, 8)
    ])
    return WG


def create_multigraph():
    """
    Create a MultiGraph where multiple edges can exist between same nodes.
    Useful for representing multiple types of relationships.
    """
    MG = nx.MultiGraph()
    MG.add_edges_from([
        (1, 2), (1, 2),   # two edges between 1 and 2
        (2, 3), (3, 4),
        (4, 1), (4, 1)    # two edges between 4 and 1
    ])
    return MG


def create_karate_club_graph():
    """
    Load the famous Zachary's Karate Club graph.
    A real-world social network with 34 nodes and 78 edges.
    """
    return nx.karate_club_graph()


def create_two_cliques_with_bridge():
    """
    Create two complete graphs (K5) connected by a single bridge edge.
    Used to illustrate community structure and bridge edges.
    """
    G = nx.Graph()

    # First K5: nodes 0-4
    G.add_nodes_from(range(5))
    for i in range(5):
        for j in range(i + 1, 5):
            G.add_edge(i, j)

    # Second K5: nodes 5-9
    G.add_nodes_from(range(5, 10))
    for i in range(5, 10):
        for j in range(i + 1, 10):
            G.add_edge(i, j)

    # Bridge edge connecting the two cliques
    G.add_edge(4, 5)

    return G


def create_eulerian_graph():
    """
    Create a graph with an Eulerian circuit.
    An Eulerian circuit visits every edge exactly once and returns to start.
    Every node must have even degree for this to exist.
    """
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (0, 2), (1, 2),
        (1, 3), (2, 3), (3, 4),
        (3, 5), (4, 5)
    ])
    return G


def create_spanning_tree_graph():
    """
    Create a connected weighted graph suitable for MST demonstration.
    """
    G = nx.Graph()
    G.add_weighted_edges_from([
        (0, 1, 2), (0, 3, 6), (1, 2, 3),
        (1, 3, 8), (1, 4, 5), (2, 4, 7),
        (3, 4, 9), (3, 5, 11), (4, 5, 10)
    ])
    return G