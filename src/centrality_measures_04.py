"""
centrality_measures.py
----------------------
Functions to compute and interpret the four core centrality measures.
"""

import networkx as nx
import pandas as pd


def degree_centrality(G):
    """
    Degree Centrality: fraction of nodes each node is connected to.
    High value → well-connected (many direct neighbours).
    Formula: C_D(v) = deg(v) / (n - 1)
    """
    return nx.degree_centrality(G)


def closeness_centrality(G):
    """
    Closeness Centrality: how close a node is to all others.
    High value → can quickly reach any node in the graph.
    Formula: C_C(v) = (n-1) / sum of shortest-path distances from v
    """
    return nx.closeness_centrality(G)


def betweenness_centrality(G):
    """
    Betweenness Centrality: how often a node lies on shortest paths between others.
    High value → acts as a bridge / bottleneck / broker.
    Formula: C_B(v) = Σ (σ_st(v) / σ_st)  for all s ≠ v ≠ t
    """
    return nx.betweenness_centrality(G, normalized=True)


def eigenvector_centrality(G):
    """
    Eigenvector Centrality: importance based on the importance of neighbours.
    High value → connected to other highly-connected nodes (prestige).
    Based on the leading eigenvector of the adjacency matrix.
    """
    return nx.eigenvector_centrality(G, max_iter=1000)


def centrality_summary(G):
    """
    Return a DataFrame comparing all four centrality measures for every node.
    Rounds values to 4 decimal places for readability.
    """
    data = {
        "Degree":       degree_centrality(G),
        "Closeness":    closeness_centrality(G),
        "Betweenness":  betweenness_centrality(G),
        "Eigenvector":  eigenvector_centrality(G),
    }
    df = pd.DataFrame(data).round(4)
    df.index.name = "Node"
    return df.sort_values("Betweenness", ascending=False)


def top_nodes_by_centrality(G, measure="betweenness", top_n=5):
    """
    Return the top N nodes ranked by the specified centrality measure.

    Parameters
    ----------
    measure : str — one of 'degree', 'closeness', 'betweenness', 'eigenvector'
    top_n   : int — number of top nodes to return
    """
    measure_map = {
        "degree":      degree_centrality,
        "closeness":   closeness_centrality,
        "betweenness": betweenness_centrality,
        "eigenvector": eigenvector_centrality,
    }
    if measure not in measure_map:
        raise ValueError(f"Unknown measure '{measure}'. "
                         f"Choose from: {list(measure_map.keys())}")

    centrality = measure_map[measure](G)
    ranked = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]