"""
graph_metrics.py
----------------
Functions to compute and display structural graph metrics:
  - Adjacency / Degree / Laplacian matrices
  - Basic statistics (nodes, edges, degree, density, diameter)
  - Eulerian properties
"""

import networkx as nx
import numpy as np
import pandas as pd


# ── Matrix representations ────────────────────────────────────────────────────

def get_adjacency_matrix(G):
    """
    Return the adjacency matrix A as a NumPy array.
    A[i][j] = 1 if edge exists between node i and node j, else 0.
    """
    return nx.to_numpy_array(G)


def get_degree_matrix(G):
    """
    Return the degree matrix D as a NumPy array.
    D is diagonal: D[i][i] = degree of node i.
    """
    n = G.number_of_nodes()
    D = np.zeros((n, n))
    for i, node in enumerate(G.nodes()):
        D[i][i] = G.degree(node)
    return D


def get_laplacian_matrix(G):
    """
    Return the Laplacian matrix L = D - A as a NumPy array.
    Encodes graph structure; eigenvalues reveal connectivity.
    """
    A = get_adjacency_matrix(G)
    D = get_degree_matrix(G)
    return D - A


def print_matrices(G):
    """Pretty-print A, D, and L for a small graph."""
    nodes = list(G.nodes())
    A = get_adjacency_matrix(G)
    D = get_degree_matrix(G)
    L = get_laplacian_matrix(G)

    print("Adjacency Matrix (A):")
    print(pd.DataFrame(A.astype(int), index=nodes, columns=nodes).to_string())
    print("\nDegree Matrix (D):")
    print(pd.DataFrame(D.astype(int), index=nodes, columns=nodes).to_string())
    print("\nLaplacian Matrix (L = D − A):")
    print(pd.DataFrame(L.astype(int), index=nodes, columns=nodes).to_string())


# ── Basic graph statistics ────────────────────────────────────────────────────

def print_graph_info(G, name="Graph"):
    """Print a concise summary of graph properties."""
    print(f"{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  Nodes          : {G.number_of_nodes()}")
    print(f"  Edges          : {G.number_of_edges()}")
    print(f"  Directed       : {G.is_directed()}")
    print(f"  Weighted       : {nx.is_weighted(G)}")
    if not G.is_directed() and nx.is_connected(G):
        print(f"  Diameter       : {nx.diameter(G)}")
        print(f"  Density        : {nx.density(G):.4f}")
    print(f"{'='*45}")


def get_degree_info(G):
    """Return a sorted list of (node, degree) tuples."""
    return sorted(G.degree(), key=lambda x: x[1], reverse=True)


def get_neighbors(G, node):
    """Return sorted list of neighbours of a given node."""
    return sorted(G.neighbors(node))


def get_adjacency_list(G):
    """Return adjacency list as a dict {node: [neighbours]}."""
    return {node: sorted(G.neighbors(node)) for node in G.nodes()}


def get_edge_list(G):
    """Return list of all edges."""
    return list(G.edges())


# ── Density and diameter ──────────────────────────────────────────────────────

def graph_diameter(G):
    """
    Return the diameter of a connected graph.
    Diameter = longest shortest path between any two nodes.
    """
    if not nx.is_connected(G):
        raise ValueError("Graph is not connected. Diameter is undefined.")
    return nx.diameter(G)


def graph_density(G):
    """
    Return graph density in [0, 1].
    0 = no edges; 1 = complete graph (every node connected to every other).
    """
    return nx.density(G)


# ── Eulerian properties ───────────────────────────────────────────────────────

def check_eulerian(G):
    """
    Check and explain Eulerian properties of a graph.

    Eulerian Circuit : visits every edge exactly once and returns to start.
                       Required: connected graph + all nodes have even degree.
    Eulerian Path    : visits every edge exactly once (doesn't need to return).
                       Required: exactly 0 or 2 nodes with odd degree.
    """
    has_circuit = nx.is_eulerian(G)
    has_path    = nx.has_eulerian_path(G)

    odd_degree_nodes = [n for n, d in G.degree() if d % 2 != 0]

    print(f"Odd-degree nodes : {odd_degree_nodes} ({len(odd_degree_nodes)} total)")
    print(f"Eulerian Circuit : {'✓ Yes' if has_circuit else '✗ No'}")
    print(f"Eulerian Path    : {'✓ Yes' if has_path else '✗ No'}")

    if has_circuit:
        circuit = list(nx.eulerian_circuit(G))
        print(f"Circuit          : {' → '.join(str(n) for n, _ in circuit)} → {circuit[0][0]}")
    elif has_path:
        path = list(nx.eulerian_path(G))
        print(f"Path             : {' → '.join(str(n) for n, _ in path)}")

    return has_circuit, has_path