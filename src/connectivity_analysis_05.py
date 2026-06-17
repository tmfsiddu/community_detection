"""
connectivity_analysis.py
------------------------
Functions to analyse graph connectivity:
  - Cliques and maximal cliques
  - Connected / weakly-connected components
  - Bridges and articulation points
  - Edge and node connectivity
"""

import networkx as nx


# ── Cliques ───────────────────────────────────────────────────────────────────

def find_all_cliques(G):
    """
    Return all maximal cliques in the graph.
    A clique is a subset of nodes where every pair is connected.
    A maximal clique cannot be extended by adding another node.
    """
    return list(nx.find_cliques(G))


def find_largest_clique(G):
    """Return the largest clique (or one of them if tied)."""
    cliques = find_all_cliques(G)
    return max(cliques, key=len)


def print_clique_info(G):
    """Print clique summary."""
    cliques = find_all_cliques(G)
    print(f"Number of maximal cliques : {len(cliques)}")
    for i, c in enumerate(sorted(cliques, key=len, reverse=True), 1):
        print(f"  Clique {i}: {sorted(c)}  (size {len(c)})")


# ── Connected components ──────────────────────────────────────────────────────

def get_connected_components(G):
    """
    Return a list of sets, each set being one connected component.
    Sorted by size (largest first).
    """
    components = list(nx.connected_components(G))
    return sorted(components, key=len, reverse=True)


def print_component_info(G):
    """Print number and members of each connected component."""
    components = get_connected_components(G)
    print(f"Number of connected components: {len(components)}")
    for i, comp in enumerate(components, 1):
        print(f"  Component {i}: {sorted(comp)}  ({len(comp)} nodes)")


# ── Bridges and articulation points ──────────────────────────────────────────

def find_bridges(G):
    """
    Return list of bridge edges.
    A bridge is an edge whose removal increases the number of components.
    These edges are critical connectivity bottlenecks.
    """
    return list(nx.bridges(G))


def find_articulation_points(G):
    """
    Return list of articulation points (cut vertices).
    Removing an articulation point disconnects the graph.
    """
    return list(nx.articulation_points(G))


def print_bridge_info(G):
    """Print bridge and articulation point information."""
    bridges = find_bridges(G)
    art_pts = find_articulation_points(G)
    print(f"Bridge edges         : {bridges}")
    print(f"Articulation points  : {art_pts}")


# ── Edge and node connectivity ────────────────────────────────────────────────

def edge_connectivity(G):
    """
    Return the edge connectivity: minimum edges to remove to disconnect graph.
    Higher value → more robust graph.
    """
    return nx.edge_connectivity(G)


def node_connectivity(G):
    """
    Return the node connectivity: minimum nodes to remove to disconnect graph.
    """
    return nx.node_connectivity(G)


def print_connectivity_info(G):
    """Print edge and node connectivity."""
    print(f"Edge connectivity : {edge_connectivity(G)}")
    print(f"Node connectivity : {node_connectivity(G)}")


# ── Simulate edge removal ─────────────────────────────────────────────────────

def remove_edge_and_check(G, edge):
    """
    Remove an edge from a copy of G and return the modified graph.
    Also prints whether the graph is now disconnected.
    """
    G_copy = G.copy()
    G_copy.remove_edge(*edge)

    n_before = nx.number_connected_components(G)
    n_after  = nx.number_connected_components(G_copy)

    print(f"Removed edge: {edge}")
    print(f"Components before removal: {n_before}")
    print(f"Components after  removal: {n_after}")
    if n_after > n_before:
        print("  ⚠ This was a bridge — graph is now disconnected!")
    else:
        print("  ✓ Graph remains connected.")
    return G_copy