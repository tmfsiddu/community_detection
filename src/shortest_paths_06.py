"""
shortest_paths.py
-----------------
Functions for computing and visualising shortest paths in graphs.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


NODE_COLOR  = "#4C72B0"
EDGE_COLOR  = "#888888"
PATH_COLOR  = "#E84040"
START_COLOR = "#2ECC71"
END_COLOR   = "#E74C3C"


# ── Path computation ──────────────────────────────────────────────────────────

def shortest_path(G, source, target, weight=None):
    """
    Return the shortest path between source and target.

    Parameters
    ----------
    weight : str or None — edge attribute to use as weight (None = unweighted)
    """
    return nx.shortest_path(G, source=source, target=target, weight=weight)


def shortest_path_length(G, source, target, weight=None):
    """
    Return the length (number of edges, or sum of weights) of the shortest path.
    """
    return nx.shortest_path_length(G, source=source, target=target, weight=weight)


def all_pairs_shortest_paths(G, weight=None):
    """
    Return a dict-of-dicts with shortest-path lengths between all node pairs.
    Useful for computing closeness centrality by hand.
    """
    return dict(nx.all_pairs_shortest_path_length(G))


def print_path_info(G, source, target, weight=None):
    """Print the path and its length between two nodes."""
    path   = shortest_path(G, source, target, weight=weight)
    length = shortest_path_length(G, source, target, weight=weight)
    label  = "weight" if weight else "hops"
    print(f"Shortest path  ({source} → {target}): {' → '.join(str(n) for n in path)}")
    print(f"Path length ({label})             : {length}")


# ── Visualisation ─────────────────────────────────────────────────────────────

def draw_shortest_path(G, source, target, weight=None,
                       title=None, figsize=(8, 6)):
    """
    Draw graph and highlight the shortest path in red.
    Start node is green; end node is red.
    """
    path = shortest_path(G, source, target, weight=weight)
    path_edges = list(zip(path, path[1:]))

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=figsize)

    # Non-path nodes
    other_nodes = [n for n in G.nodes() if n not in path]
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes,
                           node_color=NODE_COLOR, node_size=600, ax=ax)

    # Path nodes (excluding start/end)
    path_middle = path[1:-1]
    if path_middle:
        nx.draw_networkx_nodes(G, pos, nodelist=path_middle,
                               node_color="#F39C12", node_size=700, ax=ax)

    # Start and end nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[source],
                           node_color=START_COLOR, node_size=800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[target],
                           node_color=END_COLOR,   node_size=800, ax=ax)

    # Non-path edges
    non_path_edges = [e for e in G.edges() if e not in path_edges
                      and (e[1], e[0]) not in path_edges]
    nx.draw_networkx_edges(G, pos, edgelist=non_path_edges,
                           edge_color=EDGE_COLOR, width=1.5, alpha=0.5, ax=ax)

    # Path edges
    nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                           edge_color=PATH_COLOR, width=3.5, ax=ax)

    # Labels
    nx.draw_networkx_labels(G, pos, font_color="white",
                            font_weight="bold", ax=ax)

    # Edge weights (if weighted)
    if weight:
        edge_labels = nx.get_edge_attributes(G, weight)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=9, ax=ax)

    # Legend
    patches = [
        mpatches.Patch(color=START_COLOR, label=f"Start: {source}"),
        mpatches.Patch(color=END_COLOR,   label=f"End:   {target}"),
        mpatches.Patch(color=PATH_COLOR,  label="Shortest Path"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=10)

    length = shortest_path_length(G, source, target, weight=weight)
    if title is None:
        label = "weight" if weight else "hops"
        title = (f"Shortest Path: {source} → {target}   "
                 f"(length = {length} {label})")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.show()