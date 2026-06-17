"""
graph_visualization.py
----------------------
Simple visualization functions for NetworkX graphs using Matplotlib.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Default colours
NODE_COLOR = "#4C72B0"
EDGE_COLOR = "#888888"


def draw_graph(G, title="Graph", node_color=NODE_COLOR, layout="spring"):
    """Draw any basic graph (undirected, directed, multigraph)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Pick layout
    if layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx(G, pos, ax=ax,
                     node_color=node_color,
                     edge_color=EDGE_COLOR,
                     node_size=600,
                     font_color="white",
                     font_weight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_weighted_graph(G, title="Weighted Graph"):
    """Draw a weighted graph with edge weight labels shown."""
    fig, ax = plt.subplots(figsize=(7, 5))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx(G, pos, ax=ax,
                     node_color=NODE_COLOR,
                     edge_color=EDGE_COLOR,
                     node_size=600,
                     font_color="white",
                     font_weight="bold")

    # Show weights on edges
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_multiple_layouts(G, title_prefix="Graph",save_path=None):
    """Show the same graph in 5 different layouts side by side."""
    layouts = {
        "Spring":   nx.spring_layout(G, seed=42),
        "Circular": nx.circular_layout(G),
        "Shell":    nx.shell_layout(G),
        "Random":   nx.random_layout(G, seed=42),
        "Kamada":   nx.kamada_kawai_layout(G),
    }

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(f"{title_prefix} — Layout Comparison", fontsize=13, fontweight="bold")

    for ax, (name, pos) in zip(axes, layouts.items()):
        nx.draw_networkx(G, pos, ax=ax,
                         node_color=NODE_COLOR,
                         edge_color=EDGE_COLOR,
                         node_size=400,
                         font_color="white",
                         font_size=9)
        ax.set_title(f"{name}", fontsize=11)
        ax.axis("off")

    plt.tight_layout()

    # Save image if path is provided
    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight'
        )

    plt.show()


def draw_centrality(G, centrality_dict, title="Centrality", cmap="YlOrRd"):
    """Draw graph with node size and colour representing centrality value."""
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    values = list(centrality_dict.values())
    node_sizes  = [300 + v * 2500 for v in values]

    nc = nx.draw_networkx_nodes(G, pos, ax=ax,
                                node_size=node_sizes,
                                node_color=values,
                                cmap=plt.cm.get_cmap(cmap),
                                vmin=min(values), vmax=max(values))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=EDGE_COLOR, alpha=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")

    plt.colorbar(nc, ax=ax, label="Centrality Value")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_bridge_graph(G, bridge_edge, title="Bridge Edge"):
    """Highlight bridge edge in red; colour two communities differently."""
    fig, ax = plt.subplots(figsize=(9, 6))
    pos = nx.spring_layout(G, seed=7)

    community_1 = [n for n in G.nodes() if n < 5]
    community_2 = [n for n in G.nodes() if n >= 5]
    regular_edges = [e for e in G.edges() if set(e) != set(bridge_edge)]

    nx.draw_networkx_nodes(G, pos, nodelist=community_1, node_color="#4C72B0", node_size=700, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=community_2, node_color="#55A868", node_size=700, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, edge_color=EDGE_COLOR, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=[bridge_edge], edge_color="#E84040",
                           width=3.5, style="dashed", ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold", ax=ax)

    patches = [
        mpatches.Patch(color="#4C72B0", label="Community 1 (K₅)"),
        mpatches.Patch(color="#55A868", label="Community 2 (K₅)"),
        mpatches.Patch(color="#E84040", label="Bridge Edge"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_spanning_tree(G, T, title="Minimum Spanning Tree"):
    """Draw original graph with MST edges highlighted in red."""
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    non_tree_edges = [e for e in G.edges() if not T.has_edge(*e) and not T.has_edge(e[1], e[0])]

    nx.draw_networkx_nodes(G, pos, node_color=NODE_COLOR, node_size=700, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=non_tree_edges,
                           edge_color="#cccccc", style="dashed", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=list(T.edges()),
                           edge_color="#E84040", width=3.0, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "weight"),
                                 font_size=9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold", ax=ax)

    patches = [
        mpatches.Patch(color="#E84040", label="MST Edges"),
        mpatches.Patch(color="#cccccc", label="Other Edges"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_components(G, title="Connected Components"):
    """Colour each connected component a different colour."""
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    colours = plt.cm.Set2.colors
    patches = []
    for idx, component in enumerate(nx.connected_components(G)):
        colour = colours[idx % len(colours)]
        nx.draw_networkx_nodes(G, pos, nodelist=list(component),
                               node_color=[colour], node_size=700, ax=ax)
        patches.append(mpatches.Patch(color=colour, label=f"Component {idx + 1}"))

    nx.draw_networkx_edges(G, pos, edge_color=EDGE_COLOR, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="black", font_weight="bold", ax=ax)
    ax.legend(handles=patches, loc="upper right", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ── Aliases so notebook imports keep working after simplification ─────────────

def draw_simple_graph(G, title="Graph", node_color=NODE_COLOR,
                      edge_color=EDGE_COLOR, with_labels=True,
                      layout="spring", ax=None, figsize=(7, 5)):
    """Alias for draw_graph — draws a simple undirected graph."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42) if layout == "spring" else nx.circular_layout(G)
    nx.draw_networkx(G, pos, ax=ax,
                     node_color=node_color, edge_color=edge_color,
                     node_size=600, font_color="white",
                     font_weight="bold", with_labels=with_labels)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")


def draw_directed_graph(G, title="Directed Graph", ax=None, figsize=(7, 5)):
    """Draw a directed graph with arrows."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, ax=ax,
                     node_color="#E07B54", edge_color=EDGE_COLOR,
                     node_size=700, font_color="white",
                     font_weight="bold", arrows=True,
                     arrowsize=20, connectionstyle="arc3,rad=0.1")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")