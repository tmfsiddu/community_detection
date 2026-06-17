# sbm_utils.py
# ---------------------------------------------------------
# SBM graph generation using NetworkX (no manual loops needed).
#
# Since you already have graph_creation.py with NetworkX,
# this file just adds SBM-specific helpers on top of it.
#
# What is SBM?
#   Nodes are split into blocks (communities).
#   p_in  = probability of edge WITHIN the same block  (high)
#   p_out = probability of edge BETWEEN blocks         (low)
#
#   Result: a graph with clear community structure.
# ---------------------------------------------------------

import networkx as nx


def create_sbm_graph(block_sizes, p_in, p_out, seed=42):
    """
    Generate an SBM graph using NetworkX's built-in function.

    NetworkX call:  nx.stochastic_block_model(sizes, p_matrix)
    where p_matrix[i][j] = probability of edge between block i and block j.

    Parameters
    ----------
    block_sizes : list of ints, e.g. [4, 4, 4]
    p_in        : float, probability of edge within same block
    p_out       : float, probability of edge between different blocks
    seed        : int, for reproducibility

    Returns
    -------
    G            : networkx Graph
    block_labels : dict  {node_id: block_index}
    """
    n_blocks = len(block_sizes)

    # Build the probability matrix
    # p_matrix[i][j] = p_in if i==j (same block), else p_out
    p_matrix = [
        [p_in if i == j else p_out for j in range(n_blocks)]
        for i in range(n_blocks)
    ]

    # NetworkX generates the graph directly
    G = nx.stochastic_block_model(block_sizes, p_matrix, seed=seed)

    # Build block_labels: NetworkX stores block as node attribute 'block'
    block_labels = nx.get_node_attributes(G, 'block')

    return G, block_labels


def networkx_to_adj_list(G):
    """
    Convert a NetworkX graph to a plain Python dict adjacency list.

    Why? Our walk_utils.py works with plain dicts (no NetworkX dependency).
    This lets walk_utils stay simple and framework-independent.

    Returns
    -------
    graph : dict  {node: [neighbors]}
    """
    return {node: list(G.neighbors(node)) for node in G.nodes()}


def print_sbm_info(G, block_labels):
    """Print a summary of the SBM graph."""
    n_blocks = max(block_labels.values()) + 1

    print("SBM Graph Summary:")
    print(f"  Nodes  : {G.number_of_nodes()}")
    print(f"  Edges  : {G.number_of_edges()}")
    print(f"  Blocks : {n_blocks}")
    print()

    for b in range(n_blocks):
        members = [n for n, blk in block_labels.items() if blk == b]
        print(f"  Block {b}: nodes {members}")


def get_block_pairs(block_labels):
    """
    Return two example pairs for similarity comparison:
      1. Two nodes from the SAME block   → should be similar
      2. Two nodes from DIFFERENT blocks → should be less similar

    Parameters
    ----------
    block_labels : dict  {node_id: block_index}

    Returns
    -------
    pairs : list of (node_a, node_b, label) tuples
    """
    block0 = [n for n, b in block_labels.items() if b == 0]
    block1 = [n for n, b in block_labels.items() if b == 1]

    pairs = []
    if len(block0) >= 2:
        pairs.append((block0[0], block0[1], "same block"))
    if block0 and block1:
        pairs.append((block0[0], block1[0], "different block"))

    return pairs