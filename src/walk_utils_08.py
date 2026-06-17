# walk_utils.py
# ---------------------------------------------------------
# Functions for generating random walks on a graph.
# Used by both DeepWalk and Node2Vec.
# ---------------------------------------------------------

import random


def random_walk(graph, start_node, walk_length):
    """
    Uniform random walk — DeepWalk style.
    At each step, pick any neighbor with equal probability.

    Parameters
    ----------
    graph       : dict  {node: [neighbors]}
    start_node  : starting node
    walk_length : total number of nodes in the walk

    Returns
    -------
    walk : list of node names
    """
    walk = [start_node]
    current = start_node

    for _ in range(walk_length - 1):
        neighbors = graph[current]
        current = random.choice(neighbors)   # uniform pick
        walk.append(current)

    return walk


def node2vec_walk(graph, start_node, walk_length, p=1.0, q=1.0):
    """
    Biased random walk — Node2Vec style.

    p : return parameter  (low p → more likely to go back)
    q : in-out parameter  (low q → explore far, high q → stay local)

    p=1, q=1 → same as DeepWalk (uniform walk)

    Parameters
    ----------
    graph       : dict  {node: [neighbors]}
    start_node  : starting node
    walk_length : total number of nodes in the walk
    p           : float, return parameter
    q           : float, in-out parameter

    Returns
    -------
    walk : list of node names
    """
    walk = [start_node]

    if len(graph[start_node]) == 0:
        return walk   # isolated node

    # First step is always random
    walk.append(random.choice(graph[start_node]))

    for _ in range(walk_length - 2):
        current  = walk[-1]
        previous = walk[-2]
        neighbors = graph[current]

        weights = []
        for neighbor in neighbors:
            if neighbor == previous:
                # Going back to where we came from
                weights.append(1.0 / p)
            elif neighbor in graph[previous]:
                # Neighbor is also a neighbor of previous node (close)
                weights.append(1.0)
            else:
                # Exploring a new area (far)
                weights.append(1.0 / q)

        # Turn weights into probabilities
        total = sum(weights)
        probs = [w / total for w in weights]

        # Weighted random choice
        r, cumulative = random.random(), 0.0
        chosen = neighbors[-1]
        for i, prob in enumerate(probs):
            cumulative += prob
            if r <= cumulative:
                chosen = neighbors[i]
                break

        walk.append(chosen)

    return walk


def generate_walks(graph, num_walks, walk_length, walk_fn=random_walk, **kwargs):
    """
    Generate multiple walks starting from every node.

    Parameters
    ----------
    graph       : dict  {node: [neighbors]}
    num_walks   : how many walks to start from each node
    walk_length : length of each walk
    walk_fn     : which walk function to use (random_walk or node2vec_walk)
    **kwargs    : extra args passed to walk_fn (e.g. p=1, q=0.5)

    Returns
    -------
    all_walks : list of walks (each walk is a list of node names)
    """
    all_walks = []
    nodes = list(graph.keys())

    for _ in range(num_walks):
        random.shuffle(nodes)          # shuffle order to reduce bias
        for node in nodes:
            walk = walk_fn(graph, node, walk_length, **kwargs)
            all_walks.append(walk)

    return all_walks