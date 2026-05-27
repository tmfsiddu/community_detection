# Community Detection Notes

# Table of Contents

1. Introduction to Community Detection
2. Graph Basics
3. Modularity
4. Girvan–Newman Algorithm
5. Greedy Modularity (CNM)
6. Louvain Algorithm
7. Label Propagation
8. K-Clique Percolation
9. Important Evaluation Metrics
10. Important NetworkX Functions
11. Key Formula Summary
12. Comparison of Algorithms

---

# 1. Introduction to Community Detection

## What is Community Detection?

Community detection means:

> Finding groups of nodes that are more densely connected internally than externally.

---

## Real-Life Examples

### Social Networks

* Friend groups
* Family circles
* Office groups

### Biology

* Protein interaction groups
* Gene clusters

### Recommendation Systems

* Similar users
* Similar products

### Citation Networks

* Research areas
* Scientific communities

---

# 2. Graph Basics

## Graph

A graph consists of:

* Nodes (vertices)
* Edges (connections)

---

## Degree of a Node

The number of edges connected to a node.

Example:

```text
A --- B --- C
```

Degrees:

* deg(A) = 1
* deg(B) = 2
* deg(C) = 1

---

## Clique

A clique is a fully connected group.

### 3-Clique (Triangle)

```text
A
|\
| \
B---C
```

Every node connected to every other node.

---

## Path

A sequence of connected nodes.

Example:

```text
A → B → C → D
```

---

## Shortest Path

The path with minimum number of edges.

---

# 3. Modularity

## What is Modularity?

Modularity measures:

> How good a community division is.

It checks:

* Many edges inside communities
* Few edges between communities

---

## Modularity Formula

[
Q = \frac{1}{2m} \sum_{ij} \left(A_{ij} - \frac{k_i k_j}{2m}\right) \delta(c_i,c_j)
]

---

## Meaning of Symbols

| Symbol            | Meaning                          |
| ----------------- | -------------------------------- |
| (A_{ij})          | 1 if edge exists between i and j |
| (k_i)             | Degree of node i                 |
| (m)               | Number of edges                  |
| (\delta(c_i,c_j)) | 1 if nodes in same community     |

---

## Intuition

Modularity compares:

```text
actual internal edges
-
expected random edges
```

If communities are much denser than random:

```text
high modularity
```

---

## Modularity Range

| Value    | Meaning            |
| -------- | ------------------ |
| Near 1   | Strong communities |
| Near 0   | Random structure   |
| Negative | Worse than random  |

---

# 4. Girvan–Newman Algorithm

## Main Idea

Remove important bridge edges connecting communities.

---

## Important Concept: Edge Betweenness

Edge betweenness measures:

> Number of shortest paths passing through an edge.

Edges connecting communities usually have high betweenness.

---

## Working Steps

### Step 1

Compute edge betweenness.

### Step 2

Remove edge with highest betweenness.

### Step 3

Graph splits into communities.

### Step 4

Compute modularity.

### Step 5

Repeat.

---

## Important NetworkX Code

```python
nx_comm.girvan_newman(G)
```

Returns partitions progressively.

---

## Example Flow

```text
1 community
↓
2 communities
↓
3 communities
↓
4 communities
```

---

## Important Property

Girvan–Newman is:

* Divisive
* Top-down
* Based on edge removal

---

## Advantages

* Easy to understand
* Good quality communities

---

## Disadvantages

* Slow for large graphs
* Recomputes betweenness repeatedly

---

# 5. Greedy Modularity (CNM)

## Main Idea

Start with every node as its own community.

Repeatedly merge communities that increase modularity the most.

---

## Working Steps

### Initial State

```text
{A} {B} {C} {D}
```

### Step 1

Try all possible merges.

### Step 2

Compute modularity gain.

### Step 3

Choose best merge.

### Step 4

Repeat until no improvement.

---

## Important Formula

[
\Delta Q = Q_{new} - Q_{old}
]

---

## Important NetworkX Code

```python
nx_comm.greedy_modularity_communities(G)
```

Returns final best partition.

---

## Important Property

Greedy Modularity is:

* Agglomerative
* Bottom-up
* Merge-based

---

## Advantages

* Faster than Girvan–Newman
* Directly optimizes modularity

---

## Disadvantages

* Greedy decisions may not be globally optimal

---

# 6. Louvain Algorithm

## Main Idea

Optimize modularity using:

1. Local node movement
2. Community compression

---

# Two Main Phases

## Phase 1 — Local Optimization

Move nodes between communities if modularity improves.

---

## Phase 2 — Supernode Compression

Compress each community into a supernode.

Create smaller graph.

Repeat again.

---

## Important Flow

```text
Every node separate
↓
Move nodes
↓
Improve modularity
↓
Communities formed
↓
Compress into supernodes
↓
Repeat
```

---

## Important NetworkX Functions

### Final Communities

```python
nx_comm.louvain_communities(G)
```

---

### Phase-by-Phase Communities

```python
nx_comm.louvain_partitions(G)
```

---

# Local Optimization

## Main Question

```text
Should this node move to another community?
```

---

## Change in Modularity

[
\Delta Q = Q_{new} - Q_{old}
]

If:

```text
ΔQ > 0
```

move node.

---

## Simplified Delta-Q Formula

[
\Delta Q = \frac{2(k_{i,in,target}-k_{i,in,current})}{2L} - \frac{k_i(\sigma_{target}-\sigma_{current})}{2L^2}
]

---

## Meaning of Symbols

| Symbol             | Meaning                              |
| ------------------ | ------------------------------------ |
| (k_i)              | Degree of node i                     |
| (k_{i,in,target})  | Links from node to target community  |
| (k_{i,in,current}) | Links from node to current community |
| (\sigma_{target})  | Total degree of target community     |
| (\sigma_{current}) | Total degree of current community    |
| (L)                | Total number of edges                |

---

## Important Intuition

If node has more connections to target community:

```text
ΔQ becomes positive
```

Then node moves.

---

## Supernode Compression Example

Original:

```text
{A,B,C}
{D,E,F}
```

Compressed:

```text
S1 --- S2
```

where:

* S1 represents {A,B,C}
* S2 represents {D,E,F}

---

## Advantages

* Very fast
* Scalable
* High modularity
* Works on huge graphs

---

## Disadvantages

* May produce different results on different runs
* Resolution limit problem

---

# 7. Label Propagation

## Main Idea

Nodes adopt the label most common among neighbors.

---

## Initial State

Every node gets unique label.

Example:

```text
A:A
B:B
C:C
```

---

## Working Steps

### Step 1

Look at neighbor labels.

### Step 2

Choose most common label.

### Step 3

Update node label.

### Step 4

Repeat until labels stabilize.

---

## Example

Initial:

```text
A:A
B:B
C:C
```

After propagation:

```text
A:A
B:A
C:A
```

---

## Important NetworkX Function

```python
nx_comm.label_propagation_communities(G)
```

---

## Important Property

Communities emerge naturally through local voting.

---

## Advantages

* Very fast
* Simple
* Scalable

---

## Disadvantages

* Unstable results
* Randomness affects output
* May create giant communities

---

# 8. K-Clique Percolation

## Main Idea

Communities are formed using overlapping cliques.

---

## Important Property

A node can belong to multiple communities.

---

# What is a k-Clique?

A fully connected group of k nodes.

---

## Example: 3-Clique

```text
A
|\
| \
B---C
```

---

## Example: 4-Clique

All 4 nodes connected to all others.

---

# Clique Percolation

Two cliques belong to same community if they overlap.

---

## Example

```text
Clique1 = {A,B,C}
Clique2 = {B,C,D}
```

Overlap:

```text
B,C
```

So same community.

---

## Important NetworkX Function

```python
nx_comm.k_clique_communities(G, k)
```

---

# Effect of k

| k       | Result                    |
| ------- | ------------------------- |
| Small k | Larger loose communities  |
| Large k | Smaller dense communities |

---

## Overlapping Communities Example

```text
Community A = {1,2,3,4}
Community B = {3,4,5,6}
```

Nodes 3 and 4 belong to BOTH communities.

---

## Noise Nodes

Some nodes may not belong to any clique.

These are assigned to a noise community.

---

## Advantages

* Supports overlapping communities
* Finds dense groups

---

## Disadvantages

* Sensitive to choice of k
* Large k may find no communities
* Expensive on large graphs

---

# 9. Important Evaluation Metrics

# Modularity

Measures community quality.

---

# NMI (Normalized Mutual Information)

Measures similarity between:

* predicted communities
* ground truth communities

---

## NMI Range

| Value | Meaning              |
| ----- | -------------------- |
| 1     | Perfect match        |
| 0     | Completely unrelated |

---

# 10. Important NetworkX Functions

| Function                                  | Purpose                   |
| ----------------------------------------- | ------------------------- |
| `nx_comm.modularity()`                    | Compute modularity        |
| `nx_comm.girvan_newman()`                 | Girvan–Newman algorithm   |
| `nx_comm.greedy_modularity_communities()` | Greedy modularity         |
| `nx_comm.louvain_communities()`           | Final Louvain communities |
| `nx_comm.louvain_partitions()`            | Louvain phase progression |
| `nx_comm.label_propagation_communities()` | Label propagation         |
| `nx_comm.k_clique_communities()`          | K-clique communities      |

---

# 11. Key Formula Summary

# Modularity

[
Q = \frac{1}{2m} \sum_{ij} \left(A_{ij} - \frac{k_i k_j}{2m}\right) \delta(c_i,c_j)
]

---

# Change in Modularity

[
\Delta Q = Q_{new} - Q_{old}
]

---

# Simplified Louvain Delta-Q

[
\Delta Q = \frac{2(k_{i,in,target}-k_{i,in,current})}{2L} - \frac{k_i(\sigma_{target}-\sigma_{current})}{2L^2}
]

---

# 12. Comparison of Algorithms

| Algorithm         | Main Idea             | Overlap? | Speed     | Method        |
| ----------------- | --------------------- | -------- | --------- | ------------- |
| Girvan–Newman     | Remove edges          | No       | Slow      | Divisive      |
| Greedy Modularity | Merge communities     | No       | Medium    | Agglomerative |
| Louvain           | Move nodes + compress | No       | Fast      | Multilevel    |
| Label Propagation | Spread labels         | No       | Very Fast | Voting        |
| K-Clique          | Overlapping cliques   | Yes      | Medium    | Clique-based  |

---

# Important Final Summary

## Girvan–Newman

* Removes bridge edges
* Uses edge betweenness
* Divisive method

---

## Greedy Modularity

* Merges communities
* Maximizes modularity greedily
* Agglomerative method

---

## Louvain

* Moves nodes locally
* Compresses communities into supernodes
* Repeats hierarchically

---

## Label Propagation

* Labels spread through neighbors
* Based on majority voting
* Very fast

---

## K-Clique

* Uses fully connected groups
* Supports overlapping communities
* Sensitive to parameter k
