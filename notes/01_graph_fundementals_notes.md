# Graph Theory and Algorithms Notes

This document contains important graph theory concepts and algorithms commonly used in:

* Network Science
* Community Detection
* Graph Algorithms
* Competitive Programming
* Network Analysis

---

# 1. Breadth First Search (BFS)

Breadth First Search explores a graph level by level.

It first visits:

* immediate neighbors
* then neighbors of neighbors
* and so on.

BFS uses:

* Queue data structure

---

# Example

Graph:

A ----- B ----- C
|
|
D

Starting BFS from A:

Traversal:
A → B → D → C

---

# Applications of BFS

Used in:

* shortest path in unweighted graphs
* connected components
* web crawling
* social networks

---

# Simple Intuition

BFS spreads like:

* waves in water
* infection spreading level by level

---

# 2. Depth First Search (DFS)

Depth First Search explores as deep as possible before backtracking.

DFS uses:

* Stack
  or
* Recursion

---

# Example

Graph:

A ----- B ----- C
|
|
D

Starting DFS from A:

Traversal:
A → B → C → backtrack → D

---

# Applications of DFS

Used in:

* cycle detection
* topological sort
* connected components
* maze solving

---

# Simple Intuition

DFS behaves like:
"Keep moving deeper until no path exists."

---

# 3. Isomorphic Graphs

Two graphs are isomorphic if:

* they have same structure
* connections are identical
* only node labels differ

---

# Example

Graph 1:

A ----- B
|       |
C ----- D

Graph 2:

1 ----- 2
|       |
3 ----- 4

Both graphs have identical structure.

So they are isomorphic.

---

# Important Idea

Graph appearance may differ visually,
but structure remains same.

---

# Applications

Used in:

* chemistry
* pattern matching
* computer vision

---

# 4. Lowest Common Ancestor (LCA)

Lowest Common Ancestor means:

"The deepest common parent of two nodes in a tree."

---

# Example

```
    A
  /   \
 B     C
/ \
```

D   E

LCA of D and E = B

LCA of D and C = A

---

# Applications

Used in:

* tree queries
* hierarchical systems
* file systems

---

# 5. Topological Sort

Topological sorting gives:

* linear ordering of nodes

such that:

"If there is edge U → V,
then U appears before V."

Works only for:

* Directed Acyclic Graphs (DAGs)

---

# Example

Task Dependencies:

A → B → C

Possible topological order:

A → B → C

---

# Applications

Used in:

* task scheduling
* course prerequisites
* dependency management

---

# 6. Dijkstra Algorithm

Dijkstra algorithm finds:

* shortest path from source node
* in weighted graphs

Condition:

* edge weights must be non-negative

---

# Example

A --2-- B --1-- C

Shortest path:
A → B → C

Cost:
2 + 1 = 3

---

# Applications

Used in:

* GPS navigation
* routing systems
* network routing

---

# Important Idea

Always expand:

* currently closest node first

---

# 7. Bellman-Ford Algorithm

Bellman-Ford also finds:

* shortest paths

but it can handle:

* negative edge weights

---

# Example

A --(-2)-- B

Negative weights are allowed.

---

# Difference Between Dijkstra and Bellman-Ford

Dijkstra:

* faster
* no negative weights

Bellman-Ford:

* slower
* supports negative weights

---

# Important Feature

Bellman-Ford can detect:

* negative cycles

---

# 8. Euler Path and Euler Circuit

Euler Path:

* visits every edge exactly once

Euler Circuit:

* visits every edge exactly once
* and returns to starting node

---

# Conditions for Euler Path

Exactly 0 or 2 vertices must have odd degree.

---

# Conditions for Euler Circuit

All vertices must have even degree.

---

# Example

A ----- B
|       |
|       |
C ----- D

This graph has Euler Circuit.

---

# Applications

Used in:

* route optimization
* map traversal
* delivery systems

---

# 9. Bipartite Graph and Bipartite Matching

A bipartite graph has:

* two separate node groups

Edges connect:

* only between groups
* not within same group

---

# Example

Students ↔ Projects

Students:
A, B

Projects:
P1, P2

Connections:
A → P1
B → P2

---

# Bipartite Matching

Goal:

* match nodes optimally between two groups

---

# Applications

Used in:

* job assignment
* recommendation systems
* resource allocation

---

# 10. Capacity Scaling

Capacity Scaling is an optimization technique used in:

* maximum flow problems

It improves flow algorithms by:

* processing larger capacities first

---

# Example

Water Flow Network:

Source → Pipes → Destination

Each edge has:

* capacity limit

Capacity scaling efficiently finds:

* maximum possible flow

---

# Applications

Used in:

* transportation systems
* internet routing
* supply chain networks

---

# Simple Intuition

Instead of sending:
small flows repeatedly,

send:
larger flows first for efficiency.

---

# Important Graph Concepts Summary

BFS:
"Level-by-level traversal"

DFS:
"Deep traversal before backtracking"

Isomorphic Graphs:
"Different labels, same structure"

LCA:
"Deepest common parent"

Topological Sort:
"Dependency ordering"

Dijkstra:
"Shortest path with positive weights"

Bellman-Ford:
"Shortest path with negative weights"

Euler Path:
"Visit every edge once"

Bipartite Matching:
"Optimal matching between two groups"

Capacity Scaling:
"Efficient maximum flow computation"
