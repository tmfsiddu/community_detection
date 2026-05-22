# Laplacian Matrix

The Laplacian matrix is one of the most important matrix representations in graph theory and network science.

It helps us understand:

* graph structure
* connectivity
* flow between nodes
* communities inside networks

The Laplacian matrix is defined as:

L = D - A

Where:

* D = Degree Matrix
* A = Adjacency Matrix

---

# Step 1 — Example Graph

Consider the following graph:

A ----- B
|       |
|       |
C ----- D

Connections:

* A connected to B and C
* B connected to A and D
* C connected to A and D
* D connected to B and C

---

# Step 2 — Adjacency Matrix (A)

The adjacency matrix tells:

"Which node is connected to which node?"

Rules:

* 1 → nodes are connected
* 0 → nodes are not connected

Node order:
A, B, C, D

Adjacency Matrix:

A =
[[0, 1, 1, 0],
[1, 0, 0, 1],
[1, 0, 0, 1],
[0, 1, 1, 0]]

Meaning:

* Row A, Column B = 1
  → A is connected to B

* Row A, Column D = 0
  → A is NOT connected to D

What adjacency matrix represents:

* direct connections between nodes
* graph structure
* neighbor relationships

---

# Step 3 — Degree Matrix (D)

The degree matrix tells:

"How many connections does each node have?"

Degree of nodes:

* deg(A) = 2
* deg(B) = 2
* deg(C) = 2
* deg(D) = 2

The degree matrix stores node degrees only on the diagonal.

Degree Matrix:

D =
[[2, 0, 0, 0],
[0, 2, 0, 0],
[0, 0, 2, 0],
[0, 0, 0, 2]]

Meaning:

* D[0][0] = 2
  → node A has degree 2

What degree matrix represents:

* total number of connections for each node
* node importance in terms of connectivity

---

# Step 4 — Laplacian Matrix (L)

Formula:

L = D - A

Subtract adjacency matrix from degree matrix.

Laplacian Matrix:

L =
[[ 2, -1, -1,  0],
[-1,  2,  0, -1],
[-1,  0,  2, -1],
[ 0, -1, -1,  2]]

Rules:

* diagonal = node degree
* connected nodes = -1
* non-connected nodes = 0

Meaning:

* L[0][0] = 2
  → degree of node A

* L[0][1] = -1
  → A connected to B

* L[0][3] = 0
  → A not connected to D

What Laplacian matrix represents:

* overall graph connectivity
* flow inside graph
* structural relationships
* separation between communities

---

# Important Properties

1. Symmetric for undirected graphs

2. Sum of every row = 0

Example:
2 + (-1) + (-1) + 0 = 0

3. Helps identify connected components and communities

---

# Applications of Laplacian Matrix

Used in:

* spectral clustering
* community detection
* graph partitioning
* graph neural networks
* recommendation systems
* image segmentation

---

# Simple Intuition

Adjacency Matrix:
"Who is connected to whom?"

Degree Matrix:
"How many connections does each node have?"

Laplacian Matrix:
"Overall structure and connectivity of the graph"
