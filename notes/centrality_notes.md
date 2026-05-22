# Centrality Measures

Centrality measures help identify the most important or influential nodes in a graph.

Different centrality measures answer different questions about node importance.

Used in:

* social network analysis
* community detection
* recommendation systems
* communication networks
* transportation networks

---

# Example Graph

Consider the following graph:

A ----- B ----- C
|
|
D

Connections:

* A connected to B
* B connected to A, C and D
* C connected to B
* D connected to B

---

# Why Centrality is Important

In real-world networks:

* some nodes are more influential
* some nodes act as bridges
* some nodes spread information faster

Centrality measures help identify such nodes.

In the above graph:

* B and A are the most important nodes
  because it connects all other nodes.

---

# Types of Centrality

The main centrality measures are:

1. Degree Centrality
2. Closeness Centrality
3. Betweenness Centrality
4. Eigenvector Centrality

---

# 1. Degree Centrality

Degree centrality measures:

"How many direct connections does a node have?"

Formula:

Degree Centrality =
degree(node) / (n - 1)

Where:

* degree(node) = number of connections
* n = total number of nodes

---

# Example

Degrees in graph:

* deg(A) = 1
* deg(B) = 3
* deg(C) = 1
* deg(D) = 1

Node B has highest degree centrality.

What it represents:

* popularity
* direct influence
* immediate connectivity

Real-world example:

* social media influencer with many followers

---

# 2. Closeness Centrality

Closeness centrality measures:

"How quickly can a node reach all other nodes?"

A node with smaller total distance to others has higher closeness centrality.

Formula:

Closeness Centrality =
1 / (sum of shortest distances)

---

# Example

Node B can quickly reach:

* A
* C
* D

So B has highest closeness centrality.

What it represents:

* communication efficiency
* information spreading ability

Real-world example:

* person who can spread news fastest in a network

---

# 3. Betweenness Centrality

Betweenness centrality measures:

"How often does a node lie on shortest paths between other nodes?"

A node acting as bridge between groups has high betweenness centrality.

---

# Example

Many shortest paths pass through node B.

Removing B disconnects the graph.

So B has highest betweenness centrality.

What it represents:

* bridge nodes
* control over information flow
* connectivity between communities

Real-world example:

* airport connecting multiple cities
* router connecting networks

---

# 4. Eigenvector Centrality

Eigenvector centrality measures:

"A node is important if it connects to other important nodes."

Unlike degree centrality:

* quality of connections matters
* not just number of connections

---

# Example

Suppose:

* node A connected to random nodes
* node B connected to highly influential nodes

Then:

* B gets higher eigenvector centrality

What it represents:

* influence through influential neighbors

Real-world example:

* celebrity followed by other celebrities

---

# Comparison of Centrality Measures

Degree Centrality:
"Who has the most connections?"

Closeness Centrality:
"Who reaches everyone fastest?"

Betweenness Centrality:
"Who acts as bridge?"

Eigenvector Centrality:
"Who connects to important nodes?"

---

# Applications of Centrality

Used in:

* community detection
* identifying influencers
* fraud detection
* transportation systems
* communication networks
* recommendation systems
* social network analysis

---

# NetworkX Functions

Degree Centrality:
nx.degree_centrality(G)

Closeness Centrality:
nx.closeness_centrality(G)

Betweenness Centrality:
nx.betweenness_centrality(G)

Eigenvector Centrality:
nx.eigenvector_centrality(G)

---

# Simple Intuition

Degree Centrality:
"Most popular node"

Closeness Centrality:
"Fastest communicator"

Betweenness Centrality:
"Bridge between groups"

Eigenvector Centrality:
"Connected to important nodes"
