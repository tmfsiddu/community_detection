# Graph Neural Networks (GNNs)
**Source:** Hamilton — *Graph Representation Learning* (Chapter 5) | Stanford CS224W  
**Status:** Personal research notes

---

## 1. Why GNNs? — The Limits of Shallow Embeddings

Before GNNs, techniques like **DeepWalk** and **Node2Vec** used *shallow embeddings* — they ran random walks over the graph and optimized a large lookup table to assign each node a unique vector.

**Three core problems with shallow embeddings:**

| Problem | Why It Hurts |
|---|---|
| No parameter sharing | Every node gets its own vector trained from scratch → massive model size for billion-node graphs |
| Cannot handle new nodes | A new node added after training has no embedding without full retraining |
| Ignores node features | Only uses graph topology; ignores attributes like user age, atom type, etc. |

**What GNNs solve:** A GNN builds a deep neural network *over* the graph, computing each node's embedding from both the graph structure *and* its features — and the same learned parameters are reused across all nodes.

---

## 2. Why Not Just Use CNNs or RNNs?

- **CNNs** expect data on a rigid grid (e.g., image pixels always have exactly 8 neighbours). Graphs don't have a fixed grid.
- **RNNs** expect a fixed sequential order. Graphs have no natural ordering of nodes.
- **Graphs are irregular:** a node can have 1 neighbour, 100 neighbours, or none. Standard deep learning architectures simply break on this input.

---

## 3. The Mathematical Requirements — Permutation Invariance & Equivariance

If you try to flatten the adjacency matrix $A$ into a vector and feed it into an MLP, you immediately hit a problem: **node ordering is arbitrary**. The same graph with nodes labelled $\{1,2,3\}$ vs $\{3,1,2\}$ produces a completely different matrix, yet it *is* the same graph.

A correct GNN must satisfy two properties depending on the task:

### 3.1 Permutation Equivariance (Node-Level Tasks)
> Predicting something for every individual node (e.g., classifying whether each user is a bot).

If you permute the input nodes by $P$, the output vectors must permute in the *same* way:

$$f(PAP^\top) = P\,f(A)$$

The internal values stay identical — only positions in the output change to match the input shuffle.

### 3.2 Permutation Invariance (Graph-Level Tasks)
> Predicting a single label for the whole graph (e.g., is this molecule toxic?).

No matter how you shuffle the nodes, the output must be exactly the same scalar or vector:

$$f(PAP^\top) = f(A)$$

**How GNNs achieve invariance:** After computing equivariant node embeddings, a **global pooling layer** (sum or mean over all nodes) collapses them into a single graph vector. Summing/averaging over a set doesn't care about order → the final prediction is invariant.

---

## 4. Neural Message Passing — The Core Framework

GNNs achieve permutation equivariance naturally by using **symmetric aggregation functions**. The core idea is a neighbourhood gossip network: at every layer, each node listens to its direct neighbours, collects their information, and updates its own representation.

Each layer $k$ has two steps:

### Step A — Aggregation
Node $u$ collects the current hidden vectors of all its neighbours $\mathcal{N}(u)$ and merges them into a single message:

$$m_{\mathcal{N}(u)}^{(k)} = \text{AGGREGATE}^{(k)}\!\left(\{h_v^{(k)},\; \forall v \in \mathcal{N}(u)\}\right)$$

> **Key rule:** AGGREGATE must be a *symmetric function* (sum, mean, max) so that node ordering doesn't matter → guarantees equivariance.

### Step B — Update
Node $u$ combines the neighbourhood message with its own previous vector to produce its new representation:

$$h_u^{(k+1)} = \text{UPDATE}^{(k)}\!\left(h_u^{(k)},\; m_{\mathcal{N}(u)}^{(k)}\right)$$

### What Each Layer Learns

| Layer $k$ | What the embedding captures |
|---|---|
| $k = 0$ | Node's own initial features $x_u$ (or degree / one-hot if no features) |
| $k = 1$ | Immediate neighbours (1-hop local structure) |
| $k = 2$ | Neighbours of neighbours (2-hop) |
| $k = K$ | Information from a $K$-hop neighbourhood |

The final embedding $z_u = h_u^{(K)}$ encodes both **structural information** (the shape and connectivity pattern around the node) and **feature-based information** (the distributed attributes of the neighbourhood).

---

## 5. The Basic GNN Equation

The simplest concrete instantiation of the framework:

$$h_u^{(k)} = \sigma\!\left(W_{\text{self}}^{(k)} h_u^{(k-1)} + W_{\text{neigh}}^{(k)} \sum_{v \in \mathcal{N}(u)} h_v^{(k-1)} + b^{(k)}\right)$$

Breaking it down:

- $\sum_{v \in \mathcal{N}(u)} h_v^{(k-1)}$ — sum of all neighbour vectors (basic AGGREGATE)
- $W_{\text{neigh}}^{(k)}$ — trainable weight matrix applied to the neighbourhood message
- $W_{\text{self}}^{(k)}$ — trainable weight matrix applied to the node's own vector (so it doesn't forget itself)
- $b^{(k)}$ — bias vector
- $\sigma$ — non-linear activation (ReLU, Tanh, etc.)

### 5.1 Matrix Form (Efficient Implementation)

Instead of looping over every node, stack all node vectors into matrix $H$ and use sparse matrix operations:

$$H^{(t)} = \sigma\!\left(AH^{(t-1)}W_{\text{neigh}}^{(k)} + H^{(t-1)}W_{\text{self}}^{(k)}\right)$$

Multiplying $A \cdot H$ handles all neighbourhood summations simultaneously across every node.

### 5.2 Self-Loops Trick

Add a self-loop to every node (include $u$ in its own neighbourhood $\mathcal{N}(u) \cup \{u\}$). Now $W_{\text{self}}$ and $W_{\text{neigh}}$ can be merged into one matrix, and the update simplifies to:

$$H^{(t)} = \sigma\!\left((A + I)\,H^{(t-1)}\,W^{(t)}\right)$$

where $I$ is the identity matrix.

---

## 6. Neighbourhood Normalization

**The problem:** A node with 2 neighbours and a node with 10,000 neighbours produce wildly different-scale aggregated messages. This makes training numerically unstable.

### 6.1 Degree Averaging

Simply divide the sum by the number of neighbours:

$$m_{\mathcal{N}(u)} = \frac{\sum_{v \in \mathcal{N}(u)} h_v}{|\mathcal{N}(u)|}$$

Stabilises scale, but treats every neighbour as equally important regardless of its own connectivity.

### 6.2 Symmetric Normalization (GCN)

Kipf & Welling's approach — scale using *both* the target node's degree and the sending node's degree:

$$m_{\mathcal{N}(u)} = \sum_{v \in \mathcal{N}(u)} \frac{h_v}{\sqrt{|\mathcal{N}(u)|}\,\sqrt{|\mathcal{N}(v)|}}$$

- High-degree target node $u$ → incoming messages are scaled down (prevents explosion).
- High-degree source node $v$ (a massive hub) → its individual message is penalised so it doesn't drown out smaller, more specific neighbours.

> This equation is a first-order approximation of **spectral graph convolutions**.

**The normalisation trade-off:**

| Approach | Advantage | Disadvantage |
|---|---|---|
| Summing | Preserves structural info (degree differences visible) | Unstable when degrees vary wildly |
| Normalising | Fast, stable training | Loses structural count information |

---

## 7. Improved Aggregation Methods

### 7.1 Graph Convolutional Networks (GCNs)

Combines symmetric normalisation + self-loop into one layer:

$$h_u^{(k)} = \sigma\!\left(W^{(k)} \sum_{v \in \mathcal{N}(u) \cup \{u\}} \frac{h_v}{\sqrt{|\mathcal{N}(u)|}\,\sqrt{|\mathcal{N}(v)|}}\right)$$

The most widely used baseline in graph deep learning.

### 7.2 Set Aggregators (Deep Sets)

Simple sum/mean/max is invariant but not very expressive. Zaheer et al. proved that *any* permutation-invariant function can be approximated by:

$$m_{\mathcal{N}(u)} = \text{MLP}_\theta\!\left(\sum_{v \in \mathcal{N}(u)} \text{MLP}_\phi(h_v)\right)$$

1. Pass each neighbour through a small neural network $\text{MLP}_\phi$ to extract rich features.
2. Sum all transformed vectors.
3. Pass the grand sum through another network $\text{MLP}_\theta$.

### 7.3 Janossy Pooling

**Idea:** Use an order-sensitive architecture (LSTM) — which is very powerful — but make it invariant by averaging over all possible orderings:

$$m_{\mathcal{N}(u)} = \text{MLP}_\theta\!\left(\frac{1}{|\Pi|} \sum_{\pi \in \Pi} \rho_\phi(h_{v_1}, h_{v_2}, \ldots)_\pi\right)$$

**Problem:** 10 neighbours → $10! = 3{,}628{,}800$ orderings. Intractable.

**Practical approximations:**
- **Sampling:** Randomly sample a small fraction of orderings during training.
- **Canonical ordering:** Sort neighbours by a fixed rule (e.g., descending degree) before feeding into the LSTM.

### 7.4 Graph Attention Networks (GATs)

Standard GCN uses *fixed* degree-based scaling. But a hub neighbour might be completely irrelevant to a specific target node. GATs use a **trainable attention mechanism** $\alpha_{u,v}$ to dynamically decide how much to care about each neighbour:

$$m_{\mathcal{N}(u)} = \sum_{v \in \mathcal{N}(u)} \alpha_{u,v}\, h_v$$

The attention weight is computed as:

$$\alpha_{u,v} = \frac{\exp\!\left(a^\top [Wh_u \oplus Wh_v]\right)}{\sum_{v' \in \mathcal{N}(u)} \exp\!\left(a^\top [Wh_u \oplus Wh_{v'}]\right)}$$

- Both node vectors are projected by $W$, concatenated ($\oplus$), and scored by a trainable vector $a^\top$.
- Softmax ensures all attention scores for node $u$'s neighbourhood sum to 1.
- Alternative scoring: bilinear dot-product $h_u^\top W h_v$.

### 7.5 Multi-Head Attention & GNNs vs Transformers

GATs can use **multi-head attention**: run $K$ independent attention heads simultaneously, then concatenate their outputs. Each head can specialise — one for structural patterns, another for feature similarity.

**Key connection:** A standard Transformer layer is conceptually a GNN on a *fully-connected graph* (every word attends to every other word).

| Property | Transformer | GNN (GAT) |
|---|---|---|
| Graph type | Fully connected | Sparse, real-world edges |
| Complexity | $O(\|V\|^2)$ | $O(\|V\| + \|E\|)$ |
| How | All nodes attend to all | Attention masked to adjacency |

GNNs recover the expressiveness of Transformers at a fraction of the computational cost for sparse networks.

---

## 8. Improved Update Methods & Over-Smoothing

### 8.1 The Over-Smoothing Problem

Stack too many GNN layers → every node keeps aggregating from its neighbours, who aggregate from their neighbours... After many rounds, **all node representations become virtually identical**. The model can no longer distinguish nodes.

**Mathematical intuition:** For basic GNNs with self-loops, as the number of layers $K \to \infty$, the influence of a node's initial features converges to the stationary distribution of a random walk over the graph. In real-world graphs (*expander graphs*), this happens in just $k = O(\log|V|)$ steps.

### 8.2 Concatenation / Skip-Connections

**GraphSAGE approach:** Instead of adding the neighbourhood message to the node's old features, *concatenate* them to keep self-identity disentangled from neighbour noise:

$$\text{UPDATE}_{\text{concat}}(h_u, m_{\mathcal{N}(u)}) = \left[\text{UPDATE}_{\text{base}}(h_u, m_{\mathcal{N}(u)}) \oplus h_u\right]$$

**Linear interpolation (gating):** Blend old and new with a learnable mixing ratio:

$$\text{UPDATE}_{\text{interp}}(h_u, m_{\mathcal{N}(u)}) = \alpha_1 \circ \text{UPDATE}_{\text{base}}(h_u, m_{\mathcal{N}(u)}) + \alpha_2 \circ h_u$$

where $\alpha_2 = 1 - \alpha_1$. The model learns how much to listen to neighbours vs retain its own memory.

### 8.3 Gated Updates (GRU/LSTM)

Replace the update function entirely with a Gated Recurrent Unit:

$$h_u^{(k)} = \text{GRU}(h_u^{(k-1)},\; m_{\mathcal{N}(u)}^{(k)})$$

GRUs and LSTMs have built-in **forget and input gates** that decide what to drop and what to keep. This allows safely stacking 10+ layers without over-smoothing.

### 8.4 Jumping Knowledge (JK) Connections

**Insight:** Only using the final layer's embeddings may be over-smoothed. Harvest node representations from *every* layer and combine them at the end:

$$z_u = f_{\text{JK}}\!\left(h_u^{(0)} \oplus h_u^{(1)} \oplus \cdots \oplus h_u^{(K)}\right)$$

$f_{\text{JK}}$ can be:
- Simple concatenation of all layers
- Max-pooling across layers (picks strongest signals)
- LSTM attention over the layer sequence

**Analogy:** Like reading your raw notes, first draft, and final essay simultaneously when writing — you capture both fine local detail and broad global structure.

---

## 9. Edge Features & Multi-Relational GNNs

In a **Knowledge Graph** (e.g., a medical database), edges have specific types:
- `(Aspirin) --[TREATS]--> (Headache)`
- `(Aspirin) --[CAUSES]--> (Side Effect)`

These are fundamentally different and must be handled separately.

### 9.1 Relational GCN (RGCN)

Give each relation type $\tau$ its own trainable weight matrix $W_\tau$:

$$m_{\mathcal{N}(u)} = \sum_{\tau \in \mathcal{R}} \sum_{v \in \mathcal{N}_\tau(u)} \frac{W_\tau h_v}{f_n(\mathcal{N}(u), \mathcal{N}(v))}$$

Node $u$ loops through each relation type separately and aggregates using that relation's matrix.

**Parameter explosion problem:** 1,000 relation types → 1,000 separate $W_\tau$ matrices → severe overfitting.

### 9.2 Basis Decomposition (Fix for Parameter Explosion)

Learn a small set of $b$ **basis matrices** $B_1, \ldots, B_b$. Every relation matrix is a weighted combination of them:

$$W_\tau = \sum_{i=1}^{b} \alpha_{i,\tau}\, B_i$$

Only the mixing coefficients $\alpha_{i,\tau}$ are relation-specific → parameter count stays manageable.

### 9.3 Continuous Edge Features

If edges carry numeric values (e.g., distance, weight) rather than discrete types, concatenate the edge feature vector $e_{(u,\tau,v)}$ directly onto the neighbour's embedding before aggregation:

$$\text{aggregate}(h_v \oplus e_{(u,\tau,v)})$$

---

## 10. Graph Pooling — Graph-Level Tasks

To predict a single label for the whole graph, all node embeddings must be collapsed into one master vector $z_G$.

### 10.1 Set Pooling (Sum / Mean)

$$z_G = \frac{\sum_{v \in V} z_u}{f_n(|V|)}$$

Fast and simple. Loses information about how nodes are connected structurally at the last step.

### 10.2 Attention-Based Pooling (Set2Set / SortPool)

An LSTM maintains a running global query vector $q_t$. At each timestep $t$:
1. Every node computes an attention score $a_{v,t}$ based on alignment with $q_t$.
2. Weighted sum of nodes creates a graph snapshot vector $o_t$.

Final graph vector concatenates all snapshots: $z_G = o_1 \oplus o_2 \oplus \cdots \oplus o_T$.

### 10.3 Hierarchical / Graph Coarsening Pooling

Analogous to CNN max-pooling that shrinks images. A learned clustering function maps $n$ nodes into $c$ macro-clusters (where $c \ll n$). The adjacency matrix is compressed:

$$A_{\text{new}} = S^\top A S$$

**Workflow:**
1. Run GNN on full graph.
2. Cluster nodes → shrink graph.
3. Run GNN on coarsened graph.
4. Shrink again.

Extracts community-level structure hierarchically, like a CNN seeing progressively larger receptive fields.

---

## 11. Generalised Message Passing — Unified Framework

(From DeepMind's **Graph Networks** paper)

The most general GNN framework gives **Nodes, Edges, and the whole Graph** their own hidden states, all updated simultaneously at each layer $k$:

### Update Order (Strict)

**Step 1 — Edge Update:**

$$h_{(u,v)}^{(k)} = f_e\!\left(h_{(u,v)}^{(k-1)},\; h_u^{(k-1)},\; h_v^{(k-1)},\; h_G^{(k-1)}\right)$$

Every edge updates based on its own state, the two endpoint nodes, and the global graph state.

**Step 2 — Node Aggregation:**

Each node collects the newly updated edge vectors of all its incident edges.

**Step 3 — Node Update:**

$$h_u^{(k)} = f_n\!\left(h_u^{(k-1)},\; \text{AGG}(\{h_{(u,v)}^{(k)}\}),\; h_G^{(k-1)}\right)$$

**Step 4 — Global Graph Update:**

$$h_G^{(k)} = f_G\!\left(h_G^{(k-1)},\; \text{POOL}(\{h_u^{(k)}\}),\; \text{POOL}(\{h_{(u,v)}^{(k)}\})\right)$$

The master graph vector updates from all newly updated nodes and edges.

**Why this is powerful:** Information flows edges → nodes → graph and back down. The framework subsumes GCN, GAT, RGCN, and almost every other GNN architecture as special cases.

---

## 12. Quick Reference — Architecture Comparison

| Model | Aggregation | Key Idea |
|---|---|---|
| Basic GNN | Sum | Simplest baseline |
| GCN | Symmetric-normalised sum | Spectral approximation, stable |
| GraphSAGE | Mean / concat | Inductive; skip-connection via concat |
| GAT | Attention-weighted sum | Learnable per-neighbour importance |
| RGCN | Per-relation weighted sum | Multi-relational knowledge graphs |
| Deep Sets | Double-MLP sum | Theoretically universal aggregation |
| Janossy | LSTM + permutation avg | Sequence model on sets |

---

## 13. Key Formulas at a Glance

| Concept | Formula |
|---|---|
| Basic GNN layer | $h_u^{(k)} = \sigma(W_{\text{self}} h_u^{(k-1)} + W_{\text{neigh}} \sum_{v} h_v^{(k-1)} + b)$ |
| GCN layer | $h_u^{(k)} = \sigma\!\left(W \sum_{v \in \mathcal{N}(u)\cup\{u\}} \frac{h_v}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(v)|}}\right)$ |
| GAT attention | $\alpha_{u,v} = \text{softmax}_{v}\!\left(a^\top [Wh_u \oplus Wh_v]\right)$ |
| JK connections | $z_u = f_{\text{JK}}(h_u^{(0)} \oplus \cdots \oplus h_u^{(K)})$ |
| RGCN basis decomp | $W_\tau = \sum_i \alpha_{i,\tau} B_i$ |
| Graph pooling | $z_G = \text{POOL}(\{z_u \mid u \in V\})$ |

---

*Notes compiled from: Hamilton GRL Book Chapter 5 · Stanford CS224W · Personal research conversation*
