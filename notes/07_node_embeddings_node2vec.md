# Graph Representation Learning — Notes
### Everything I Learned: DeepWalk · Node2Vec · GNN · GCN

---

## 1. Why Graph Learning?

Traditional machine learning works on flat feature tables:

```
| Age | Salary | Marks |
|-----|--------|-------|
| 25  | 50000  | 88    |
```

But many real problems have **relationships** between entities:

```
Social Network  →  Alice — Bob — Charlie
Drug Discovery  →  Atom  — Bond — Atom
Fraud Detection →  Account — Transaction — Account
```

These are **graphs**. A graph has:
- **Nodes** → the entities (people, atoms, accounts)
- **Edges** → the connections between them

**The problem:** Machine learning models need numbers (vectors), not graphs.

**The solution:** Convert each node into a vector. This is called **Node Embedding** (also called Graph Representation Learning).

---

## 2. Random Walk

A random walk is the basic tool for exploring a graph.

**How it works:**
1. Start at a node
2. Randomly move to one of its neighbors
3. Repeat

**Example:**

```
Graph:
A ----- B
|       |
C ----- D

Start at A. Neighbors = [B, C]. Pick B randomly.
At B. Neighbors = [A, D]. Pick D randomly.
At D. Neighbors = [B, C]. Pick C randomly.

Walk: A → B → D → C
```

> **Key idea:** Nodes that appear **together often** in random walks are **structurally related** in the graph.

Each walk is like a **sentence**, where nodes are **words**. This is exactly what DeepWalk exploits.

---

## 3. DeepWalk

DeepWalk (2014) was the first major algorithm for learning node embeddings.

### Core Idea

```
Graph
  ↓
Generate many Random Walks
  ↓
Treat each Walk as a Sentence
  ↓
Feed into Word2Vec (Skip-Gram)
  ↓
Node Embeddings
```

### The Analogy

| NLP (Text)                  | DeepWalk (Graph)         |
|-----------------------------|--------------------------|
| Word                        | Node                     |
| Sentence                    | Random Walk              |
| Words appearing near each other | Nodes in the same walk |
| Word2Vec                    | DeepWalk trainer         |

### How the Random Walk Works in DeepWalk

At every step: pick **any** neighbor with **equal probability**. This is called a **uniform random walk** — fully random, no bias.

```
At node A with neighbors [B, C]:
  P(go to B) = 1/2
  P(go to C) = 1/2
```

### DeepWalk does NOT use BFS or DFS

It just randomly picks a neighbor at each step. No systematic traversal.

### Training Pairs (Skip-Gram)

From walk `[A, B, D, C]` with window size = 1:

```
(A, B)  ← A appeared near B
(B, D)  ← B appeared near D
(D, C)  ← D appeared near C
```

These pairs are used to train the embeddings.

### What Training Does

For each pair (A, B):
- Push A's embedding **closer** to B's embedding
- Push unrelated nodes' embeddings **farther** apart

---

## 4. Node Embeddings

After training, every node gets a vector:

```
A → [0.3,  0.1,  0.8]
B → [0.2,  0.4,  0.7]
C → [0.9,  0.6,  0.1]
D → [0.1,  0.5,  0.3]
```

These vectors capture the graph structure:
- Nodes with similar neighbors → similar vectors
- Nodes far apart in the graph → different vectors

### How Similarity is Measured

Using the **dot product**:

```
similarity(A, B) = zA · zB
```

Higher dot product = more similar nodes.

### What the Training Objective Is

For a training pair (A, B):

1. **Dot product score** → `score = zA · zB`
2. **Softmax** → converts scores to probabilities → `P(B | A)`
3. **Cross Entropy Loss** → `L = -log P(B | A)`
4. **Gradient Descent** → update zA and zB to minimize L

Goal: make `P(B | A)` close to 1 for co-occurring node pairs.

---

## 5. Node2Vec

Node2Vec (2016) is DeepWalk with **smarter random walks**.

### The Only Difference from DeepWalk

DeepWalk: **uniform** random walk (equal probability to all neighbors)

Node2Vec: **biased** random walk (controlled by p and q)

Everything else — the Word2Vec training, loss, gradient descent — is identical.

### The Two Parameters

| Parameter | Name               | Controls                              |
|-----------|--------------------|---------------------------------------|
| `p`       | Return parameter   | Probability of going **back**         |
| `q`       | Explore parameter  | Probability of going to a **new area** |

### What p and q Do

```
Small p → walk tends to return to where it came from
Large p → walk avoids going back

Small q → walk explores far-away nodes (DFS-like, global)
Large q → walk stays near current node (BFS-like, local)
```

### Important

```
Node2Vec with p=1, q=1  =  DeepWalk
```

DeepWalk is a **special case** of Node2Vec.

### Node2Vec Full Pipeline

```
Graph
  ↓
Biased Random Walks (using p and q)
  ↓
Training Pairs (Skip-Gram window)
  ↓
Dot Product:      score = zA · zB
  ↓
Softmax:          P(B | A)
  ↓
Cross Entropy:    L = -log P(B | A)
  ↓
Backpropagation:  compute gradients
  ↓
Gradient Descent: update zA and zB
  ↓
Repeat → embeddings improve
```

---

## 6. GNN — Graph Neural Network

**GNN is not one algorithm. It is a family of algorithms.**

```
GNN (family)
 ├── GCN         (Graph Convolutional Network)
 ├── GraphSAGE
 ├── GAT         (Graph Attention Network)
 └── GIN         (Graph Isomorphism Network)
```

Just like:

```
Vehicle (family)
 ├── Car
 ├── Bike
 ├── Bus
 └── Truck
```

So: **GCN ⊂ GNN** — GCN is one type of GNN.

### GNN vs DeepWalk / Node2Vec

| Method            | Uses Graph Structure | Uses Node Features |
|-------------------|----------------------|--------------------|
| DeepWalk          | ✅ Yes               | ❌ No              |
| Node2Vec          | ✅ Yes               | ❌ No              |
| GCN / GNN         | ✅ Yes               | ✅ Yes             |

GCN uses **both** the graph connections **and** the actual feature values of each node.

### Core Idea: Message Passing

Every node **collects information from its neighbors**, then **transforms** it.

```
Node A (neighbors: B, C)
  ↓
Gather features of A, B, C
  ↓
Average (or sum) them
  ↓
Multiply by Weight Matrix
  ↓
Apply ReLU
  ↓
New embedding of A
```

---

## 7. GCN — Graph Convolutional Network

### The Formula

```
H(l+1) = σ( Â · H(l) · W(l) )
```

| Symbol    | Meaning                                     |
|-----------|---------------------------------------------|
| `H(l)`    | Node feature matrix at layer l              |
| `W(l)`    | Trainable weight matrix at layer l          |
| `Â`       | Adjacency matrix with self-loops added      |
| `σ`       | Activation function (usually ReLU)          |
| `H(l+1)`  | New node embeddings after this layer        |

### What Each Part Does

**`Â · H` — Neighbor Aggregation**

For node A with neighbors B and C:
```
new_A = features(A) + features(B) + features(C)
```
This is what the matrix multiplication `Â @ X` computes for every node at once.

**`(Â · H) · W` — Linear Transformation**

Multiply by a learned weight matrix to transform the aggregated features.

**`σ(...)` — ReLU Activation**

```
ReLU(x) = max(0, x)
```
Sets negative values to 0. Adds non-linearity so the model can learn complex patterns.

### Self-Loops

The `Â` matrix has **self-loops added** — meaning each node also includes its own features when aggregating, not just its neighbors.

---

## 8. Complete Comparison

### Side by Side

```
┌──────────────┬───────────────────────────────┬────────────────────────┐
│  Algorithm   │  How It Works                 │  Needs                 │
├──────────────┼───────────────────────────────┼────────────────────────┤
│ DeepWalk     │ Uniform walks → Word2Vec      │ Graph structure only   │
│ Node2Vec     │ Biased walks (p,q) → Word2Vec │ Graph structure only   │
│ GCN          │ A @ X @ W → ReLU              │ Graph + Node features  │
└──────────────┴───────────────────────────────┴────────────────────────┘
```

### Timeline

```
2014  DeepWalk   →  Random walks + Word2Vec
2016  Node2Vec   →  Biased walks (p, q) + Word2Vec
2017  GCN        →  Message passing + neural layers
2017  GraphSAGE  →  Inductive GCN (works on unseen nodes)
2018  GAT        →  Attention-based neighbor aggregation
```

### All Three Produce the Same Thing

```
DeepWalk  →  Node Embeddings
Node2Vec  →  Node Embeddings
GCN       →  Node Embeddings
```

Just through different methods.

---

## 9. Key Takeaways

```
1. DeepWalk = Random Walks + Word2Vec

2. Node2Vec = DeepWalk + smarter walks (p and q)
             Node2Vec with p=1, q=1 = DeepWalk

3. GNN = family of models (GCN, GraphSAGE, GAT, GIN)
         GCN is one member

4. GCN = uses actual node features + graph structure
         Does NOT use random walks

5. All methods output: Node Embeddings
```

---

## Real World Uses

| Application         | What the Graph Looks Like           |
|---------------------|-------------------------------------|
| Drug Discovery      | Atoms = nodes, Bonds = edges        |
| Fraud Detection     | Accounts = nodes, Transactions = edges |
| Recommendation      | Users + Items = nodes, Interactions = edges |
| Traffic Prediction  | Road intersections = nodes, Roads = edges |
| Social Networks     | People = nodes, Friendships = edges |

---

## 10. Word2Vec — The Engine Inside DeepWalk

Before understanding DeepWalk deeply, you need to understand Word2Vec. DeepWalk is literally just:

```
Random Walk Generator  +  Word2Vec
```

Nothing more.

### What Problem Does Word2Vec Solve?

A computer sees words as plain strings — `"love"`, `"machine"`, `"learning"` mean nothing to it numerically. Word2Vec converts each word into a vector (a list of numbers):

```
love     → [0.2, 0.8]
machine  → [0.4, 0.3]
learning → [0.5, 0.2]
```

These vectors are called **Word Embeddings**.

### The Main Idea

> Words appearing in similar contexts should have similar vectors.

Example:

```
I love machine learning
I enjoy machine learning
```

Both `love` and `enjoy` appear in the same position around the same words. So Word2Vec learns:

```
love ≈ enjoy   (similar vectors)
```

### Skip-Gram (The version DeepWalk uses)

Given a **center word**, predict the **neighboring words**.

Sentence: `I love machine learning` | Window size = 1

| Center Word | Neighbors     | Training Pairs          |
|-------------|---------------|-------------------------|
| love        | I, machine    | (love, I), (love, machine) |
| machine     | love, learning | (machine, love), (machine, learning) |

### DeepWalk Does Exactly the Same Thing

Random Walk: `A B D C` | Window size = 1

Training pairs:

```
(A, B)
(B, A)
(B, D)
(D, B)
(D, C)
(C, D)
```

Node = Word. Walk = Sentence. Identical process.

---

## 11. Word2Vec Math — Step by Step

### Setup

Graph has 3 nodes: A, B, C

Initial embeddings (random):

```
A = [0.2, 0.1]
B = [0.3, 0.5]
C = [0.7, 0.2]
```

Training pair: **(A, B)** — meaning: given A, the correct answer is B.

---

### Step 1 — Compute Dot Product Scores

For center node A, compute a score with every other node using the **dot product**:

```
score(A, B) = zA · zB
            = (0.2 × 0.3) + (0.1 × 0.5)
            = 0.06 + 0.05
            = 0.11

score(A, C) = zA · zC
            = (0.2 × 0.7) + (0.1 × 0.2)
            = 0.14 + 0.02
            = 0.16
```

Higher score = model currently thinks these nodes are more related.

---

### Step 2 — Softmax (Convert Scores to Probabilities)

```
P(B | A) = e^0.11 / (e^0.11 + e^0.16)
         = 1.116  / (1.116 + 1.173)
         = 1.116  / 2.289
         = 0.487  (48.7%)

P(C | A) = 1.173 / 2.289
         = 0.513  (51.3%)
```

The model currently says C is slightly more likely than B — which is **wrong** (our training pair says B is the correct neighbor).

---

### Step 3 — Cross Entropy Loss

We know B is the correct answer. So:

```
Target:     [1,   0  ]   ← B should be 100%, C should be 0%
Prediction: [0.487, 0.513]
```

Loss:

```
L = -log(P(B | A))
  = -log(0.487)
  = 0.719
```

Large loss = bad prediction. We want this to decrease.

---

### Step 4 — Backpropagation (Update Embeddings)

Goal after seeing the error:
- **Increase** score(A, B) → push A and B closer
- **Decrease** score(A, C) → push A and C apart

Embeddings update (approximate direction, not exact values):

```
Before:
  A = [0.2,  0.1 ]
  B = [0.3,  0.5 ]

After one gradient step:
  A = [0.25, 0.15]
  B = [0.35, 0.55]
```

Now `zA · zB` is larger than before → model is slightly more correct.

---

### Why This Works Over Many Steps

Suppose random walks generate the pair (A, B) many times:

```
Walk 1: A → B → ...
Walk 2: A → B → ...
Walk 3: A → B → ...
```

Every time, the training pushes A and B's embeddings closer. After thousands of steps:

```
A and B appear together often
          ↓
zA · zB becomes very large
          ↓
A and B are close in embedding space
```

---

### Complete DeepWalk Pipeline (with Math)

```
Step 1  Graph
          ↓
Step 2  Generate Random Walks
        A B A C
        A C A B
          ↓
Step 3  Treat Walks as Sentences
        Sentence: [A, B, A, C]
          ↓
Step 4  Generate Skip-Gram Pairs
        (A,B), (B,A), (A,C), (C,A)
          ↓
Step 5  Initialize Embeddings (random)
        A=[0.1,0.2]  B=[0.5,0.1]  C=[0.3,0.4]
          ↓
Step 6  Dot Product Score
        score = zA · zB
          ↓
Step 7  Softmax
        P(v | u)
          ↓
Step 8  Cross Entropy Loss
        L = -log(P(v | u))
          ↓
Step 9  Backpropagation → update embeddings
          ↓
Step 10 Repeat thousands of times
          ↓
        Nodes appearing together in walks
        → have similar embeddings
```

### The Beautiful Summary

```
DeepWalk = Random Walk Generator + Word2Vec

That's it. Nothing more.
```