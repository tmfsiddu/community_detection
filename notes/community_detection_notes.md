# Community Detection in Network Science
### Complete Study Notes

---

## Table of Contents

1. [What is a Community?](#1-what-is-a-community)
2. [Belgium Case Study](#2-belgium-case-study--the-real-world-example)
3. [Zachary's Karate Club](#3-zacharys-karate-club)
4. [Connectedness & Density Hypothesis](#4-connectedness--density-hypothesis)
5. [Types of Communities](#5-types-of-communities)
6. [Number of Communities — The Explosion Problem](#6-number-of-communities--the-explosion-problem)
7. [Hierarchical Clustering (Ravasz Algorithm)](#7-hierarchical-clustering-ravasz-algorithm)
8. [Modularity](#8-modularity)
9. [Greedy Algorithm](#9-greedy-algorithm)
10. [Louvain Algorithm](#10-louvain-algorithm)
11. [Limitations of Modularity](#11-limitations-of-modularity)
12. [Quick Reference Summary](#12-quick-reference-summary)

---

## 1. What is a Community?

A **community** in a network is a group of nodes that:

- Have **more connections with each other** (internally dense)
- Have **fewer connections with outsiders** (external sparse)

### Real-Life Analogy

```
[ Company A employees ]       [ Company B employees ]
   many internal meetings           many internal meetings
        very few cross-company interactions
```

Employees within a company talk more with each other than with people from other companies — they form a **community**.

---

## 2. Belgium Case Study — The Real World Example

### Background

Belgium has two major language groups:

| Group    | Language | Population |
|----------|----------|------------|
| Flemish  | Dutch    | ~59%       |
| Walloons | French   | ~40%       |

**Question asked:** Do these two groups freely mix with each other, or do they stay separate?

### What Researchers Did

**Researcher:** Vincent Blondel (2007)

They used **mobile phone call data** to build a social network:

```
    Person = Node
    Phone call = Edge
    Frequent callers = placed closer together
```

### How the Network Was Built

```
     [Person A] ──────── [Person B]
          │                   │
          │    frequent        │
          │    calls           │
          └──── [Person C] ───┘
```

### Community Detection Algorithm Applied

The algorithm discovered **two large clusters**:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ┌──────────────────────┐  ┌──────────────────────┐ │
│  │   DUTCH SPEAKERS     │  │   FRENCH SPEAKERS    │ │
│  │   (Flemish cluster)  │  │   (Walloon cluster)  │ │
│  │                      │  │                      │ │
│  │  ████ many calls ███ │  │  ████ many calls ███ │ │
│  │      within group    │  │      within group    │ │
│  └──────────┬───────────┘  └──────────┬───────────┘ │
│             │   very few cross-calls  │             │
│             └─────────────────────────┘             │
│                   [Brussels - mixed]                │
└─────────────────────────────────────────────────────┘
```

### Key Insight

> Belgium looks **united politically**, but its social network shows people mainly interact **within their own language community**.

---

## 3. Zachary's Karate Club

A famous small network used to **test community detection algorithms**.

### Setup

- 34 members in a karate club
- 78 links between members who interacted outside the club
- A **conflict** between the president and instructor split the club into two groups

### Actual Network (from your dataset)

![Zachary's Karate Club Graph](karate_club_graph.png)

> **Blue nodes** = Mr. Hi's (Instructor's) faction  
> **Red nodes** = Officer's (President's) faction  
> Node **0** = Instructor, Node **33** = President (the two hubs)

### After the Split (Ground Truth)

```
  Group 1 (Instructor's / Mr. Hi's followers)  |  Group 2 (President's / Officer's followers)
  ──────────────────────────────────────────    |  ──────────────────────────────────────────
  Nodes: 0,1,2,3,4,5,6,7,8,10,11,12,           |  Nodes: 9,14,15,18,20,22,23,24,25,
         13,17,19,21                            |         26,27,28,29,30,31,32,33
```

**Why is this important?**

Community detection algorithms are tested by checking if they can **predict this split** correctly, just from the network structure — without knowing about the conflict!

---

## 4. Connectedness & Density Hypothesis

### The Core Hypothesis (H2)

> **A community is a locally dense connected subgraph in a network.**

This has **two parts**:

### Part A — Connectedness Hypothesis

All members must be **reachable through other members** of the same community.

```
✅ VALID COMMUNITY            ❌ INVALID COMMUNITY

   A ── B ── C ── D              A ── B     C ── D
   (all connected)               (disconnected — two separate parts)
```

### Part B — Density Hypothesis

Nodes inside a community should connect **more internally** than externally.

```
    Internal = many connections         External = few connections

    ●──●──●                                  ●──●──●
    │\ │ /│                                       │
    │ \│/ │                                       │
    ●──●──●   ·····················  ●──●──●──●──●
    (dense)        (sparse bridge)      (another dense group)
```

---

## 5. Types of Communities

### 5.1 Clique (Strictest Definition)

A clique = **every node connected to every other node** (complete subgraph).

```
        A
       /|\
      / | \
     B──┼──C
      \ | /
       \|/
        D

Every node A,B,C,D is connected to all others → CLIQUE
```

**Problem with Cliques:**
- Real communities are rarely perfect cliques
- Large cliques are very rare in real networks
- Too restrictive — misses many real communities

---

### 5.2 Internal & External Degree

For a node `i` inside community `C`:

```
┌─────────────────────────────────────────────┐
│  COMMUNITY C                                │
│                                             │
│   [A] ──── [B] ──── [C]                    │
│    │          ╲                             │
│    │            ╲                           │
│  k_int (A) = 1   ╲── [X]  ← OUTSIDE        │
│  k_ext (A) = 1                              │
└─────────────────────────────────────────────┘

k_int = links going INSIDE the community
k_ext = links going OUTSIDE the community
```

---

### 5.3 Strong Community

> Every **individual node** must satisfy: `k_int > k_ext`

```
Strong Community Check (per node):

Node A: internal links = 3, external links = 1 ✅ (3 > 1)
Node B: internal links = 4, external links = 0 ✅ (4 > 0)
Node C: internal links = 2, external links = 1 ✅ (2 > 1)

All pass → STRONG COMMUNITY
```

---

### 5.4 Weak Community

> The **group as a whole** must satisfy: `Σ k_int > Σ k_ext`

```
Weak Community Check (for the whole group):

Node A: internal = 3, external = 1
Node B: internal = 4, external = 0
Node C: internal = 1, external = 3 ← fails individually

Total internal = 3+4+1 = 8
Total external = 1+0+3 = 4

8 > 4 → WEAK COMMUNITY (still qualifies!)
```

---

### Hierarchy of Definitions

```
  CLIQUE  ⊂  STRONG COMMUNITY  ⊂  WEAK COMMUNITY
  (most strict)                    (most relaxed)

  Every clique is a strong community.
  Every strong community is a weak community.
  But NOT the other way around.
```

---

## 6. Number of Communities — The Explosion Problem

### Graph Bisection

The simplest problem: **divide a network into two groups** with minimum links cut between them (cut size).

```
Group 1: A ── B ── C
                    │
                    │ ← CUT SIZE = 1 (only this link between groups)
                    │
Group 2:       D ── E ── F
```

### Why Brute Force Fails

**Formula for number of ways to split N nodes into 2 groups:**

```
        N!
    ─────────
    N1! × N2!
```

| Network Size | Possible Splits | Time to Check |
|-------------|----------------|---------------|
| 10 nodes    | 252            | ~1 ms         |
| 100 nodes   | ~10²⁹          | ~10¹⁶ years   |
| 1000 nodes  | astronomical   | impossible    |

> The number of partitions grows **exponentially** with network size.

### Community Detection is Even Harder

In community detection, we don't even know **how many groups** to make. The number of possible partitions is given by the **Bell Number**:

```
B_N grows FASTER than exponential

For N = 50 nodes → more than 10^40 possible partitions

Impossible to check one by one.
```

This is why we need **smart algorithms** instead of brute force.

---

## 7. Hierarchical Clustering (Ravasz Algorithm)

### Core Idea

> Group similar nodes together step by step.

### Similarity = Topological Overlap

Two nodes are **similar** if they:
- Are directly connected
- Share many common neighbors

```
HIGH SIMILARITY:                   LOW SIMILARITY:

    A ── B                             A           E
   /|\ /|\                            │           │
  C D E F G                           B           F
  (A and B share C,D,E,F,G)           (no common neighbors)
```

### Steps of the Algorithm

```
STEP 1: Each node starts as its own community

   [A]  [B]  [C]  [D]  [E]

STEP 2: Find most similar pair → MERGE them

   [A,B]  [C]  [D]  [E]

STEP 3: Recalculate similarities → MERGE again

   [A,B,C]  [D,E]

STEP 4: Continue...

   [A,B,C,D,E]  (all merged)
```

### Dendrogram (Merge Tree)

```
Height
  │
5 │                 ┌────────────────┐
  │                 │                │
4 │          ┌──────┤           ┌────┤
  │          │      │           │    │
3 │    ┌─────┤      │     ┌─────┤    │
  │    │     │      │     │     │    │
2 │  ┌─┤     │    ┌─┤   ┌─┤   ┌─┤   │
  │  │ │     │    │ │   │ │   │ │   │
  A  B  C    D    E  F   G  H  I  J

Cut at height 3 → gives 2 communities: {A,B,C,D} and {E,F,G,H,I,J}
Cut at height 2 → gives 4 smaller communities
```

### How to Find Communities from Dendrogram

- **Cut LOW** → more, smaller communities
- **Cut HIGH** → fewer, larger communities

### Computational Complexity

```
Brute Force:   EXPONENTIAL time   → impossible for large networks
Hierarchical:  O(N²) time         → much faster, practical
```

---

## 8. Modularity

### The Core Question

> Is this partition **better** than a random wiring?

### Formula

For a community `C` with `L_c` internal links and total degree `k_c`:

```
          L_c        k_c  ²
M_c  =   ─────  -  (────)
           L         2L
```

Where:
- `L` = total links in the whole network
- `L_c` = links inside community C
- `k_c` = sum of degrees of nodes in C

### Summing Over All Communities

```
         nc   [  L_c        k_c  ² ]
M    =   Σ    [ ─────  -  (────)   ]
        c=1   [  L          2L     ]
```

### Visual Examples of Modularity Values

![Modularity Examples](modularity_examples.png)

### What Modularity Values Mean

```
M > 0  →  More links than expected by chance  →  REAL COMMUNITY exists

M = 0  →  Same as random  →  No community structure

M < 0  →  Fewer links than expected  →  NOT a community
```

### Visual Intuition

```
GOOD PARTITION (high M):           BAD PARTITION (low M):

 ●──●──●     ●──●──●               ●──●  │  ●──●──●
 │\ │ /│     │\ │ /│               │  │  │  │     │
 ●──●──●     ●──●──●               ●──●──┼──●──●──●
  [Group 1]  [Group 2]             (cut through middle)

M = 0.41 (good)                    M much lower (bad)
```

### Key Properties of M

| Situation | M value |
|-----------|---------|
| Whole network = 1 community | M = 0 |
| Each node = separate community | M < 0 |
| Perfect community structure | M close to 1 |
| Maximum possible | M < 1 |

---

## 9. Greedy Algorithm

### Hypothesis (H4)

> The partition with the **highest modularity** is the best community structure.

### Algorithm Steps

```
STEP 1: Each node = its own community
         [A]  [B]  [C]  [D]  [E]

STEP 2: Try ALL connected pairs
         Ask: "Does merging A+B increase M?"
         Calculate ΔM for each possible merge.

STEP 3: Perform the merge with LARGEST ΔM (if ΔM > 0)
         [A,B]  [C]  [D]  [E]

STEP 4: Repeat...
         [A,B,C]  [D,E]

STEP 5: Record M at every step.
         Choose the step with MAXIMUM M as final answer.
```

### Important Note

The algorithm does **NOT stop** when modularity starts decreasing.  
It **keeps going** to the end, tracks all M values, then picks the best.

### Example: Physicist Collaboration Network

Applied to **56,276 physicists** connected by co-authored papers:

```
Result: ~600 communities discovered

Most physicists in same community worked in same field:

Community 1: Condensed matter physicists (~93% same field)
Community 2: High-energy physicists
Community 3: Astrophysicists
...
```

### Complexity

```
Basic Greedy:     O(N²)
Optimized:        O(N log² N)   ← much faster
```

---

## 10. Louvain Algorithm

The Louvain algorithm is one of the **fastest and most popular** community detection methods.

**Complexity: O(L)** where L = number of links → works on millions of nodes.

### Louvain Method — Visual Diagram

![Louvain Hierarchical Agglomeration](louvain_diagram.png)

> **Top:** Original network with nodes colored by community after Phase 1  
> **Bottom left → middle:** After 1st pass — 4 supernodes (green=14, red=16, cyan=2, blue=4 internal links)  
> **Middle → right:** After 2nd pass — 2 supernodes (green=26, cyan=24 internal links)

### Overview

Louvain works in **two repeating phases**:

```
┌─────────────────────────────────────────────────────────┐
│  PHASE 1: Modularity Optimization                       │
│  Move individual nodes between communities              │
│  if modularity increases                                │
├─────────────────────────────────────────────────────────┤
│  PHASE 2: Community Aggregation                         │
│  Compress each community into ONE supernode             │
│  Build a smaller network                                │
└─────────────────────────────────────────────────────────┘
                     ↕  REPEAT
```

---

### Phase 1 — Modularity Optimization in Detail

**Starting point:** Every node is its own community.

```
Initial state:
[A]  [B]  [C]  [D]  [E]  [F]  [G]  [H]
```

For **each node**, ask:

> "If I move this node into my neighbor's community, does ΔQ > 0?"

```
       Neighbor's community
            ┌────────┐
  Node i ──►│ Try    │  → Calculate ΔQ
            │joining │
            └────────┘

If ΔQ > 0 → MOVE
If ΔQ ≤ 0 → STAY
```

Keep doing this until **no node wants to move** (local maximum reached).

**Result of Phase 1:**

```
Before Phase 1:
[A]  [B]  [C]  [D]  [E]  [F]  [G]  [H]

After Phase 1:
[A,B,C]     [D,E]      [F,G,H]
(green)     (blue)     (red)
```

---

### Phase 2 — Community Aggregation (The Key Transition)

**This is the most important step to understand.**

Each community from Phase 1 becomes **ONE supernode**.

```
BEFORE PHASE 2 (result of Phase 1):

    ●──●──●           ●──●           ●──●──●
    │\ │ /│           │ /            │\ │ /│
    ●──●──●           ●              ●──●──●
   Community A       Community B    Community C
   (many nodes)     (few nodes)    (many nodes)

AFTER PHASE 2 (compressed graph):

       ╔═══╗          ╔═══╗          ╔═══╗
       ║ A ║──────────║ B ║──────────║ C ║
       ╚═══╝          ╚═══╝          ╚═══╝
    (supernode)    (supernode)    (supernode)
```

### What Do Edge Weights Mean?

```
Between supernodes:
═══ A ═══ ─────4───── ═══ B ═══
                 ↑
     4 links existed between
     communities A and B
     → edge weight = 4

Inside supernodes (self-loops):
═══ A ═══ ↺
      ↑
  14 internal links inside A
  → self-loop weight = 14
```

### How 4 Nodes Become 2 Nodes

This is exactly what happens in the Louvain phases:

```
ORIGINAL NETWORK:
Many individual nodes

     ↓ PHASE 1

4 Communities discovered:
  ╔══════╗   ╔══════╗   ╔══════╗   ╔══════╗
  ║Green ║   ║Blue  ║   ║Red   ║   ║Cyan  ║
  ╚══════╝   ╚══════╝   ╚══════╝   ╚══════╝

     ↓ PHASE 2 (compress into supernodes)

Compressed graph (4 supernodes):
  [Green]─────[Blue]─────[Red]─────[Cyan]

     ↓ PHASE 1 AGAIN on compressed graph

Algorithm discovers:
  Green + Blue belong together
  Red + Cyan belong together

     ↓ PHASE 2 AGAIN

Final result (2 supernodes):
  ╔══════════════╗         ╔══════════════╗
  ║ Green + Blue ║─────────║ Red + Cyan   ║
  ╚══════════════╝         ╚══════════════╝

4 communities → 2 larger communities
```

### Why This Happens

When the compressed graph runs Phase 1 again:
- The 4 supernodes are treated like ordinary nodes
- The algorithm checks: "Would merging these supernodes increase modularity?"
- If **Green↔Blue** and **Red↔Cyan** have strong connections → they merge
- Result: **4 → 2 communities**

---

### Full Louvain Flow Diagram

```
ORIGINAL GRAPH
(N individual nodes)
        │
        ▼
┌───────────────┐
│   PHASE 1     │ ← Move nodes, maximize modularity locally
│ Find small    │
│ communities   │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   PHASE 2     │ ← Compress: community = 1 supernode
│ Build smaller │ ← Add self-loops for internal links
│ graph         │ ← Add weighted edges for inter-community links
└───────┬───────┘
        │
        ▼
  Is modularity still improving?
        │
   Yes  │           No
        ▼           ▼
  Repeat Phase 1  DONE → Final partition
  on smaller graph
```

---

## 11. Limitations of Modularity

### 11.1 Resolution Limit

**Problem:** Modularity sometimes **merges small communities** into larger ones, even when they are actually separate.

```
REALITY:                              WHAT MODULARITY SEES:

  [A]──[B]──[C]       [D]──[E]         [A,B,C,D,E]
  (small community)  (small community)  (merged incorrectly!)

If communities are small enough,
even one link between them may cause merging.
```

**Resolution Threshold:**

```
If community size k ≤ √(2L)
→ modularity optimization may FAIL to detect it as separate
```

### 11.2 Modularity Plateau

**Problem:** Many different partitions can have **almost the same modularity**.

```
Expected:              Reality:
  M                       M
  │   *                   │  ████████████
  │  * *                  │ █           █
  │ *   *                 │█             █
  │*     *                │               █
  └──────────            └──────────────────
    (one sharp peak)        (flat plateau)
                            Many partitions are equally "good"
```

This means:
- Different algorithms may give **different** communities
- Yet all have **similar** modularity values
- There is **no unique correct answer**

### 11.3 Random Networks Can Show High Modularity

Even completely **random networks** can accidentally have high-M partitions.  
So high M alone doesn't prove real communities exist.

---

### Alternative: Infomap Algorithm

Instead of maximizing M, Infomap **minimizes** the Map Equation (L).

Uses **information theory + random walks:**

```
Intuition:
If a random walker keeps getting "trapped"
inside a group for a long time
→ that group is a community
```

---

## 12. Quick Reference Summary

### Community Detection Methods Compared

| Method | Main Idea | Speed | Best For |
|--------|-----------|-------|----------|
| Brute Force | Check all partitions | Impossible | Not usable |
| Hierarchical Clustering | Merge similar nodes | O(N²) | Medium networks |
| Greedy Modularity | Merge if M increases | O(N log² N) | Large networks |
| **Louvain** | Phase 1 + Phase 2 iteratively | **O(L)** | **Very large networks** |
| Infomap | Minimize map equation | Fast | Hierarchical structures |

---

### Community Definitions Compared

```
     STRICTEST                              MOST RELAXED
     ────────────────────────────────────────────────────►
     CLIQUE     STRONG COMMUNITY     WEAK COMMUNITY
     (all         (every node:         (whole group:
     connected)   k_int > k_ext)       Σk_int > Σk_ext)
```

---

### Modularity at a Glance

```
M > 0  →  Real community (more links than random)
M = 0  →  No structure (same as random)
M < 0  →  Anti-community (fewer links than random)

Best partition = partition with highest M
But watch out: resolution limit + modularity plateau
```

---

### Louvain Phase Transition Summary

```
Step 1: All nodes = individual communities
Step 2: Phase 1 → small communities form (local optimization)
Step 3: Phase 2 → each community = one supernode
Step 4: Phase 1 again → supernodes merge into larger communities
Step 5: Phase 2 again → compress again
Step 6: Repeat until no improvement
```

---

*Notes compiled from Network Science (Barabási) — Chapter 9: Communities*