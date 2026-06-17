# Community Detection & Graph Representation Learning

A comprehensive self-study repository documenting a complete learning journey from graph fundamentals and classical community detection through spectral methods, shallow node embeddings, and modern Graph Neural Networks — culminating in a full GNN pipeline on a real-world dataset.

---

## Repository Structure

```
communityDetection/
│
├── notebooks/                         # Six progressive Jupyter notebooks
│   ├── 01_networkx_basics.ipynb
│   ├── 02_community_detection_karate.ipynb
│   ├── 03_SBM_spectral_clustering.ipynb
│   ├── 04_shallow_node_embeddings.ipynb
│   ├── 05_Basic_gnn_implementation.ipynb
│   └── 06_gnn_dataset_implementation.ipynb
│
├── src/                               # Modular Python source files
│   ├── graph_creation_01.py
│   ├── graph_metrics_02.py
│   ├── graph_visualization_03.py
│   ├── centrality_measures_04.py
│   ├── connectivity_analysis_05.py
│   ├── shortest_paths_06.py
│   ├── sbm_utils_07.py
│   ├── walk_utils_08.py
│   ├── embeddings_utils_09.py
│   ├── evaluation_utils_10.py
│   └── gnn_dataset_implementation/    # Package for the final GNN pipeline
│       ├── __init__.py
│       ├── dataset.py
│       ├── models.py
│       ├── train_eval.py
│       └── recommend.py
│
├── notes/                             # Self-prepared study notes (Markdown)
│   ├── 01_graph_fundementals_notes.md
│   ├── 02_centrality_notes.md
│   ├── 03_community_detection_karate_notes.md
│   ├── 04_community_detection_notes.md
│   ├── 05_laplacian_matrices_notes.md
│   ├── 06_spectral_clustering_GCN_notes.md
│   ├── 07_node_embeddings_node2vec.md
│   └── 08_GNN_Notes.md
│
├── presentations/
│   └── 01_graph_shallow_embeddings.pptx   # Academic presentation (29 slides)
│
├── results/
│   └── 01_SBM_Spectral_Clustering_Report.pdf
│
├── images/                            # All generated visualizations
├── requirements.txt
└── README.md
```

---

## Notebooks

### 01 — NetworkX Basics

**File:** `notebooks/01_networkx_basics.ipynb`  
**Source module:** `src/graph_creation_01.py`, `src/graph_metrics_02.py`, `src/graph_visualization_03.py`

Covers all foundational graph theory concepts implemented from scratch with NetworkX:

- Creating undirected, directed, and weighted graphs
- Computing adjacency, degree, and Laplacian matrices
- Core graph statistics: node count, edge count, average degree, density, diameter
- Eulerian path/circuit detection
- Visualisation: spring layout, circular layout, spectral layout comparison
- Degree and betweenness centrality (visual comparison)
- Shortest paths (Dijkstra / BFS)
- Connected components, bridges, and articulation points
- Cliques and the largest clique on the Karate Club graph
- Spanning trees

---

### 02 — Community Detection on Zachary's Karate Club

**File:** `notebooks/02_community_detection_karate.ipynb`  
**Source module:** `src/centrality_measures_04.py`, `src/connectivity_analysis_05.py`

Implements and compares **six community detection algorithms** on the classic Zachary's Karate Club benchmark (34 nodes, 78 edges, known ground-truth split):

| # | Algorithm | Type | Key idea |
|---|-----------|------|----------|
| 1 | **Girvan-Newman** | Divisive | Iteratively remove highest-betweenness edges |
| 2 | **Greedy Modularity** | Agglomerative | Merge nodes/communities that maximise modularity Q |
| 3 | **Louvain** | Hierarchical | Two-phase local modularity optimisation |
| 4 | **Label Propagation** | Propagation | Nodes adopt the majority label of their neighbours |
| 5 | **K-Clique Percolation** | Clique-based | Communities as adjacent overlapping k-cliques |
| 6 | **Spectral Clustering** | Algebraic | K-Means on Laplacian eigenvectors |

Each algorithm is visualised with a colour-coded graph and compared against the ground-truth communities. An ARI (Adjusted Rand Index) comparison bar chart is produced across all six methods.

---

### 03 — Spectral Clustering on Stochastic Block Models

**File:** `notebooks/03_SBM_spectral_clustering (2).ipynb`  
**Source module:** `src/sbm_utils_07.py`, `src/evaluation_utils_10.py`

Explores why spectral clustering works through controlled synthetic experiments:

- **Stochastic Block Models (SBM):** generate graphs with known community structure using within-block probability `p_in` and between-block probability `p_out`
- **Graph Laplacian:** construction and properties (unnormalised and normalised forms)
- **Eigenvalue decomposition:** eigengap heuristic for choosing the number of communities K
- **Spectral partitioning:** running K-Means on the leading eigenvectors
- **ARI vs node count experiments:** how community recovery degrades as inter-community edges increase
- Full accessible PDF report generated at `results/01_SBM_Spectral_Clustering_Report.pdf`

---

### 04 — Shallow Node Embeddings: DeepWalk & Node2Vec

**File:** `notebooks/04_shallow_node_embeddings.ipynb`  
**Source modules:** `src/walk_utils_08.py`, `src/embeddings_utils_09.py`, `src/evaluation_utils_10.py`

Implements the two foundational graph embedding methods from scratch, built on a simple 4-node graph (A–B–C–D):

**Random Walks (`walk_utils_08.py`)**
- Uniform random walks (DeepWalk)
- Biased random walks with `p` (return) and `q` (in-out) parameters (Node2Vec)

**Word2Vec Training (`embeddings_utils_09.py`)**
- Skip-Gram model via Gensim `Word2Vec`
- Treats each walk as a sentence and each node as a word
- Shared training pipeline for both DeepWalk and Node2Vec

**Evaluation (`evaluation_utils_10.py`)**
- Spectral Clustering as a baseline
- K-Means on embeddings + ARI and NMI scoring
- PCA and t-SNE 2D projection for visual comparison of all three methods

---

### 05 — Basic GNN Implementation (Karate Club)

**File:** `notebooks/05_Basic_gnn_implementation.ipynb`

Builds GNNs from the ground up on the Karate Club dataset, one concept per cell:

- **Message Passing by hand:** manually computing one round of neighbourhood aggregation
- **GCNConv layer:** graph convolution, normalised aggregation, ReLU, dropout
- **GATConv layer:** attention weights per edge, multi-head attention
- **Over-smoothing demonstration:** tracking embedding variance as depth increases (with saved plot `images/over_smoothing.png`)
- **Jumping Knowledge (JK-Net):** concatenating all layer outputs to preserve multi-scale information
- **t-SNE visualisation:** projecting learned embeddings to 2D to see cluster separation (`images/tsne_gcn.png`)
- **Training curves:** loss and validation accuracy logged across epochs (`images/gcn_loss.png`)

---

### 06 — GNN Pipeline on Amazon Computers Dataset

**File:** `notebooks/06_gnn_dataset_implementation.ipynb`  
**Package:** `src/gnn_dataset_implementation/`

A complete end-to-end GNN research pipeline on the **Amazon Computers** co-purchase graph (~13k nodes, ~491k edges, 10 product categories), demonstrating three models with progressively better design:

#### Dataset (`dataset.py`)
- Downloads Amazon Computers via PyTorch Geometric's `Amazon` loader
- Creates reproducible 60/20/20 train/val/test masks with `torch.manual_seed`

#### Models (`models.py`)

| Model | Layers | Design | Expected failure |
|-------|--------|--------|-----------------|
| **DeepGCN** | 6× GCNConv | No skip connections | Over-smoothing — embeddings collapse |
| **StandardGAT** | 2× GATConv (8 heads) | Multi-head attention | Overfitting on dense graph |
| **ResAttJKNet** | 3× dual blocks | GCN + GAT + Residual + JK | — (proposed fix) |

**ResAttJKNet** (novel architecture) fuses three ideas:
1. **Dual-Perspective Blocks** — GCNConv (structural) and GATConv (semantic) run in parallel per layer; their outputs are averaged
2. **Residual Connections** — `output = fused + input`, preventing over-smoothing and vanishing gradients
3. **Jumping Knowledge** — all layer outputs are concatenated into a master embedding so the classifier sees both local and global structure

#### Training & Evaluation (`train_eval.py`)
- `train_model()` — Adam optimiser, cross-entropy loss, val accuracy logged every 30 epochs
- `evaluate_classification()` — test accuracy on held-out nodes
- `evaluate_clustering()` — K-Means on extracted embeddings + ARI against ground-truth categories; uses `return_embedding=True` for ResAttJKNet and penultimate-layer extraction for the baselines

#### Recommendation (`recommend.py`)
- `recommend_products()` — extracts JK embeddings, L2-normalises them, and returns top-K most similar products to a target node by cosine similarity

---

## Source Modules (`src/`)

| File | Purpose |
|------|---------|
| `graph_creation_01.py` | Build undirected, directed, weighted graphs with NetworkX |
| `graph_metrics_02.py` | Adjacency/degree/Laplacian matrices, basic stats, Eulerian checks |
| `graph_visualization_03.py` | Colour-coded graph drawing with legend patches |
| `centrality_measures_04.py` | Degree, closeness, betweenness, eigenvector centrality |
| `connectivity_analysis_05.py` | Cliques, components, bridges, articulation points, connectivity |
| `shortest_paths_06.py` | Dijkstra, BFS shortest paths with visualisation |
| `sbm_utils_07.py` | SBM graph generation helpers on top of NetworkX |
| `walk_utils_08.py` | Uniform random walks (DeepWalk) and biased walks (Node2Vec p/q) |
| `embeddings_utils_09.py` | Word2Vec training on walks; embedding inspection utilities |
| `evaluation_utils_10.py` | Spectral clustering, ARI/NMI scoring, PCA/t-SNE projection |

---

## Notes

Eight self-prepared Markdown notes covering the full theory behind every notebook:

| File | Topics |
|------|--------|
| `01_graph_fundementals_notes.md` | BFS, DFS, graph types, matrix representations |
| `02_centrality_notes.md` | All four centrality measures with intuition |
| `03_community_detection_karate_notes.md` | Karate Club analysis, modularity, algorithm comparison |
| `04_community_detection_notes.md` | Girvan-Newman, Louvain phases, K-Clique theory |
| `05_laplacian_matrices_notes.md` | Unnormalised & normalised Laplacian, properties |
| `06_spectral_clustering_GCN_notes.md` | Eigengap heuristic, spectral partitioning, GCN intuition |
| `07_node_embeddings_node2vec.md` | DeepWalk, Skip-Gram, negative sampling, Node2Vec biased walks |
| `08_GNN_Notes.md` | Message passing framework, permutation invariance/equivariance, GCN, GAT, over-smoothing, JK-Net |

---

## Presentations & Results

- **`presentations/01_graph_shallow_embeddings.pptx`** — 29-slide academic presentation covering all six matrix factorisation methods from Hamilton's *Graph Representation Learning* (Table 3.1), DeepWalk, Node2Vec, the Qiu et al. 2018 unification proof, and a GNN preview. Built with a navy/teal colour palette.
- **`results/01_SBM_Spectral_Clustering_Report.pdf`** — Full accessible report on the SBM spectral clustering experiments with embedded figures.

---

## Installation

```bash
git clone https://github.com/tmfsiddu/community_detection.git
cd communityDetection
pip install -r requirements.txt
```

**Requirements:**

```
networkx, numpy, pandas, matplotlib, scipy, scikit-learn,
gensim, python-louvain, torch, torch-geometric, jupyter, ipykernel
```

> PyTorch Geometric installation requires matching your CUDA version. See [pytorch-geometric.com/install](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## Running the Notebooks

Open notebooks in order — each one builds on the concepts of the previous:

```bash
jupyter notebook notebooks/01_networkx_basics.ipynb
```

For notebook `06`, the `src/` path is injected automatically in Cell 0 — no manual configuration needed.

---

## References

**Books**
- Hamilton, W. L. — *Graph Representation Learning* (2020)
- Barabási, A.-L. — *Network Science* (2016)

**Papers**
- Perozzi et al. (2014) — *DeepWalk: Online Learning of Social Representations*
- Grover & Leskovec (2016) — *Node2Vec: Scalable Feature Learning for Networks*
- Kipf & Welling (2017) — *Semi-Supervised Classification with Graph Convolutional Networks*
- Veličković et al. (2018) — *Graph Attention Networks*
- Hu et al. (2020) — *Open Graph Benchmark*
- Qiu et al. (2018) — *Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec*

**Libraries**
- [NetworkX](https://networkx.org/)
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Gensim](https://radimrehurek.com/gensim/)

---

## Author

**Sangavarapu Venkata Sai Siddardha**  
B.Tech Computer Science and Engineering — NIT Puducherry  
Research interests: Graph Machine Learning · Community Detection · Network Science · Deep Learning