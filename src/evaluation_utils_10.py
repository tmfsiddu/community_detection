"""
evaluation_utils_10.py
----------------------
Functions for evaluating and comparing community detection algorithms.
Covers:
  - Spectral Clustering on SBM graphs
  - ARI / NMI scoring (how accurate are the detected communities?)
  - 2D visualisation of embeddings via t-SNE or PCA

Used in notebook 04 to compare:
  Spectral Clustering vs DeepWalk vs Node2Vec
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster        import SpectralClustering, KMeans
from sklearn.metrics        import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition  import PCA
from sklearn.manifold       import TSNE
from scipy.linalg           import eigh


# ── Spectral Clustering ────────────────────────────────────────────────────────

def run_spectral_clustering(G, n_clusters, seed=42):
    """
    Run Spectral Clustering on a NetworkX graph.

    Steps (so you can explain each one):
      1. Build adjacency matrix A
      2. Build degree matrix D
      3. Compute unnormalised Laplacian  L = D - A
      4. Find the k smallest eigenvectors of L
      5. Run K-Means on those eigenvectors

    Parameters
    ----------
    G          : networkx Graph
    n_clusters : how many communities to find
    seed       : random seed

    Returns
    -------
    labels : list of predicted community labels (one per node)
    """
    # Step 1 & 2: adjacency and degree
    A = np.array(nx_adjacency(G))

    # Step 3: Laplacian
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Step 4: smallest eigenvectors (scipy eigh returns them sorted)
    _, eigenvectors = eigh(L)
    X = eigenvectors[:, :n_clusters]   # keep first k columns

    # Step 5: K-Means on the eigenvector rows
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X)

    return list(labels)


def nx_adjacency(G):
    """Return adjacency matrix as a numpy array (helper)."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    A = np.zeros((n, n))
    for u, v in G.edges():
        A[idx[u]][idx[v]] = 1
        A[idx[v]][idx[u]] = 1
    return A


# ── Embedding → Cluster Labels ─────────────────────────────────────────────────

def embeddings_to_labels(model, G, n_clusters, seed=42):
    """
    Extract embedding vectors from a trained Word2Vec model,
    then run K-Means to get community labels.

    Parameters
    ----------
    model      : trained gensim Word2Vec model
    G          : networkx Graph (to get node list)
    n_clusters : how many clusters
    seed       : random seed

    Returns
    -------
    nodes  : list of node ids (in sorted order)
    labels : list of predicted cluster labels (one per node)
    """
    nodes = sorted(G.nodes())
    # Convert node ids to strings (Word2Vec stores string keys)
    vecs = np.array([model.wv[str(n)] for n in nodes])

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(vecs)

    return nodes, list(labels)


# ── Accuracy Metrics ───────────────────────────────────────────────────────────

def evaluate(true_labels_dict, predicted_labels, nodes):
    """
    Score predicted community labels against ground truth.

    Parameters
    ----------
    true_labels_dict : dict  {node_id: true_block_index}
                       e.g. from sbm_utils.create_sbm_graph
    predicted_labels : list of predicted labels (same order as nodes)
    nodes            : list of node ids (same order as predicted_labels)

    Returns
    -------
    ari : Adjusted Rand Index  (1.0 = perfect, 0.0 = random)
    nmi : Normalised Mutual Info (1.0 = perfect, 0.0 = random)
    """
    true = [true_labels_dict[n] for n in nodes]
    ari  = adjusted_rand_score(true, predicted_labels)
    nmi  = normalized_mutual_info_score(true, predicted_labels)
    return round(ari, 3), round(nmi, 3)


def print_scores(method_name, ari, nmi):
    """Print ARI and NMI scores for a method."""
    print(f"  {method_name:20s}  ARI={ari:.3f}  NMI={nmi:.3f}")


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_embeddings_2d(model, G, block_labels, title, method="pca"):
    """
    Reduce embeddings to 2D and scatter-plot, coloured by true block.

    Parameters
    ----------
    model        : trained gensim Word2Vec model
    G            : networkx Graph
    block_labels : dict {node_id: true_block_index}
    title        : plot title string
    method       : "pca" (fast) or "tsne" (better but slower)
    """
    nodes  = sorted(G.nodes())
    vecs   = np.array([model.wv[str(n)] for n in nodes])
    colors = [block_labels[n] for n in nodes]

    # Reduce to 2D
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=min(15, len(nodes)-1),
                       random_state=42)
    else:
        reducer = PCA(n_components=2, random_state=42)

    vecs_2d = reducer.fit_transform(vecs)

    # Plot
    cmap = plt.cm.get_cmap("Set1", max(colors) + 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, node in enumerate(nodes):
        ax.scatter(vecs_2d[i, 0], vecs_2d[i, 1],
                   color=cmap(colors[i]), s=120, zorder=3)
        ax.annotate(f"N{node}(B{colors[i]})",
                    (vecs_2d[i, 0], vecs_2d[i, 1]),
                    fontsize=7, xytext=(4, 3),
                    textcoords="offset points")

    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} dim 1")
    ax.set_ylabel(f"{method.upper()} dim 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()