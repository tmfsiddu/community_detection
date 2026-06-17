# embedding_utils.py
# ---------------------------------------------------------
# Functions to train Word2Vec on walks and inspect embeddings.
# Used by both DeepWalk and Node2Vec (same training, different walks).
# ---------------------------------------------------------

from gensim.models import Word2Vec


def train_word2vec(walks, vector_size=8, window=3, epochs=200):
    """
    Train a Word2Vec (Skip-Gram) model on the provided walks.

    Treat each walk as a sentence and each node as a word.
    Word2Vec then learns embeddings so that nodes
    appearing near each other in walks get similar vectors.

    Parameters
    ----------
    walks       : list of walks (each walk = list of node names)
    vector_size : size of the embedding vector (dimensions)
    window      : how many nodes on each side count as context
    epochs      : training iterations

    Returns
    -------
    model : trained gensim Word2Vec model
    """
    model = Word2Vec(
        sentences   = walks,
        vector_size = vector_size,
        window      = window,
        min_count   = 1,
        sg          = 1,       # sg=1 → Skip-Gram (used by DeepWalk/Node2Vec)
        epochs      = epochs,
        workers     = 1,       # reproducibility
    )
    return model


def print_embeddings(model, nodes):
    """Print the embedding vector for each node."""
    print("Node Embeddings:")
    for node in nodes:
        vec = model.wv[node].round(3)
        print(f"  {node}: {vec}")


def print_similarities(model, pairs):
    """
    Print cosine similarity between pairs of nodes.

    Parameters
    ----------
    model : trained Word2Vec model
    pairs : list of (node_a, node_b, label) tuples
            label is just a human-readable description
    """
    print("Cosine Similarities (1.0 = identical, -1.0 = opposite):")
    for a, b, label in pairs:
        sim = model.wv.similarity(a, b)
        print(f"  {a} ↔ {b}  [{label}]: {sim:.3f}")