"""
recommend.py
------------
Product recommendation using cosine similarity on ResAttJKNet embeddings.

The idea:
  Two products are "similar" if their multi-scale graph embeddings
  (capturing both structural neighbourhood and semantic attention)
  point in the same direction in embedding space.
  Cosine similarity measures the angle between two vectors —
  it equals 1 for identical direction and 0 for orthogonal.
"""

import torch
import torch.nn.functional as F


def get_all_embeddings(model, data):
    """
    Extracts the JK embedding for every product node.

    Returns:
        emb : Tensor of shape [N, num_layers * hidden_channels]
    """
    model.eval()
    with torch.no_grad():
        emb = model(data.x, data.edge_index, return_embedding=True)
    return emb   # [N, D]


def recommend_products(product_id, model, data, num_recommendations=5):
    """
    Given a product node ID, returns the top-K most similar products.

    Cosine Similarity between product i and product j:
        sim(i, j) = (emb_i · emb_j) / (||emb_i|| * ||emb_j||)

    Higher similarity → products co-purchased more often or
    structurally connected in the graph.

    Args:
        product_id         : integer node index of the target product
        model              : trained ResAttJKNet
        data               : PyG Data object
        num_recommendations: K (default = 5)

    Returns:
        List of (node_id, similarity_score) tuples, highest first
    """
    emb = get_all_embeddings(model, data)          # [N, D]

    # L2-normalise every embedding so dot product = cosine similarity
    emb_norm  = F.normalize(emb, p=2, dim=1)      # [N, D]

    target    = emb_norm[product_id].unsqueeze(0)  # [1, D]

    # cosine similarity of target vs all nodes
    sims      = (emb_norm @ target.T).squeeze(1)   # [N]

    # exclude the product itself
    sims[product_id] = -1.0

    # top-K indices sorted by similarity (highest first)
    top_k     = sims.topk(num_recommendations)

    results = [(idx.item(), score.item())
               for idx, score in zip(top_k.indices, top_k.values)]
    return results