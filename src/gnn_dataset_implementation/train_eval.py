"""
train_eval.py
-------------
Clean, reusable training loop + two evaluation modes:
  1. Node Classification  → test accuracy
  2. Community Detection  → K-Means + Adjusted Rand Index (ARI)
"""

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# ══════════════════════════════════════
#  Training loop
# ══════════════════════════════════════
def train_model(model, data, epochs=150, lr=0.005, label="Model"):
    """
    Trains any of the three models with cross-entropy loss.

    Args:
        model  : nn.Module (DeepGCN, StandardGAT, or ResAttJKNet)
        data   : PyG Data object
        epochs : number of epochs
        lr     : learning rate
        label  : string name for console printouts

    Returns:
        model  : trained model (in eval mode)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    print(f"Training {label} for {epochs} epochs ...")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Handle the signature difference safely during training
        if _supports_embedding(model):
            out = model(data.x, data.edge_index, return_embedding=False)
        else:
            out = model(data.x, data.edge_index)
            
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 30 == 0:
            val_acc = _accuracy(model, data, data.val_mask)
            print(f"  Epoch {epoch:>3} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

    model.eval()
    print(f"{label} training complete ✓\n")
    return model


# ══════════════════════════════════════
#  Evaluation 1 – Node Classification
# ══════════════════════════════════════
def evaluate_classification(model, data):
    """Returns test accuracy (fraction of correctly labelled test nodes)."""
    model.eval()
    with torch.no_grad():
        if _supports_embedding(model):
            out = model(data.x, data.edge_index, return_embedding=False)
        else:
            out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc  = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return acc


# ══════════════════════════════════════
#  Evaluation 2 – Community Detection
# ══════════════════════════════════════
def evaluate_clustering(model, data, num_classes, model_name="Model"):
    """
    Extracts hidden embeddings, runs K-Means, and computes ARI.

    ARI (Adjusted Rand Index):
        Compares two clusterings ignoring permutations of cluster labels.
        ARI = 1.0  → perfect match with ground truth
        ARI = 0.0  → random clustering
        ARI < 0.0  → worse than random
    """
    model.eval()
    with torch.no_grad():
        # --- get embeddings (before the classifier) ---
        if _supports_embedding(model):
            emb = model(data.x, data.edge_index, return_embedding=True)
        else:
            # For baseline models: use the penultimate layer output
            # We re-run the forward pass but intercept the hidden state
            emb = _get_penultimate_embedding(model, data)

    emb_np  = emb.cpu().numpy()
    labels  = data.y.cpu().numpy()

    # K-Means clustering in the embedding space
    kmeans  = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    pred_clusters = kmeans.fit_predict(emb_np)

    ari = adjusted_rand_score(labels, pred_clusters)
    return ari


# ── Helpers ─────────────────────────────────────────────────────────────

def _accuracy(model, data, mask):
    """Calculates model accuracy over a specific validation/test mask."""
    model.eval()
    with torch.no_grad():
        if _supports_embedding(model):
            out = model(data.x, data.edge_index, return_embedding=False)
        else:
            out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc  = (pred[mask] == data.y[mask]).float().mean().item()
    return acc


def _supports_embedding(model):
    """Check if model.forward accepts return_embedding kwarg (ResAttJKNet)."""
    import inspect
    sig = inspect.signature(model.forward)
    return 'return_embedding' in sig.parameters


def _get_penultimate_embedding(model, data):
    """
    For DeepGCN / StandardGAT: run all layers except the final one
    to get the last hidden representation (used as the 'embedding').
    """
    # FIXED: Added the dot prefix to make it a local relative package import
    from .models import DeepGCN, StandardGAT
    
    x, edge_index = data.x, data.edge_index

    if isinstance(model, DeepGCN):
        for conv in model.convs[:-1]:   # skip last conv
            x = F.relu(conv(x, edge_index))
        return x

    elif isinstance(model, StandardGAT):
        x = F.dropout(x, p=model.dropout, training=False)
        x = F.elu(model.conv1(x, edge_index))
        return x   # after first layer = "hidden embedding"

    else:
        raise ValueError("Unknown baseline model type encountered inside helper configuration.")