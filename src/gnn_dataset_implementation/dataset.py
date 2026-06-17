"""
dataset.py
----------
Loads the Amazon Computers dataset and creates train/val/test masks.
"""

import torch
from torch_geometric.datasets import Amazon


def load_data(root="data/Amazon", seed=42):
    """
    Downloads and returns the Amazon Computers graph.

    Returns:
        data  : PyG Data object (graph with node features & labels)
        num_classes : number of product categories
    """
    dataset = Amazon(root=root, name="Computers")
    data = dataset[0]                    # single graph
    num_classes = dataset.num_classes

    # ── Reproducible 60 / 20 / 20 split ──────────────────────────────────
    torch.manual_seed(seed)
    n = data.num_nodes
    perm = torch.randperm(n)             # random ordering of node ids

    train_end = int(0.6 * n)
    val_end   = int(0.8 * n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)

    train_mask[perm[:train_end]]        = True
    val_mask  [perm[train_end:val_end]] = True
    test_mask [perm[val_end:]]          = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    print(f"[Dataset] Nodes: {n}  |  Edges: {data.edge_index.size(1)}")
    print(f"[Dataset] Features: {data.num_node_features}  |  Classes: {num_classes}")
    print(f"[Dataset] Train: {train_mask.sum()}  Val: {val_mask.sum()}  Test: {test_mask.sum()}")

    return data, num_classes