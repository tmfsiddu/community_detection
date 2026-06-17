"""
models.py
---------
Three models:

1. DeepGCN      – intentionally deep (6 layers), no skip connections
                  → demonstrates OVER-SMOOTHING
2. StandardGAT  – 2-layer multi-head attention
                  → demonstrates OVERFITTING on dense graphs
3. ResAttJKNet  – our novel hybrid (GCN + GAT + Residual + JK-Net)
                  → fixes both failure modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


# ══════════════════════════════════════════════════════
#  BASELINE 1 – Deep Vanilla GCN  (over-smoothing demo)
# ══════════════════════════════════════════════════════
class DeepGCN(nn.Module):
    """
    6-layer GCN with NO residual connections.

    Why it fails:
        Each GCNConv averages a node's features with its neighbours.
        After 6 rounds of averaging, every node looks like the local
        average of its 6-hop neighbourhood → all embeddings converge
        to the same vector → the classifier can no longer tell
        products apart.  This is called OVER-SMOOTHING.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6):
        super().__init__()
        self.convs = nn.ModuleList()

        # first layer: input → hidden
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # middle layers: hidden → hidden
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # last layer: hidden → classes
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:   # ReLU on all but last
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x                           # raw logits


# ══════════════════════════════════════════════════════
#  BASELINE 2 – Standard GAT  (overfitting demo)
# ══════════════════════════════════════════════════════
class StandardGAT(nn.Module):
    """
    2-layer Graph Attention Network.

    Why it fails:
        GAT learns an attention weight for every edge.
        Amazon Computers has ~490 k edges with only 13 k nodes
        (very dense).  The model memorises specific attention
        patterns on the training paths but cannot generalise
        to unseen test nodes → overfitting / structural instability.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=8, dropout=0.6):
        super().__init__()
        self.dropout = dropout

        # layer 1: 8 attention heads → hidden_channels each
        self.conv1 = GATConv(in_channels, hidden_channels,
                             heads=heads, dropout=dropout)

        # layer 2: collapse heads → out_channels
        self.conv2 = GATConv(hidden_channels * heads, out_channels,
                             heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x                           # raw logits


# ══════════════════════════════════════════════════════
#  NOVEL MODEL – ResAttJKNet
# ══════════════════════════════════════════════════════
class ResAttJKNet(nn.Module):
    """
    Residual Attention Jumping Knowledge Network.

    Three innovations fused together:

    ① Dual-Perspective Blocks
        Each hidden layer runs TWO branches in parallel:
          • GCNConv  → captures stable structural neighbourhood info
          • GATConv  → captures dynamic semantic attention
        Their outputs are averaged: fused = (gcn_out + gat_out) / 2

    ② Residual Connections
        fused_output = fused + layer_input  (like ResNet)
        This prevents vanishing gradients and over-smoothing.

    ③ Jumping Knowledge (JK-Net)
        We SAVE the fused output of every hidden layer.
        At the end we CONCATENATE all of them:
            master_emb = [layer1_out | layer2_out | ... | layerN_out]
        The classifier sees ALL scales (local + global structure)
        rather than just the final layer's view.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, gat_heads=4, dropout=0.5):
        super().__init__()
        self.dropout    = dropout
        self.num_layers = num_layers

        # ── input projection (shared starting point for both branches) ──
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # ── one GCNConv + one GATConv per hidden layer ──────────────────
        self.gcn_convs = nn.ModuleList()
        self.gat_convs = nn.ModuleList()

        for _ in range(num_layers):
            self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
            # GATConv outputs hidden_channels (heads * (hidden//heads))
            self.gat_convs.append(
                GATConv(hidden_channels,
                        hidden_channels // gat_heads,   # per-head size
                        heads=gat_heads,
                        dropout=dropout)
            )

        # ── final classifier: input = num_layers * hidden_channels ──────
        jk_dim = num_layers * hidden_channels
        self.classifier = nn.Linear(jk_dim, out_channels)

    # ── helper: one hidden layer block ──────────────────────────────────
    def _layer_block(self, x, edge_index, layer_idx):
        """
        Runs one dual-perspective block with residual connection.
        Returns fused output (same shape as input).
        """
        gcn_out = self.gcn_convs[layer_idx](x, edge_index)          # structural
        gat_out = self.gat_convs[layer_idx](x, edge_index)          # semantic

        fused   = (gcn_out + gat_out) / 2.0                        # ① Dual-Perspective
        fused   = F.relu(fused)
        fused   = F.dropout(fused, p=self.dropout, training=self.training)

        return fused + x                                             # ② Residual

    # ── forward pass ────────────────────────────────────────────────────
    def forward(self, x, edge_index, return_embedding=False):
        """
        Args:
            x              : node feature matrix [N, in_channels]
            edge_index     : graph connectivity [2, E]
            return_embedding: if True, return the JK embedding instead
                              of class logits (used for clustering & recommendation)
        """
        # project input to hidden dimension
        x = F.relu(self.input_proj(x))

        layer_outputs = []                  # ③ JK-Net: collect all layers

        for i in range(self.num_layers):
            x = self._layer_block(x, edge_index, i)
            layer_outputs.append(x)

        # concatenate all layer outputs side-by-side → master embedding
        master_emb = torch.cat(layer_outputs, dim=1)   # [N, num_layers * hidden]

        if return_embedding:
            return master_emb              # raw embeddings for downstream tasks

        return self.classifier(master_emb) # logits for classification