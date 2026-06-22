"""
models.py
---------
Four models in progressive order:

1. ShallowGCN   – standard 2-layer GCN (proper baseline)
                  → establishes what a well-designed GCN can achieve
2. DeepGCN      – intentionally deep (6 layers), no skip connections
                  → demonstrates OVER-SMOOTHING as depth increases
3. StandardGAT  – 2-layer multi-head attention
                  → demonstrates OVERFITTING on dense graphs
4. ResAttJKNet  – hybrid (GCN + GAT + Residual + JK-Net)
                  → fixes both failure modes observed above
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


# ══════════════════════════════════════════════════════
#  MODEL 1 – Shallow GCN  (standard 2-layer baseline)
# ══════════════════════════════════════════════════════
class ShallowGCN(nn.Module):
    """
    A standard 2-layer GCN — the correct, minimal baseline.

    Architecture:
        Layer 1:  GCNConv(in_channels  → hidden_channels) + ReLU + Dropout
        Layer 2:  GCNConv(hidden_channels → out_channels)   [no activation]

    Why 2 layers?
        Each GCNConv layer aggregates 1-hop neighbourhood information.
        2 layers means the model sees up to 2-hop neighbours — sufficient
        for most node classification tasks. Adding more layers causes
        over-smoothing (demonstrated in Model 2 below).

    This model gives us the performance ceiling of a standard GCN
    before over-smoothing degrades it.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GCNConv(in_channels, hidden_channels)    # input → hidden
        self.conv2 = GCNConv(hidden_channels, out_channels)   # hidden → classes

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2 (no activation — raw logits for cross-entropy)
        x = self.conv2(x, edge_index)
        return x

    def get_embedding(self, x, edge_index):
        """Returns penultimate-layer embedding (after layer 1, before classifier)."""
        with torch.no_grad():
            x = F.relu(self.conv1(x, edge_index))
        return x


# ══════════════════════════════════════════════════════
#  MODEL 2 – Deep GCN  (over-smoothing demonstration)
# ══════════════════════════════════════════════════════
class DeepGCN(nn.Module):
    """
    A configurable N-layer GCN with NO residual connections.

    Used in two ways:
        • num_layers=2  → same architecture as ShallowGCN (sanity check)
        • num_layers=4,6,8,10 → progressively deeper to show over-smoothing

    Why it fails at depth:
        Each GCNConv averages a node's features with its neighbours.
        After K rounds of averaging, every node looks like the average
        of its K-hop neighbourhood. When K is large, all embeddings
        converge to the same vector — the classifier can no longer
        distinguish nodes. This is OVER-SMOOTHING.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6):
        super().__init__()
        self.num_layers = num_layers
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
            if i < len(self.convs) - 1:        # ReLU + Dropout on all but last
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x                               # raw logits

    def get_embedding(self, x, edge_index):
        """Returns embedding after the penultimate layer."""
        with torch.no_grad():
            for conv in self.convs[:-1]:
                x = F.relu(conv(x, edge_index))
        return x


# ══════════════════════════════════════════════════════
#  MODEL 3 – Standard GAT  (overfitting demonstration)
# ══════════════════════════════════════════════════════
class StandardGAT(nn.Module):
    """
    2-layer Graph Attention Network with multi-head attention.

    Architecture:
        Layer 1:  GATConv(in → hidden, heads=8)   + ELU + Dropout
        Layer 2:  GATConv(hidden*8 → out, heads=1) [collapses multi-head]

    Why attention is useful in principle:
        Unlike GCN which gives equal weight to all neighbours,
        GAT learns a separate attention score for every edge.
        This lets the model focus more on informative neighbours
        and ignore noisy ones.

    Why it struggles on the Cora graph:
        Cora is relatively sparse, so attention overfits to specific
        neighbourhood patterns in training, showing lower test accuracy
        than expected.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=8, dropout=0.6):
        super().__init__()
        self.dropout = dropout

        # layer 1: 8 attention heads → hidden_channels each
        self.conv1 = GATConv(in_channels, hidden_channels,
                             heads=heads, dropout=dropout)

        # layer 2: collapse all heads → out_channels
        self.conv2 = GATConv(hidden_channels * heads, out_channels,
                             heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x                               # raw logits

    def get_embedding(self, x, edge_index):
        """Returns embedding after the first GAT layer."""
        with torch.no_grad():
            x = F.dropout(x, p=self.dropout, training=False)
            x = F.elu(self.conv1(x, edge_index))
        return x


# ══════════════════════════════════════════════════════
#  MODEL 4 – ResAttJKNet  (proposed hybrid)
# ══════════════════════════════════════════════════════
class ResAttJKNet(nn.Module):
    """
    Residual Attention Jumping Knowledge Network.

    Motivation:
        Model 2 (DeepGCN) fails due to over-smoothing when depth increases.
        Model 3 (StandardGAT) can struggle with generalisation.
        This model addresses both by fusing three published techniques:

    Three components (all from published papers):

    ① Dual-Perspective Blocks  [our combination]
        Each hidden layer runs TWO branches in parallel:
          • GCNConv  → captures stable structural neighbourhood info
                       (Kipf & Welling, 2017)
          • GATConv  → captures dynamic attention-weighted info
                       (Veličković et al., 2018)
        Their outputs are averaged: fused = (gcn_out + gat_out) / 2

    ② Residual Connections  (He et al., 2016 / Chen et al., 2020)
        output = fused + layer_input
        Prevents over-smoothing — the original signal is always preserved.

    ③ Jumping Knowledge  (Xu et al., 2018)
        Save the output of EVERY hidden layer.
        At the end, concatenate all of them:
            master_emb = [layer1_out | layer2_out | ... | layerN_out]
        The classifier sees ALL scales simultaneously.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, gat_heads=4, dropout=0.5):
        super().__init__()
        self.dropout    = dropout
        self.num_layers = num_layers

        # ── input projection (maps raw features to hidden dimension) ────
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # ── one GCNConv + one GATConv per hidden layer ──────────────────
        self.gcn_convs = nn.ModuleList()
        self.gat_convs = nn.ModuleList()

        for _ in range(num_layers):
            self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
            self.gat_convs.append(
                GATConv(hidden_channels,
                        hidden_channels // gat_heads,   # per-head dim
                        heads=gat_heads,
                        dropout=dropout)
            )

        # ── final classifier: input = all layers concatenated ───────────
        jk_dim = num_layers * hidden_channels
        self.classifier = nn.Linear(jk_dim, out_channels)

    # ── helper: one hidden layer block ──────────────────────────────────
    def _layer_block(self, x, edge_index, layer_idx):
        """
        One dual-perspective block with residual connection.
        Both branches take the same input x and their outputs are averaged.
        """
        gcn_out = self.gcn_convs[layer_idx](x, edge_index)   # ① structural
        gat_out = self.gat_convs[layer_idx](x, edge_index)   # ① semantic

        fused   = (gcn_out + gat_out) / 2.0                  # ① Dual-Perspective
        fused   = F.relu(fused)
        fused   = F.dropout(fused, p=self.dropout, training=self.training)

        return fused + x                                       # ② Residual

    # ── forward pass ────────────────────────────────────────────────────
    def forward(self, x, edge_index, return_embedding=False):
        """
        Args:
            x               : node feature matrix  [N, in_channels]
            edge_index      : graph connectivity    [2, E]
            return_embedding: if True, return JK master embedding
                              instead of class logits.
                              Used for clustering and recommendation.
        """
        x = F.relu(self.input_proj(x))       # project to hidden dim

        layer_outputs = []                    # ③ JK-Net: collect every layer

        for i in range(self.num_layers):
            x = self._layer_block(x, edge_index, i)
            layer_outputs.append(x)

        # concatenate all layer outputs → master embedding [N, layers*hidden]
        master_emb = torch.cat(layer_outputs, dim=1)

        if return_embedding:
            return master_emb                 # for clustering / recommendation

        return self.classifier(master_emb)   # class logits for classification