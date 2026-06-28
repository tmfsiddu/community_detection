import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class AggregationGAT(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_dim=32,heads=8):
        super(AggregationGAT,self).__init__()
        #layer1 taking raw input features then using 8 heads which produce 32 dim each concatenating all of them and sending it to second layer
        #32*8=256 dim vector will be send to next layer
        self.layer1=GATConv(num_features,hidden_dim,heads=heads,dropout=0.6)
        #in layer2 we not concatenate all heads output instead we will try to take average and try to get probability scores 
        self.layer2=GATConv(hidden_dim*heads,num_classes,heads=1,concat=False,dropout=0.6)
    def forward(self,x,edge_index):
        x=F.dropout(x,p=0.6,training=self.training)
        x=self.layer1(x,edge_index)
        x=F.elu(x)
        x=F.dropout(x,p=0.6,training=self.training)
        x=self.layer2(x,edge_index)
        return F.log_softmax(x,dim=1) 
#now as we know we got an overfit while training the model we try to decrease it by increasing the drop out
class AggregationGATO(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_dim=32,heads=8):
        super(AggregationGATO,self).__init__()
        #layer1 taking raw input features then using 8 heads which produce 32 dim each concatenating all of them and sending it to second layer
        #32*8=256 dim vector will be send to next layer
        self.layer1=GATConv(num_features,hidden_dim,heads=heads,dropout=0.6)
        #in layer2 we not concatenate all heads output instead we will try to take average and try to get probability scores 
        self.layer2=GATConv(hidden_dim*heads,num_classes,heads=1,concat=False,dropout=0.6)
    def forward(self,x,edge_index):
        x=F.dropout(x,p=0.75,training=self.training)
        x=self.layer1(x,edge_index)
        x=F.elu(x)
        x=F.dropout(x,p=0.75,training=self.training)
        x=self.layer2(x,edge_index)
        return F.log_softmax(x,dim=1) 

#using this class we will try to do community detection using normalized spectral clustering not on just pure topology using the embeddings given by GAT  
def cluster_embeddings_with_spectral(embeddings_np, true_labels_np, num_clusters=7, n_neighbors=10):
    #builds a KNN graph from the embedding vectors and runs normalized spectral clustering on it
    spectral_model = SpectralClustering(
        n_clusters=num_clusters,
        affinity='nearest_neighbors',  #builds similarity graph from nearest neighbors in embedding space
        n_neighbors=n_neighbors,
        random_state=42,
        assign_labels='kmeans'         #after getting eigenvectors use kmeans to assign final cluster labels
    )
    cluster_assignments = spectral_model.fit_predict(embeddings_np)
    ari = adjusted_rand_score(true_labels_np, cluster_assignments)
    nmi = normalized_mutual_info_score(true_labels_np, cluster_assignments)
    return ari, nmi, cluster_assignments

# Append this to the bottom of src/evaluation_utils_10.py


def plot_tsne_comparison(embeddings_np, kmeans_labels, spectral_labels, true_labels, title_suffix=""):
    # Reduce 256D down to 2D coordinates
    # perplexity=30 is standard for datasets around this size to balance local/global focus
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    scatter_kwargs = {'s': 15, 'alpha': 0.8, 'cmap': 'tab10'}
    
    # 1. Left Plot: Colored by K-Means
    sc1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=kmeans_labels, **scatter_kwargs)
    axes[0].set_title("GAT Embeddings Partitioned by K-Means\n(Spherical Bubble Philosophy)", fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.5)
    
    # 2. Right Plot: Colored by Spectral Clustering
    sc2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=spectral_labels, **scatter_kwargs)
    axes[1].set_title("GAT Embeddings Partitioned by Spectral Clustering\n(Manifold / Graph Geometry Philosophy)", fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.5)
    
    # Add a global overarching title
    plt.suptitle(f"t-SNE Latent Space Analysis: {title_suffix}", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    return embeddings_2d
 
