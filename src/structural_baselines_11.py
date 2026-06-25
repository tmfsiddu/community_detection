#this file contains all the required functions in order to implement random walks and spectral clustering
import random
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
import torch.nn.functional as F

#first one is random walk where it will generate bised random wlak for node2vec and if we p and q parameters as 1 then it becoms unbiased one
def generate_biased_walks(edge_index,num_nodes,walk_length,walks_per_node,p=1.0,q=1.0):
    #computer geenrally stores all these node pairs which all are connected so to reduce complexity we are converting
    #that to adjacency list for faster neighbor lookup
    adj_list = {i: set() for i in range(num_nodes)}
    edge_index_np = edge_index.cpu().numpy()
    for src, dst in zip(edge_index_np[0], edge_index_np[1]):
        adj_list[src].add(dst)
        adj_list[dst].add(src)
    #converting sets to list to optimize inner loop speeds
    adj_list = {k: list(v) for k, v in adj_list.items()}
    walks = []
    #starting simulating a random walk
    for _ in range(walks_per_node):
        nodes = list(range(num_nodes))
        random.shuffle(nodes) 
        #shuffling the neighbors so that each time we start we get different nodes 
        for start_node in nodes:
            walk = [start_node]
            
            while len(walk) < walk_length:
                curr = walk[-1]
                neighbors = adj_list[curr]
                
                if len(neighbors) == 0:
                    break # Safe exit for dead ends/isolated nodes
                    
                if len(walk) == 1:
                    #first node is choosen completely random and we need previous node also to compute probabilities 
                    walk.append(random.choice(neighbors))
                else:
                    prev = walk[-2]
                    probabilities = []
                    #calculating probabilities and decide which node to explore next 
                    for neighbor in neighbors:
                        if neighbor == prev:
                            probabilities.append(1.0 / p)  # Return to previous
                        elif neighbor in adj_list[prev]:
                            probabilities.append(1.0)       # Stay local (BFS bias)
                        else:
                            probabilities.append(1.0 / q)  # Venture outward (DFS bias)
                    
                    # Normalize to valid probability distribution
                    prob_sum = sum(probabilities)
                    normalized_probs = [prob / prob_sum for prob in probabilities]
                    
                    # Sample step stochastically
                    #even after calculating probability there will be some randomness involved 
                    next_node = np.random.choice(neighbors, p=normalized_probs)
                    walk.append(int(next_node))
                    
            walks.append(walk)
    return walks

#based on the window size we are going to ectract the pairs and they will act as dataset 
def extract_skipgram_pairs(walks, window_size=2):
    targets, contexts = [], []
    for walk in walks:
        for i, target in enumerate(walk):
            start = max(0, i - window_size)
            end = min(len(walk), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    targets.append(target)
                    contexts.append(walk[j])
    return torch.tensor(targets, dtype=torch.long), torch.tensor(contexts, dtype=torch.long)
#now we nend to write a function which randomly intialize the embeddings of node at starting and then compute the dot product based on pairs which we got through previous function
class ShallowEmbeddingModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim=32):
        super(ShallowEmbeddingModel, self).__init__()
        self.w_target = nn.Embedding(num_nodes, embedding_dim)
        self.w_context = nn.Embedding(num_nodes, embedding_dim)
        
        # Initialize vectors uniformly across space bounds
        self.w_target.weight.data.uniform_(-1.0, 1.0)
        self.w_context.weight.data.uniform_(-1.0, 1.0)

    def forward(self, target_nodes, context_nodes):
        v_t = self.w_target(target_nodes)
        v_c = self.w_context(context_nodes)
        return torch.sum(v_t * v_c, dim=1)
#if the dot product is high then we can say that those two node pairs are very similar and no means they are not
#going for spetral clustering based on the algorithm
#like build ajacency matrix,degree matrix then compute laplacian matrix and compute the eigen vectors for it 
#remove the first one as it is trivial go for the second smallest one
def compute_spectral_embeddings(edge_index, num_nodes, num_components=7):
    edge_index_np = edge_index.cpu().numpy()
    v = np.ones(edge_index_np.shape[1])
    
    #creating a sparse matrix so that remove storage wastage for storing all 0 where in most nodes does not connect to all other nodes 
    A = sp.coo_matrix((v, (edge_index_np[0], edge_index_np[1])), shape=(num_nodes, num_nodes))
    A = A + A.T#to keep the matrix symmetric and suitable for calculation 
    A.data = np.ones_like(A.data) # Wipe duplicate linkages
    
    # builing sparse  Degree Matrix (D)
    degrees = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    
    # formula for laplacian matrix 
    L = D - A
    
    # 4. Extract Smallest k Eigenvectors
    # Add small shift to avoid exact singularity — Laplacian always has a zero eigenvalue
    # which causes "Factor is exactly singular" when sigma=0.0 is used with eigsh
    # Instead we shift L by a tiny epsilon so the matrix becomes invertible
    k = num_components + 1  # +1 because smallest eigenvector is always trivial zero vector
    L_shifted = L.astype(float) + 1e-8 * sp.eye(num_nodes)
    eigenvalues, eigenvectors = eigsh(L_shifted, k=k, which='SM')
    #we are computing the L-1 because for computng the smallest eigen values is much complex so this eighs will try
    #to compute the large eigen values in L^-1 which is simpler onw that y we are computing L-1
    # Drop first trivial vector (slice column 1 onward). Returns pristine structural embedding coordinates.
    return eigenvectors[:, 1:]


#this is the normalized spectral clustering function where the degree matrix^-1/2 gets multiplied to adjacency matrix 
def compute_spectral_embeddings_normalized(edge_index, num_nodes, num_components=7):
    edge_index_np = edge_index.cpu().numpy()
    v = np.ones(edge_index_np.shape[1])
    
    A = sp.coo_matrix((v, (edge_index_np[0], edge_index_np[1])), shape=(num_nodes, num_nodes))
    A = A + A.T
    A.data = np.ones_like(A.data) # Wipe duplicate linkages
    
    degrees = np.array(A.sum(axis=1)).flatten()
    
    degrees_safe = np.where(degrees == 0, 1e-12, degrees)
    #till here is is same as previous spetral clustering algorithm 
    # Compute D^(-1/2)
    deg_inv_sqrt = 1.0 / np.sqrt(degrees_safe)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    
    I = sp.eye(num_nodes)
    L_sym = I - D_inv_sqrt.dot(A).dot(D_inv_sqrt)
    
    # Extract Smallest k Eigenvectors
    k = num_components + 1
    L_sym_shifted = L_sym.astype(float) + 1e-8 * sp.eye(num_nodes)
    eigenvalues, eigenvectors = eigsh(L_sym_shifted, k=k, which="SM", ncv=min(num_nodes, max(2*k+1, 50)), maxiter=num_nodes*10)
    
    # Drop first trivial vector (slice column 1 onward)
    return eigenvectors[:, 1:]
 