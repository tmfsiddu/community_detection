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


#here we are going to implement louvian algorithm and try experimenting with other greedy modularit algorithms
# Append this to the bottom of src/structural_baselines_11.py
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph

def compute_louvain_from_scratch(embeddings_np, n_neighbors=30, max_iter=1000):
   
    #as we use 256d vectors as input to this but louvian wants adjacency matrix for its computatio
    #we first convert convert that to an adjacency matrix like this kNN function looks at 256d embe and take 10 neighbors from it 
    #build_knn_similarity_graph and run_phase1_shuffling already do this exact logic so we reuse them
    A = build_knn_similarity_graph(embeddings_np, n_neighbors=n_neighbors)
    return run_phase1_shuffling(A, max_iter=max_iter)
 
#the above function does is phase one of louvian model now we will write functions for phase2
def aggregate_graph_into_super_nodes(A, cluster_assignments):
    num_nodes = A.shape[0]
    num_communities = len(np.unique(cluster_assignments))
    
    #create an empty sparse matrix blueprint for the new coarser graph
    #shape changes from (2708, 2708) down to (num_communities, num_communities)
    A_new = sp.dok_matrix((num_communities, num_communities), dtype=np.float32)
    
    #convert original matrix to COO(coordinate format) format to loop over active edges easily
    A_coo = A.tocoo()
    
    #loop over every single active link in the old graph
    for u, v, weight in zip(A_coo.row, A_coo.col, A_coo.data):
        #find which super-node (community) each endpoint belongs to
        super_u = cluster_assignments[u]
        super_v = cluster_assignments[v]
        
        #add the weight of the old link to the link between the two super-nodes
        #if super_u == super_v, this automatically creates a weighted self-loop!
        A_new[super_u, super_v] += weight
        #here is both nodes belong to same community we will increase the weight between those two super nodes 
    #convert back to compressed row format for fast math processing
    return A_new.tocsr()


#this function will build the adjacency matrix which louvian algo takes as input
#it takes out 256d vector and build a symmetric matrix by drawing lines b/w 15 closest points
def build_knn_similarity_graph(embeddings_np, n_neighbors=30):
    # converts dense embedding vectors into a sparse KNN adjacency matrix
    from sklearn.neighbors import kneighbors_graph
    A = kneighbors_graph(embeddings_np, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
    A = A + A.T
    A.data = np.ones_like(A.data)
    return A

#this function actually takes care of executing the phase1 of louvian algo
def run_phase1_shuffling(A, max_iter=1000):
    # runs phase1 of louvain (local greedy node reassignment) on a given adjacency matrix
    num_nodes = A.shape[0]
    communities = np.arange(num_nodes)
    adj_dict = {i: A.getrow(i).nonzero()[1] for i in range(num_nodes)}
    #these are variables for fast look up of nodes degree and neighbors 
    degrees = np.array(A.sum(axis=1)).flatten()
    m = np.sum(degrees) / 2.0
    if m == 0:
        return communities
    Sigma_tot = {c: degrees[c] for c in range(num_nodes)}
    iteration = 0
    improvement = True
    while improvement and iteration < max_iter:
        improvement = False
        iteration += 1
        for node in range(num_nodes):
            current_comm = communities[node]
            node_deg = degrees[node]
            neighbors = adj_dict[node]
            if len(neighbors) == 0:
                continue
            neighbor_communities = {}
            for neighbor in neighbors:
                comm = communities[neighbor]
                neighbor_communities[comm] = neighbor_communities.get(comm, 0) + 1
            neighbor_communities[current_comm] = neighbor_communities.get(current_comm, 0)
            Sigma_tot[current_comm] -= node_deg
            best_comm = current_comm
            max_delta_q = 0.0
            for comm, k_i_in in neighbor_communities.items():
                if comm == current_comm:
                    continue
                delta_q = (k_i_in / m) - (Sigma_tot[comm] * node_deg) / (2 * (m**2))
                if delta_q > max_delta_q:
                    max_delta_q = delta_q
                    best_comm = comm
            Sigma_tot[best_comm] += node_deg
            if best_comm != current_comm:
                communities[node] = best_comm
                improvement = True
    unique_comms = np.unique(communities)
    comm_map = {old: new for new, old in enumerate(unique_comms)}
    return np.array([comm_map[c] for c in communities])


def full_hierarchical_louvain(embeddings_np, n_neighbors=15):
    # phase 0  convert 256d spatial embeddings into a topological structural graph
    A = build_knn_similarity_graph(embeddings_np, n_neighbors)
    current_graph = A
    
    # Track the cluster assignment for every single original node (e.g., all 2708 papers)
    num_nodes = A.shape[0]
    macro_communities = np.arange(num_nodes)
    
    loop_condition = True
    while loop_condition:
        # phase1  local greedy node shuffling
        local_clusters = run_phase1_shuffling(current_graph)
        
        #if the number of communities didnt change, we hit peak modularity optimization
        if len(np.unique(local_clusters)) == current_graph.shape[0]:
            loop_condition = False
            break
            
        #mapping original nodes communities to its corresponding super nodes
        macro_communities = np.array([local_clusters[c] for c in macro_communities])
        
        # phase 2 convert the graph into super node structure 
        current_graph = aggregate_graph_into_super_nodes(current_graph, local_clusters)
        
    # Compress scattered indices down to a clean array from 0 to (unique_count - 1)
    unique_comms = np.unique(macro_communities)
    comm_map = {old: new for new, old in enumerate(unique_comms)}
    return np.array([comm_map[c] for c in macro_communities])