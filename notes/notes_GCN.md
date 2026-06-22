I want to implement GCN on a communtiy detection dataset starting from scartch.
so first i have to undestand what GCN is ,
intially we learned about Deepwalk and node2vec which are shallow node embeddings like representing the nodes in a low dimesional vector without using neural networks.

There are certain disadvantages of it , 
1.No parameter sharing-Like no node will share the information between them every node will have their own embeddings and own vector.
So here the major thing the algorithm just gives the embeddings of the nodes using lookup table  present in the graph does not learn how to give those embeddings.

2.Ignores node features-The shallow embedding consider only local structure in which the nodes are connected it does not consider what information does the node contains like what type of node and stuff.

3.Transductive-Like if a new node comes to the network then the algorithm has to re run again to get the new node embeddings as it only gives embeddings does not learn how to give them.

To overcome these disadvantages we are going with GNN(graph neural networks) where they will calculate the node embeddings based on both graph structure and node features.

But why GNN? why cant we use normal neural networks ??
Because as if we make this adjacency matrix of graph flat and feed that into an perceptron we will hit a major problem called node ordering Like in graph there is nothing like starting and ending node we can count arbitarily but if we chage the order the adjacency matrix changes but the graph structure remains same . Bur the output which we got in previous case will be different from which we get now as the input ( adjacency matrix ) changed

So as good GNN it should have following properties , permutation invariance and permutation equivariance.

permutation invariance- like if u change the order of nodes and tell the model to predict about the structure of graph then the output should be same .
f(P A P^T, P X) = f(A, X)
The above is the formula of permutation invariance where P is some permutation matrix it will change the rows and columns of given martrix and even after doing doing that the output shoudl be same as previous then only the GNN is said to be permutation invariant.
permutation equivariance- like in the same way if change the order of nodes and now tell the model to predict about embeddings of individual nodes then it should also change the output in that respective order.
f(P A P^T, P X) = P  f(A, X)
The above is the formula for permuatation equivariance where the order in which the input is changed the output also should change.

The GNN should use symmetric aggregation function in order to use acheive both and the three common used aggregaters which we can use are sum , average , max .
Like summing will always will give you the same answer irrespective of which order u sum them.

h_v^(new) = Update( h_v, Aggregate(  h_u : u N(v) ))

Like if you go for global aggregation then you can get permutation invariance and if you do local neighborhood aggregation the can acheive permutation equivariance.

There are three more important things which we should learn about GNN how they will update,

Neural message passing - Listens to every immediate neighbor and gethers information regarding them.

Aggregation - After gathering the information it will make a message vector contains info about all neighbors based on some aggregation function.

Update - Then finally for the message vector it adds its own information and then make changes to node embeddings 

The below is the core formula for GNN, 

$$
h_v^{(k)} = \sigma \left( W_k \sum_{u \in \mathcal{N}(v)} \frac{1}{|\mathcal{N}(v)|} h_u^{(k-1)} + B_k h_v^{(k-1)} \right)
$$

this is thing that GNN does to each node in each layer, where wk and bk are learnable weight matries that are given to a node neighbors and node itself.

in order to reduce that complex summation then there is something called message vector collects information from  all the incoming neighbor nodes and sums them up to a single context vector .

$$m_{\mathcal{N}(v)}^{(k)} = \sum_{u \in \mathcal{N}(v)} m_u^{(k)}$$

Now to reduce this summation operation there is a matrix representation of all this ,

$$H^{(k)} = \sigma \left( A H^{(k-1)} W_k \right)$$

But in the above formula the node is not considering itself as a neighbor so we are to add a indentity matrix for adjacency matrix and make the node remember its own information also 
 
$$\tilde{A} = A + I$$
By adding this the above thing becomes,
$$H^{(k)} = \sigma \left( \tilde{A} H^{(k-1)} W_k \right)$$

To normalize the formula as nodes with more neighbors might explode the message vector to overcome this the final formula is turned out to be ,

$$H^{(k)} = \sigma \left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(k-1)} W_k \right)$$

As in general the 1st layer in GCN will take 1-hop neighbors like the nodes which are immediately connected to it and 2nd layer 2-hop in that way how many layers our GCN contains that many no of hops we will go to neighbors and collect information.

At each layer we use above formula and update the node embeddings of nodes in graph.

Now i wanted to implement a GCN of two layers on Cora dataset and want to observe the results.The implementation is done in GCN.ipynb



