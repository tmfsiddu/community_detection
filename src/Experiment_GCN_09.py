import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv

#class for experimenting depth
class DepthExperimentGCN(torch.nn.Module):
    def __init__(self, num_features,num_classes,num_layers):
        super(DepthExperimentGCN,self).__init__()
        self.layers=torch.nn.ModuleList()
        #we are doing this inheritance from class and calling super because it is fundemental steps to build neural netwrok in pytorch
        #with this only our network weights and bias will automatically gets updated in models brain without this python will throw error
        self.layers.append(GCNConv(num_features,32))
        #this is the first layer which by default will output 16 dimension vector
        for _ in range(num_layers-2):
            self.layers.append(GCNConv(32,32))
        self.final_layer=GCNConv(32,num_classes)
    def forward(self,data):
        x,edge_index=data.x,data.edge_index
        #X is feature matrix and edge_index is adjacency matrix 
        for layer in self.layers:
            x=layer(x,edge_index)
            x=F.relu(x)
            x=F.dropout(x,p=0.5,training=self.training)
        x=self.final_layer(x,edge_index)
        return F.log_softmax(x,dim=1)
    
#class for experimenting the aggregator
class AggregationGCN(torch.nn.Module):
    def __init__(self, num_features,num_classes,aggr_type):
        super(AggregationGCN,self).__init__()
        self.conv1=GraphConv(num_features,32,aggr=aggr_type)
        self.conv2=GraphConv(32,num_classes,aggr=aggr_type)
    def forward(self,data):
        x,edge_index=data.x,data.edge_index
        x=self.conv1(x,edge_index)
        x=F.relu(x)
        x=F.dropout(x,p=0.5,training=self.training)
        x=self.conv2(x,edge_index)
        return F.log_softmax(x,dim=1)
    
#after finding the optimal architechture like using 2 layer depth and 32 dimension hidden vector and mean as an aggregator we are getting the nice results
#building an architechture containing all these 
class OptimalAggregationGCN(torch.nn.Module):
    def __init__(self, num_features,num_classes):
        super(OptimalAggregationGCN,self).__init__()
        self.conv1=GraphConv(num_features,32,aggr='mean')
        self.conv2=GraphConv(32,num_classes,aggr='mean')

    def forward(self,data):
        x,edge_index=data.x,data.edge_index
        x=self.conv1(x,edge_index)
        x=F.relu(x)
        x=F.dropout(x,p=0.5,training=self.training)
        x=self.conv2(x,edge_index)
        return F.log_softmax(x,dim=1)
    
#building an new inductive model because we have to change no of parameters in forward function
#we have to pass x and edge_index instead of complete data 
class InductiveGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(InductiveGCN, self).__init__()
        self.conv1 = GraphConv(num_features, 32, aggr='mean')
        self.conv2 = GraphConv(32, num_classes, aggr='mean')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

#building a standard baseline mlp without using any graph structure using only node features
# Add this class to the bottom of your Experiment_GCN_09.py file
class BaselineMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(BaselineMLP, self).__init__()
        #normal linear layers wihtout any graph convolution
        self.fc1 = torch.nn.Linear(num_features, 32)
        self.fc2 = torch.nn.Linear(32, num_classes)

    def forward(self, x, edge_index=None):
        #here we are not using any edge_index parameter which contains graph structure
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    