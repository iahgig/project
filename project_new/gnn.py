import torch 
import torch.nn as nn
import torch.nn.functional as F
from graph_learner import GraphLearner

class GCNLayer(nn.Module):
    def __init__(self,in_features,out_features,bias=True,batch_norm=True):
        super(GCNLayer,self).__init__()
        self.weight=torch.Tensor(in_features,out_features).to(torch.device('cpu'))
        self.weight=nn.Parameter(nn.init.xavier_uniform_(self.weight),requires_grad=True)
        if bias:
            self.bias=torch.Tensor(out_features)
            self.bias=nn.Parameter(nn.init.xavier_uniform_(self.bias.unsqueeze(0)),requires_grad=True)
        else:
            self.register_parameter('bias',None)
        
        self.bn=nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self,input,adj,batch_norm=True):
        support=torch.matmul(input,self.weight)
        output=torch.matmul(adj,support)
        ######################################
       
        ######################################
        if self.bias is not None:                                      
            output=output+self.bias

        if self.bn is not None and batch_norm:
            output=self.compute_bn(output)

        return output

    def compute_bn(self,x):
        if len(x.shape)==2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1,x.size(-1))).view(x.size())

class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.config=config

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GCNLayer(self.config['node_dim'], self.config['node_hidden_dim'], batch_norm=self.config['batch_norm']))

        for _ in range(self.config['graph_hops'] - 2):
            self.graph_encoders.append(GCNLayer(self.config['node_hidden_dim'], self.config['node_hidden_dim'], batch_norm=self.config['batch_norm']))

        self.graph_encoders.append(GCNLayer(self.config['node_hidden_dim'], self.config['node_dim'], batch_norm=self.config['batch_norm']))


    def forward(self, node_emb, adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            node_emb = F.relu(encoder(node_emb,adj))
            #node_emb = F.dropout(node_emb, self.config['dropout'], training=self.training)

        node_emb = F.relu(self.graph_encoders[-1](node_emb, adj))
        return node_emb



#### (batch_size,sensor_number,node_dim) ---> (batch_size,sensor_number,node_dim)
class Iter_GCN(nn.Module):
    def __init__(self,config):
        super(Iter_GCN,self).__init__()
        self.config=config
        self.gcn=GCNLayer(self.config['node_dim'],self.config['node_dim'],bias=self.config['bias'],batch_norm=self.config['batch_norm'])

    def forward(self,input,adj):
        output=F.relu(self.gcn(input,adj))
        return output


'''
z=torch.randn(3,5,5)
print(z)
gcn=GCN(5,10,2,2,dropout=True)
gl=GraphLearner(5,12,epsilon=0.5)
adj=gl(z)
print(adj)
output_z=gcn(z,adj)
print(output_z)
'''

