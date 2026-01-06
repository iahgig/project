import torch 
import torch.nn as nn
import torch.nn.functional as F


def compute_normalized_laplacian(adj):
    row_sum=torch.sum(adj,-1)
    d_inv_sqrt=torch.pow(row_sum,-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)]=0.0
    d_mat_inv_sqrt=torch.diagflat(d_inv_sqrt)
    L_norm=torch.mm(torch.mm(d_mat_inv_sqrt,adj),d_mat_inv_sqrt)
    return L_norm


class GraphLearner(nn.Module):
    def __init__(self,input_size,num_pers=16,device=None):
        super(GraphLearner,self).__init__()
        self.device=device
        #self.metric_type = metric_type
        ###########  W_s  ################
        self.weight_s=torch.Tensor(num_pers,input_size)
        self.weight_s=nn.Parameter(nn.init.xavier_uniform_(self.weight_s),requires_grad=True)



    def forward(self,x,topk,epsilon):
        '''
        x (batch_size,sensor_number(N),dim)
        '''
        expand_weight_tensor = self.weight_s.unsqueeze(1)
        if len(x.shape) == 3:
            expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

        x_fc = x.unsqueeze(0) * expand_weight_tensor
        #####所有的值除以某个维度的L2范数
        x_norm = F.normalize(x_fc, p=2, dim=-1)
        similarity_score_matrix = torch.matmul(x_norm, x_norm.transpose(-1, -2)).mean(0)

        if epsilon is not None:
            adj=self.build_epsilon_neighbour(similarity_score_matrix,epsilon)
        if topk is not None:
            adj=self.build_knn_neighbourhood(similarity_score_matrix,topk)
        
        return adj

    def build_knn_neighbourhood(self, attention, topk):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        #weighted_adjacency_matrix=(torch.ones_like(attention)).scatter_(-1,knn_ind,knn_val).to(self.device)
        weighted_adjacency_matrix = (torch.zeros_like(attention)).scatter_(-1, knn_ind, knn_val).to(self.device)
        adj=torch.where(weighted_adjacency_matrix>0,torch.ones_like(weighted_adjacency_matrix),weighted_adjacency_matrix)
        return adj


    def build_epsilon_neighbour(self,attention,epsilon):
        adj=torch.where(attention>epsilon,torch.ones_like(attention),torch.zeros_like(attention))
        #weighted_adjacency_matrix=F.normalize(weighted_adjacency_matrix,p=1,dim=-1)
        #weighted_adjacency_matrix = weighted_adjacency_matrix / torch.clamp(torch.sum(weighted_adjacency_matrix, dim=-1, keepdim=True), min=1e-12)
        return adj
        

