from graph_learner import GraphLearner
import torch
import torch.nn as nn
import torch.nn.functional as F
from measure_net import Dev_net
from measure_net import JSD_MI,New_MI,Raw_MI
from gnn import Iter_GCN,GCN
from graph_learner import GraphLearner


def diff(X,Y,Z):
    diff=torch.sum(torch.pow(X-Y,2),(1,2))
    norm=torch.sum(torch.pow(Z,2),(1,2))
    diff=diff/torch.clamp(norm,min=1e-12)
    return diff

def kNN(node_emb,topk):
    node_emb_norm=F.normalize(node_emb,p=2,dim=-1)
    similarity_score_matrix=torch.matmul(node_emb_norm,node_emb_norm.transpose(-1,-2))
    topk = min(topk, similarity_score_matrix.size(-1))
    knn_val, knn_ind = torch.topk(similarity_score_matrix, topk, dim=-1)
    #weighted_adjacency_matrix=(torch.ones_like(attention)).scatter_(-1,knn_ind,knn_val).to(self.device)
    similarity_score_matrix = (torch.zeros_like(similarity_score_matrix)).scatter_(-1, knn_ind, knn_val).to(torch.device('cpu'))
    #adj=F.normalize(adj,p=1,dim=-1)
    #adj =adj / torch.clamp(torch.sum(adj, dim=-1, keepdim=True), min=1e-12)
    adj=torch.where(similarity_score_matrix>0,torch.ones_like(similarity_score_matrix),similarity_score_matrix)
    return adj

def Laplacian_matrix(adj):
    row_sum=torch.sum(adj,-1)
    d_inv_sqrt=torch.pow(row_sum,-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)]=0.0
    d_mat_inv_sqrt=torch.diagflat(d_inv_sqrt)
    L_norm=torch.mm(torch.mm(d_mat_inv_sqrt,adj),d_mat_inv_sqrt)
    return L_norm

def batch_normalize_adj(adj):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    # mx: shape: [batch_size, N, N]

    # strategy 1)
    # rowsum = mx.sum(1)
    # r_inv_sqrt = torch.pow(rowsum, -0.5)
    # r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0. # I got this error: copy_if failed to synchronize: device-side assert triggered

    # strategy 2)
    rowsum = torch.clamp(adj.sum(1), min=1e-12)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    r_mat_inv_sqrt = []
    for i in range(r_inv_sqrt.size(0)):
        r_mat_inv_sqrt.append(torch.diag(r_inv_sqrt[i]))

    r_mat_inv_sqrt = torch.stack(r_mat_inv_sqrt, 0)
    return torch.matmul(torch.matmul(adj, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)

class AvgReadOut(nn.Module):
    def __init__(self):
        super(AvgReadOut,self).__init__()
    def forward(self,input):
        return torch.mean(input,dim=1)

class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout,self).__init__()
    def forward(self,input):
        return torch.max(input,dim=1)[0]

class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.config=config
        self.sensor_emb=torch.Tensor(config['number of sensors'],config['sensor_emb_dim'])
        self.sensor_emb=nn.init.xavier_uniform_(self.sensor_emb)
        self.sensor_emb=nn.Parameter(self.sensor_emb,requires_grad=True)
        self.w1=torch.Tensor(config['slide_win'],config['sensor_emb_dim'])
        self.w1=nn.init.xavier_uniform_(self.w1)
        self.w1=nn.Parameter(self.w1,requires_grad=True)
        self.graph_learner=GraphLearner(input_size=self.config['node_dim'],num_pers=self.config['num_pers'],device=self.config['device'])
        self.iter_gnn=GCN(self.config)
        #self.iter_gnn=Iter_GCN(self.config)
        self.readout=AvgReadOut()
        #self.readout=MaxReadout()
        #### JSD_MI (init_emb,s)
        ### Raw_MI (x,s)
        ### New_MI (cur_node_emb,s)
        #self.mine=JSD_MI(self.config)
        #self.mine=New_MI(self.config)
        self.mine=Raw_MI(self.config)
        self.dev_net=Dev_net(self.config)

    def forward(self,x,labels):

        ## x:(batch_size,sensor number, slide_win)
        ### compute z0
        x=x.to(torch.float32)
        init_node_emb=torch.matmul(x,self.w1)

        #################################################

        sensor_emb=self.sensor_emb.unsqueeze(0)
        sensor_emb=self.sensor_emb.repeat(x.shape[0],1,1)
        #################################################
        
        init_node_emb=torch.cat([init_node_emb,sensor_emb],dim=-1)

        init_adj=kNN(init_node_emb,self.config['topk'])

        
        #### 循环迭代
        iter=1
        first_raw_adj=self.graph_learner(init_node_emb,topk=None,epsilon=self.config['epsilon'])
        normed_cur_adj=self.config['lambda']*F.normalize(first_raw_adj,p=1,dim=-1)+(1-self.config['lambda'])*batch_normalize_adj(init_adj)
        #normed_cur_adj=self.config['lambda']*F.normalize(first_raw_adj,p=1,dim=-1)+(1-self.config['lambda'])*F.normalize(init_adj,p=1,dim=-1)
        cur_node_emb=self.iter_gnn(init_node_emb,normed_cur_adj)
        pre_raw_adj=init_adj.clone()     
        cur_raw_adj=first_raw_adj.clone()
        if self.config['iterative_flag']:
            iter_loss=[]
            iter_loss_clf=[]
            iter_loss_MI=[]
            iter_prec=[]
            iter_rec=[]
            iter_f1=[]
            z=self.readout(cur_node_emb)
            loss_MI=self.mine(x,z)
            #loss_MI=self.mine(cur_node_emb,z)
            #loss_MI=self.mine(init_node_emb,z)
            score,loss_clf,prec,rec,f1=self.dev_net(z,labels)
            iter_loss.append((loss_clf+loss_MI).item())
            iter_loss_clf.append(loss_clf.item())
            iter_loss_MI.append(loss_MI.item())
            iter_prec.append(prec)
            iter_rec.append(rec)
            iter_f1.append(f1)
            while iter<10:
                iter+=1
                #pre_adj=cur_adj
                #pre_raw_adj=cur_raw_adj
                cur_raw_adj=self.graph_learner(cur_node_emb,topk=None,epsilon=self.config['epsilon'])
                normed_cur_adj=self.config['lambda']*F.normalize(cur_raw_adj,p=1,dim=-1)+(1-self.config['lambda'])*F.normalize(init_adj,p=1,dim=-1)
                cur_node_emb=self.iter_gnn(init_node_emb,normed_cur_adj)                
                z=self.readout(cur_node_emb)
                loss_MI=self.mine(x,z)
                #loss_MI=self.mine(cur_node_emb,z)
                #loss_MI=self.mine(init_node_emb,z)
                score,loss_clf,prec,rec,f1=self.dev_net(z,labels)
                iter_loss.append((loss_clf+loss_MI).item())
                iter_loss_clf.append(loss_clf.item())
                iter_loss_MI.append(loss_MI.item())
                iter_prec.append(prec)
                iter_rec.append(rec)
                iter_f1.append(f1)
            return iter_loss,iter_loss_clf,iter_loss_MI,iter_prec,iter_rec,iter_f1
            
                

        else:  
            while (iter==1 or torch.sum(torch.gt(diff(cur_raw_adj,pre_raw_adj,first_raw_adj),self.config['delta']*torch.ones_like(diff(cur_raw_adj,pre_raw_adj,first_raw_adj)))).item()) and iter<self.config['max_iter']:
                iter+=1
                pre_raw_adj=cur_raw_adj.clone()
                cur_raw_adj=self.graph_learner(cur_node_emb,topk=None,epsilon=self.config['epsilon'])
                normed_cur_adj=self.config['lambda']*F.normalize(cur_raw_adj,p=1,dim=-1)+(1-self.config['lambda'])*F.normalize(init_adj,p=1,dim=-1)
                #L_norm=batch_normalize_adj(cur_adj)
                cur_node_emb=self.iter_gnn(cur_node_emb,normed_cur_adj)
            
            ####(batch_size,node_num,dim)-->(batch_size,dim)
            z=self.readout(cur_node_emb)    

            #loss_MI=self.mine(cur_node_emb,z)
            #loss_MI=self.mine(init_node_emb,z)
            loss_MI=self.mine(x,z)
            score,loss_clf,prec,rec,f1=self.dev_net(z,labels)
            return score,loss_MI,loss_clf,prec,rec,f1
            
       


