import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score,recall_score
'''
def dev_net(y,labels,l,margine):
    l_samples=torch.randn(l,requires_grad=False)
    mean=torch.mean(l_samples)
    std=torch.std(l_samples)
    y=torch.softmax(y,dim=-1)
    y=torch.argmax(y,dim=-1)
    deviation=(y-mean)/std
    loss_clf=(1-labels)*torch.abs(deviation)+labels*torch.max(0,margine-deviation)
    return torch.sum(loss_clf,dim=-1)
'''




class AvgReadOut(nn.Module):
    def __init__(self):
        super(AvgReadOut,self).__init__()
    def forward(self,input):
        return torch.max(input,dim=1)[0]

#####  (batch_size,node_dim) ---> (batch_size) (anomaly score)
class Dev_net(nn.Module):
    def __init__(self,config):
        super(Dev_net,self).__init__()
        self.config=config
 
        self.lin1=nn.Linear(self.config['node_dim'],self.config['dev_hidden_dim1'])
        self.lin2=nn.Linear(self.config['dev_hidden_dim1'],self.config['dev_hidden_dim2'])
        self.lin3=nn.Linear(self.config['dev_hidden_dim2'],1)
        ##swat initialize
        torch.nn.init.xavier_normal_(self.lin1.weight.data)
        torch.nn.init.xavier_normal_(self.lin2.weight.data)
        torch.nn.init.xavier_normal_(self.lin3.weight.data)
        #wadi initialize
        #torch.nn.init.normal_(self.lin1.weight.data, 0.01, 0.01)
        #torch.nn.init.normal_(self.lin2.weight.data, 0.01, 0.01)
        #torch.nn.init.normal_(self.lin3.weight.data, 0.01, 0.01)

    def forward(self,z,labels):
        ref = torch.normal(mean=0., std=torch.full([5000], 1.)).to(self.config['device'])
        #### z (batch_size,node_dim)

        score=F.relu(self.lin1(z),inplace=True)
        score=F.relu(self.lin2(score),inplace=True)
        score=F.relu(self.lin3(score),inplace=True)
        ### anomaly scores
        score=score.squeeze(1)  
        ### deviation loss
        deviation=(score-torch.mean(ref))/torch.std(ref)
       
        ##训练阶段不应该设置阈值
        #######################################################
        if self.config['test']==True:
            ones=torch.ones_like(score)
            zeros=torch.zeros_like(score)
            ### >zp为异常
            y=torch.where(deviation>self.config['threshold'],ones,zeros)
            y_hat=y.tolist()
            prec=precision_score(labels.tolist(),y_hat,zero_division=0)
            rec=recall_score(labels.tolist(),y_hat,zero_division=0)
            # 转换为float以保持类型一致
            prec = float(prec)
            rec = float(rec)
        else:
            prec=rec=0.0

        if prec + rec == 0:
            f1=0
        else:
            f1=2*prec*rec/(prec+rec)
        loss_clf=(1-labels)*torch.abs(deviation)+labels*torch.max(torch.zeros_like(deviation),self.config['margin']-deviation)
        return score,torch.mean(loss_clf,dim=0),prec,rec,f1



class MINE(nn.Module):
    def __init__(self,config):
        super(MINE,self).__init__()
        self.config=config
        self.layers=nn.Sequential(nn.Linear((config['slide_win']+config['node_dim'])*config['number of sensors'],config['MI_hidden_size1']),
                                nn.ReLU(),nn.Linear(config['MI_hidden_size1'],config['MI_hidden_size2']),nn.Linear(config['MI_hidden_size2'],1))
    def forward(self,x,y):
        batch_size=x.shape[0]
        x=x.view(x.shape[0],-1)
        y=y.view(y.shape[0],-1)
        tiled_x=torch.cat([x,x],dim=0)
        idx=torch.randperm(x.shape[0])
        shuffled_y=y[idx,:]
        concat_y=torch.cat([y,shuffled_y],dim=0)
        inputs=torch.cat([tiled_x,concat_y],dim=-1)
        logits=self.layers(inputs)


        ### joint distribution
        pred_xy=logits[:batch_size,:]
        ### marginal distribution
        pred_x_y=logits[batch_size:,:]
        loss_MI=-torch.mean(pred_xy,dim=0)+torch.log(torch.mean(torch.exp(pred_x_y),dim=0))
        return loss_MI



class New_MI(nn.Module):
    def __init__(self,config):
        super(New_MI,self).__init__()
        self.config=config
        self.discriminator=nn.Bilinear(config['node_dim'],config['node_dim'],1)
        self.obj=nn.BCEWithLogitsLoss()
    def forward(self,cur_node_emb,z):
        # cur_node_emb(batch_size,node_num,node_dim) 
        # z(batch_size,node_num,node_dim)
        batch_size=cur_node_emb.shape[0]
        node_num=cur_node_emb.shape[1]
        s=z.unsqueeze(1)
        s=s.expand_as(cur_node_emb)
        ### s(batch_size,node_num,node_dim)
        s=torch.cat([s,s],dim=0)
        index=torch.randperm(batch_size)
        shuffle_emb=cur_node_emb[index,:,:]
        cur_node_emb=torch.cat([cur_node_emb,shuffle_emb],dim=0)
        output=self.discriminator(cur_node_emb,s)
        output=output.squeeze(2)
        ones=torch.ones(batch_size,node_num)
        zeros=torch.zeros(batch_size,node_num)
        label=torch.cat([ones,zeros],dim=0).to(self.config['device'])
        return self.obj(output,label)
        #return -torch.mean(-torch.log(1+torch.exp(-real)))+torch.mean(torch.log(1+torch.exp(fake)))

class Raw_MI(nn.Module):
    def __init__(self,config):
        super(Raw_MI,self).__init__()
        self.config=config
        self.discriminator=nn.Bilinear(self.config['slide_win'],self.config['node_dim'],1)
        self.obj=nn.BCEWithLogitsLoss()
        torch.nn.init.normal_(self.discriminator.weight.data, 0.1, 0.01)
    def forward(self,raw_data,z):
        ## raw_data (batch_size,node_num,node_dim)
        ##  z (batch_size,node_dim)
        batch_size=raw_data.shape[0]
        node_num=raw_data.shape[1]
        s=z.unsqueeze(1)
        s=s.repeat(1,node_num,1)
        s=torch.cat([s,s],dim=0)
        index=torch.randperm(batch_size)
        shuffle_x=raw_data[index,:,:]
        x=torch.cat([raw_data,shuffle_x],dim=0)
        output=self.discriminator(x,s)
        output=output.squeeze(2)
        ones=torch.ones(batch_size,node_num)
        zeros=torch.zeros(batch_size,node_num)
        label=torch.cat([ones,zeros],dim=0).to(self.config['device'])
        return self.obj(output,label)


class JSD_MI(nn.Module):
    def __init__(self,config):
        super(JSD_MI,self).__init__()
        self.config=config
        self.readout=AvgReadOut()
        self.bili=nn.Bilinear(2*config['sensor_emb_dim'],config['node_dim'],1)
        self.obj=nn.BCEWithLogitsLoss()
        torch.nn.init.xavier_normal(self.bili.weight.data)


    def forward(self,x,z):
        ###  x(batch_size,node_num,sensor_emb_dim*2(node_dim))   z(batch_size,node_dim)   emb(batch_size,node_num,emb_dim)
        batch_size=x.shape[0]
        node_num=x.shape[1]
        #x=torch.matmul(x,w)
        #x=torch.cat([x,emb],dim=-1)
        s=z.unsqueeze(1)
        s=s.expand_as(x)
        ##### s (batch_size,node_num,node_dim)
        s=torch.cat([s,s],dim=0)
        index=torch.randperm(batch_size)
        shuffle_x=x[index,:,:]
        x=torch.cat([x,shuffle_x],dim=0)
        output=self.bili(x,s)
        #output=torch.sigmoid(output)
        output=output.squeeze(2)
        ones=torch.ones(batch_size,node_num)
        zeros=torch.zeros(batch_size,node_num)
        label=torch.cat([ones,zeros],dim=0).to(self.config['device'])
        return self.obj(output,label)

'''
config={'slide_win':15,'node_dim':128,'batch_size':5,'number of sensors':27,'MI_hidden_size1':64,'sensor_emb_dim':64,'MI_hidden_size2':32}
model=JSD_MI(config)
x=torch.randn(5,27,15)
z=torch.randn(5,27,128)
emb=torch.rand(5,27,64)
loss=model(x,z,emb)
print(loss)
'''
