import torch
from torch.utils.data import Dataset,DataLoader,random_split
import pandas as pd

#######  (time,sensor)->(sensor,time)(27,1564)    ############
def construct_data(raw):
    res=[]
    for sensor in raw.columns:
        res.append(raw.loc[:,sensor].tolist())
    return res


####### 将数据按照时间划分为图   ##########
####### slide_win, slide_stride #########
class TimeDataset(Dataset):
    def __init__(self,raw_data,config=None):
        self.raw_data=construct_data(raw_data)
        self.config=config

        x_data=self.raw_data[:-1]
        labels=self.raw_data[-1]

        data=x_data

        # to tensor
        data=torch.tensor(data).double()
        labels=torch.tensor(labels).double()

        self.x,self.labels=self.process(data,labels)

    def __len__(self):
        return len(self.x)
    
    def process(self,data,labels):

        x_arr=[]
        labels_arr=[]

        slide_win,slide_stride=[self.config[k] for k in ['slide_win','slide_stride']]
        total_time_len=data.shape[1]  

    
        for i in range(slide_win,total_time_len,slide_stride):
            x_arr.append(data[:,i-slide_win:i])
            labels_arr.append(labels[i-1])

        ### x_arr (batch_size,27,slide_win)  labels_arr: (batch_size,1)  batch_size:310
        x=torch.stack(x_arr).contiguous()
        labels=torch.stack(labels_arr).contiguous()
        #### 数据归一化
        max=torch.max(x)
        min=torch.min(x)
        x=(x-min)/(max-min)
        #############
        return x,labels

    def __getitem__(self,index):
        graph=self.x[index].double()
        label=self.labels[index].double()
        return graph,label


'''
config={'number of sensors':27,'sensor_emb_dim':64,'slide_win':15,'slide_stride':5,'class number':2,'l':100,
        'node_dim':128,'node_hidden_dim':256,'batch_size':64,'epoches':1000,'device':torch.device('cuda'),'dev_hidden_dim1':64,'dev_hidden_dim2':32,
        'graph_hidden_size':50,'topk':20,'epsilon':0.3,'max_iter':10,'graph_hops':2,'num_pers':12,'margine':1,
        'bias':'False','batch_norm':'True','lambda':0.4,'delta':1e-4,'MI_hidden_size1':64,'MI_hidden_size2':32,'dropout':'True'}

data=pd.read_csv('.\\data\\data\swat\\test.csv')
data=data.drop(columns='Timestamp')
data=TimeDataset(data,config)
train_size=int(len(data)*0.7)
test_size=len(data)-train_size
train_dataset,test_dataset=random_split(data,[train_size,test_size])
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=1,shuffle=True)

print("train: "+str(len(train_dataset)))
print("test: "+str(len(test_dataset)))
'''

