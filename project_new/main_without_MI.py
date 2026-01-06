import torch
from torch import autograd
import torch.nn as nn 
import torch.nn.functional as F
from model import Model
from process import TimeDataset,construct_data
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader,random_split
import matplotlib.pyplot as plt
import argparse
import datetime

parser=argparse.ArgumentParser()
parser.add_argument('-batch_size',help='batch size',type=int,default=256)
parser.add_argument('-epoches',help='Epoches to train',type=int,default=1000)
parser.add_argument('-slide_win',help='slide win',type=int,default=5)
parser.add_argument('-slide_stride',help='slide stride',type=int,default=5)
parser.add_argument('-dataset',help='wadi/swad',type=str,default='swat')
parser.add_argument('-class_number',help='class number',type=int,default=2)
parser.add_argument('-device',help='cuda/cpu',type=str,default='cuda')
parser.add_argument('-sensor_emb_dim',help='sensor embeddings dim',type=int,default=64)
parser.add_argument('-train_size_ratio',help='train samples ratio',type=float,default=0.7)
parser.add_argument('-node_dim',help='node embedding dim',type=int,default=128)
parser.add_argument('-node_hidden_dim',help='hidden layer node embedding dim',type=int,default=256)
parser.add_argument('-dev_hidden_dim1',help='deviation network first layer dim',type=int,default=64)
parser.add_argument('-dev_hidden_dim2',help='deviation network second layer dim',type=int,default=32)
parser.add_argument('-topk',help='topk links in graph',type=int,default=35)
parser.add_argument('-epsilon',help='threshold to select links in graph',type=float,default=0.09)
parser.add_argument('-lr',help='learning rate',type=float,default=1e-4)
parser.add_argument('-l',help='number of anomaly score samples',type=int,default=5000)
parser.add_argument('-margine',help='margine confidence',type=int,default=5)
parser.add_argument('-num_pers',help='number of perspectives',type=int,default=16)
parser.add_argument('-graph_hops',help='graph hops',type=int,default=2)
parser.add_argument('-threshold',help='anomaly score threshold',type=float,default=1.96)
parser.add_argument('-bias',help='bias',type=bool,default=True)
parser.add_argument('-batch_norm',help='batch norm',type=bool,default=True)
parser.add_argument('-lambd',help='linear combination in graph',type=float,default='0.4')
parser.add_argument('-delta',help='threshold to stop iteration',type=float,default=1e-4)
parser.add_argument('-max_iter',help='max iterations',type=int,default=10)
parser.add_argument('-MI_hidden_size1',help='MI first layer dim',type=int,default=64)
parser.add_argument('-MI_hidden_size2',help='MI second layer dim',type=int,default=32)
parser.add_argument('-dropout',help='dropout',type=bool,default=True)
args=parser.parse_args()


config={'dataset':args.dataset,'sensor_emb_dim':args.sensor_emb_dim,'slide_win':args.slide_win,'slide_stride':args.slide_stride,'class number':args.class_number,'l':args.l,'train_size_ratio':args.train_size_ratio,
        'node_dim':args.node_dim,'node_hidden_dim':args.node_hidden_dim,'batch_size':args.batch_size,'epoches':args.epoches,'device':torch.device('cuda'),'dev_hidden_dim1':args.dev_hidden_dim1,'dev_hidden_dim2':args.dev_hidden_dim2,
       'topk':args.topk,'epsilon':args.epsilon,'max_iter':args.max_iter,'graph_hops':args.graph_hops,'num_pers':args.num_pers,'margine':args.margine,'threshold':args.threshold,'lr':args.lr,
        'bias':args.bias,'batch_norm':args.batch_norm,'lambda':args.lambd,'delta':args.delta,'MI_hidden_size1':args.MI_hidden_size1,'MI_hidden_size2':args.MI_hidden_size2,'dropout':args.dropout}

f=open('.\\data\\data\{}\\list.txt'.format(config['dataset']))
sensors = f.readlines()  # 直接将文件中按行读到list里
f.close()  # 关
config['number of sensors']=len(sensors)


data=pd.read_csv('.\\data\\data\swat\\test.csv')
data=data.drop(columns='Timestamp')
data=TimeDataset(data,config)
train_size=int(len(data)*config['train_size_ratio'])
train_dataset,test_dataset=random_split(data,[train_size,len(data)-train_size])
train_loader=DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=True)
model=Model(config).to(config['device'])
print(model.sensor_emb)

##打印模型需要学习的参数
#for name, param in model.named_parameters():
	#print(name, '      ', param.size())


optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
total_loss=[]
total_loss_clf=[]
#total_loss_MI=[]
total_prec=[]
total_rec=[]
total_f1=[]
for epoch in range(1,config['epoches']+1):
    losses=[]
    losses_clf=[]
    #losses_MI=[]
    prec=[]
    rec=[]
    f1=[]
    #####每次全部训练集的训练都要分批次训练
    for i, d in enumerate(train_loader):
        loss=0
        x=d[0].to(config['device'])
        labels=d[1].to(config['device'])
        score,loss_MI,loss_clf,prec_score,rec_score,f1_score=model(x,labels)
        score=score.tolist()
        loss=loss_clf
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #losses_MI.append(loss_MI.item())
        #losses_clf.append(loss_clf.item())
        losses.append(loss.item())
        prec.append(prec_score.item())
        rec.append(rec_score.item())
        f1.append(f1_score.item())
    print('Epoch:{}/{},loss:{},Prec:{},Rec:{},F1:{}'.format(epoch,config['epoches'],np.mean(losses),np.mean(prec),np.mean(rec),np.mean(f1)))
    total_loss.append(np.mean(losses))
    total_loss_clf.append(np.mean(losses_clf))
    #total_loss_MI.append(np.mean(losses_MI))
    total_prec.append(np.mean(prec))
    total_rec.append(np.mean(rec))
    total_f1.append(np.mean(f1))
plt.plot(list(range(config['epoches'])),total_loss,label='total_loss')
#plt.plot(list(range(config['epoches'])),total_loss_clf,label='CLF_loss')
#plt.plot(list(range(config['epoches'])),total_loss_MI,label='MI_loss')
plt.legend()
plt.ylim((-10,30))
plt.show()
print(model.sensor_emb)

print("##########################  test phase ########################")
with torch.no_grad():
    for i,d in enumerate(test_loader):
        losses=[]
        losses_MI=[]
        losses_clf=[]
        test_prec=[]
        test_rec=[]
        test_f1=[]
        x=d[0].to(config['device'])
        labels=d[1].to(config['device'])
        score,loss_MI,loss_clf,prec_score,rec_score,f1_score=model(x,labels)
        losses.append((loss_MI+loss_clf).item())
        #losses_MI.append(losses_MI.item())
        losses_clf.append(loss_clf.item())
        prec.append(prec_score.item())
        rec.append(rec_score.item())
        f1.append(f1_score.item())
    print("losses:{},Prec:{},Rec:{},F1:{}".format(np.mean(losses),np.mean(prec),np.mean(rec),np.mean(f1)))

file=open('Experiment_log.txt','a')
file.write('---------------------------------------------------------------------------\n')
file.write('Dataset:{}       '.format(config['dataset']))
file.write("Time: {}\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
file.write("Experiment config:{} \n".format(config))
file.write("Training result:loss:{},loss_clf:{},prec:{},rec:{},f1:{}\n".format(total_loss[-1],total_loss_clf[-1],total_prec[-1],total_rec[-1],total_f1[-1]))
file.write("Test result:loss:{},loss_clf:{},prec:{},rec:{},f1:{}\n".format(np.mean(losses),np.mean(losses_clf),np.mean(test_prec),np.mean(test_rec),np.mean(test_f1)))
file.close()
