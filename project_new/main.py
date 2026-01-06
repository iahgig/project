import torch
import torch.nn.functional as F
from model import Model
from process import TimeDataset
import numpy as np
import pandas as pd
import datetime
from torch.utils.data import Dataset,DataLoader,random_split
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import argparse
import os
from pytorchtools import EarlyStopping

from torch.utils.tensorboard import SummaryWriter

#writer=SummaryWriter('./wadi')
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

parser=argparse.ArgumentParser()
parser.add_argument('-batch_size',help='batch size',type=int,default=64)
parser.add_argument('-epoches',help='Epoches to train',type=int,default=1)
parser.add_argument('-slide_win',help='slide win',type=int,default=5)
parser.add_argument('-slide_stride',help='slide stride',type=int,default=5)
parser.add_argument('-dataset',help='wadi/swad/kddcup99',type=str,default='kddcup99')
parser.add_argument('-class_number',help='class number',type=int,default=2)
parser.add_argument('-device',help='cuda/cpu',type=str,default='cuda') 
parser.add_argument('-sensor_emb_dim',help='sensor embeddings dim',type=int,default=64)
parser.add_argument('-train_size_ratio',help='train samples ratio',type=float,default=0.8)
parser.add_argument('-eval_size_ratio',help='evaluation samples ratio',type=float,default=0.3)
parser.add_argument('-node_hidden_dim',help='hidden layer node embedding dim',type=int,default=256)
parser.add_argument('-dev_hidden_dim1',help='deviation network first layer dim',type=int,default=64)
parser.add_argument('-dev_hidden_dim2',help='deviation network second layer dim',type=int,default=32)
parser.add_argument('-topk',help='topk links in graph',type=int,default=15)
parser.add_argument('-epsilon',help='threshold to select links in graph',type=float,default=0.85)
parser.add_argument('-lr',help='learning rate',type=float,default=1e-4)
parser.add_argument('-l',help='number of anomaly score samples',type=int,default=5000)
parser.add_argument('-margin',help='margin confidence',type=int,default=5)
parser.add_argument('-num_pers',help='number of perspectives',type=int,default=16)
parser.add_argument('-graph_hops',help='graph hops',type=int,default=2)
parser.add_argument('-threshold',help='anomaly score threshold',type=float,default=None)
parser.add_argument('-bias',help='bias',type=bool,default=True)
parser.add_argument('-batch_norm',help='batch norm',type=bool,default=True)
parser.add_argument('-lambd',help='linear combination in graph',type=float,default=0.6)
parser.add_argument('-delta',help='threshold to stop iteration',type=float,default=1e-5)
parser.add_argument('-weight_decay',help='weight_decay',type=float,default=1e-1)
parser.add_argument('-max_iter',help='max iterations',type=int,default=10)
#parser.add_argument('-MI_hidden_size1',help='MI first layer dim',type=int,default=64)
#parser.add_argument('-MI_hidden_size2',help='MI second layer dim',type=int,default=32)
parser.add_argument('-dropout',help='dropout',type=bool,default=True)
parser.add_argument('-MI',help='Whether MI',type=bool,default=True)
parser.add_argument('-patience',help='early stoppping patience',type=int,default=10)
parser.add_argument('-random_seed',help='random seed',type=int,default=1)
parser.add_argument('-test',type=bool,help='during testing phase',default=False)
parser.add_argument('-loss',help='loss parameter',type=float,default=0.8)
parser.add_argument('-iterative_flag',help='whether reveal iterative results',type=bool,default=False)
args=parser.parse_args()

# 数据集特定的阈值配置
DATASET_THRESHOLDS = {
    'kddcup99': 2.1576,  # 根据理论最佳阈值
    'swat': 0.0,         # 根据实际分析调整（理论最佳阈值-0.0404，但假阳性率太高）
    'wadi': 1.8          # 需要根据实际分析调整
}

# 如果用户没有显式指定阈值，使用数据集特定的默认阈值
if args.threshold is None:
    threshold = DATASET_THRESHOLDS.get(args.dataset, 1.0)
    print(f"使用数据集 '{args.dataset}' 的默认阈值: {threshold}")
else:
    threshold = args.threshold
    print(f"使用用户指定的阈值: {threshold}")

config={'delta':args.delta,'epsilon':args.epsilon,'topk':args.topk,'lambda':args.lambd,'weight_decay':args.weight_decay,'sensor_emb_dim':args.sensor_emb_dim,'slide_win':args.slide_win,'slide_stride':args.slide_stride,'batch_size':args.batch_size,'loss':args.loss,'random_seed':args.random_seed,'dataset':args.dataset,'class number':args.class_number,'l':args.l,'train_size_ratio':args.train_size_ratio,'eval_size_ratio':args.eval_size_ratio,
        'node_dim':2*args.sensor_emb_dim,'epoches':args.epoches,'device':torch.device('cpu'),'dev_hidden_dim1':args.dev_hidden_dim1,'dev_hidden_dim2':args.dev_hidden_dim2,
       'max_iter':args.max_iter,'node_hidden_dim':args.node_hidden_dim,'graph_hops':args.graph_hops,'num_pers':args.num_pers,'margin':args.margin,'threshold':threshold,'lr':args.lr,'patience':args.patience,
        'bias':args.bias,'batch_norm':args.batch_norm,'test':args.test,'dropout':args.dropout,'MI':args.MI,'iterative_flag':args.iterative_flag}
set_random_seed(config['random_seed'])
# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
f=open('./data/{}/list.txt'.format(config['dataset']))
sensors = f.readlines()  # 直接将文件中按行读到list里
f.close()  # close file 
config['number of sensors']=len(sensors)

data=pd.read_csv('./data/{}/test.csv'.format(config['dataset']))
train_data=pd.read_csv('./data/{}/train.csv'.format(config['dataset']))
data=data.drop(columns='Timestamp')
train_data=train_data.drop(columns='Timestamp')
data=TimeDataset(data,config)
train_dataset=TimeDataset(train_data,config)
train_size=int(len(train_data))
eval_size=int(len(data)*config['eval_size_ratio'])
eval_dataset,test_dataset=random_split(data,[eval_size,len(data)-eval_size])

train_anomaly=0
train_normal=0
eval_anomaly=0
eval_normal=0
test_anomaly=0
test_normal=0
for x,label in train_dataset:
    if label==1:
        train_anomaly+=1
    else:
        train_normal+=1
for x,label in eval_dataset:
    if label==1:
        eval_anomaly+=1
    else:
        eval_normal+=1
for x,label in test_dataset:
    if label==1:
        test_anomaly+=1
    else:
        test_normal+=1

train_weights=[1/(train_anomaly*2) if label==1 else 1/train_normal for x,label in train_dataset]
eval_weights=[1/eval_anomaly if label==1 else 1/eval_normal for x,label in eval_dataset]
test_weights=[1/(test_anomaly*2) if label==1 else 1/test_normal for x,label in test_dataset]
train_sampler=WeightedRandomSampler(weights=train_weights,num_samples=train_size,replacement=True)
eval_sampler=WeightedRandomSampler(weights=eval_weights,num_samples=eval_size,replacement=True)
test_sampler=WeightedRandomSampler(weights=test_weights,num_samples=len(data)-eval_size,replacement=True)

train_loader=DataLoader(train_dataset,batch_size=config['batch_size'])
eval_loader=DataLoader(eval_dataset,batch_size=config['batch_size'],sampler=eval_sampler)
test_loader=DataLoader(test_dataset,batch_size=config['batch_size'],sampler=test_sampler)
model=Model(config).to(config['device'])
print(model.sensor_emb)
print(config['dataset'])
##打印模型需要学习的参数
#for name, param in model.named_parameters():
	#print(name, '      ', param.size())

early_stopping=EarlyStopping(config['patience'],verbose=True)
optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
#scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
train_total_loss=[]
train_total_loss_clf=[]
train_total_loss_MI=[]
train_total_prec=[]
train_total_rec=[]
train_total_f1=[]
eval_total_loss=[]
eval_total_clf_loss=[]
eval_total_MI_loss=[]
eval_total_prec=[]
for epoch in range(1,config['epoches']+1):
    losses=0
    losses_MI=0
    losses_clf=0
    prec=0
    rec=0
    f1=0
    #####每次全部训练集的训练都要分批次训练
    for i, train_data in enumerate(train_loader):
        loss=0
        train_x=train_data[0].to(config['device'])
        train_labels=train_data[1].to(config['device'])
        score,loss_MI,loss_clf,prec_score,rec_score,f1_score=model(train_x,train_labels)
        #score=score.tolist()
        #if epoch%20==0:
            #plt.scatter(list(range(len(score))),score)
            #plt.show()
        if config['MI']:
            loss=loss_MI+loss_clf
        else:
            loss=loss_clf
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step(loss_clf)
        losses_clf+=loss_clf.item()
        losses_MI+=loss_MI.item()
        losses+=loss.item()
        prec+=prec_score if isinstance(prec_score, (float, int)) else prec_score.item()
        rec+=rec_score if isinstance(rec_score, (float, int)) else rec_score.item()
        f1+=f1_score if isinstance(f1_score, (float, int)) else f1_score.item()
    losses_clf=losses_clf/len(train_loader)
    losses_MI=losses_MI/len(train_loader)
    losses=losses/len(train_loader)
    prec=prec/len(train_loader)
    rec=rec/len(train_loader)
    f1=f1/len(train_loader)
    
    if epoch%1==0:
        with torch.no_grad():
            eval_losses=0
            eval_MI_losses=0
            eval_clf_losses=0
            eval_prec=0
            for i,eval_data in enumerate(eval_loader):
                eval_x=eval_data[0].to(config['device'])
                eval_labels=eval_data[1].to(config['device'])
                score,loss_MI,loss_clf,prec_score,rec_score,f1_score=model(eval_x,eval_labels)
                if config['MI']:
                    loss=loss_clf+loss_MI
                else:
                    loss=loss_clf
                eval_prec+=prec_score if isinstance(prec_score, (float, int)) else prec_score.item()
                eval_losses+=loss.item()
                eval_MI_losses+=loss_MI.item()
                eval_clf_losses+=loss_clf.item()
            eval_losses=eval_losses/len(eval_loader)
            eval_MI_losses=eval_MI_losses/len(eval_loader)
            eval_clf_losses=eval_clf_losses/len(eval_loader)
            eval_prec=eval_prec/len(eval_loader)

            eval_total_loss.append(eval_losses)
            eval_total_clf_loss.append(eval_clf_losses)
            eval_total_MI_loss.append(eval_MI_losses)
            eval_total_prec.append(eval_prec)
    

        early_stopping(eval_losses,model)
        if early_stopping.early_stop:
            print('Early stopping')
            break


    print('Epoch:{}/{},loss_MI:{},loss_clf:{},Prec:{},Rec:{},F1:{}'.format(epoch,config['epoches'],losses_MI,losses_clf,prec,rec,f1))
    train_total_loss.append(losses)
    train_total_loss_clf.append(losses_clf)
    train_total_loss_MI.append(losses_MI)
    train_total_prec.append(prec)
    train_total_rec.append(rec)
    train_total_f1.append(f1)

    
plt.plot(list(range(len(train_total_loss))),train_total_loss,label='total loss')
plt.plot(list(range(len(eval_total_loss))),eval_total_loss,label='eval loss')
plt.legend()
plt.savefig('CLF_loss.jpg')
plt.clf()
plt.plot(list(range(len(train_total_prec))),train_total_prec,label='train prec')
plt.plot(list(range(len(eval_total_prec))),eval_total_prec,label='eval prec')
plt.legend()
plt.savefig('Prec.jpg')
plt.clf()
plt.plot(list(range(len(train_total_loss_MI))),train_total_loss_MI,label='MI_loss')
plt.legend()
plt.savefig('MI_loss.jpg')

sensors=model.sensor_emb.detach().cpu().numpy()
sensors=pd.DataFrame(sensors)
sensors.to_excel('sensor_emb.xlsx')



print("##########################  test phase ########################")
config['test']=True
with torch.no_grad():
    losses=[]
    losses_MI=[]
    losses_clf=[]
    test_prec=[]
    test_rec=[]
    test_f1=[]
    for i,d in enumerate(test_loader):
        x=d[0].to(config['device'])
        labels=d[1].to(config['device'])
        score,loss_MI,loss_clf,prec_score,rec_score,f1_score=model(x,labels)
        if config['iterative_flag']==False:
            losses.append((loss_clf+loss_MI).item())
            losses_MI.append(loss_MI.item())
            losses_clf.append(loss_clf.item())
            test_prec.append(prec_score if isinstance(prec_score, (float, int)) else prec_score.item())
            test_rec.append(rec_score if isinstance(rec_score, (float, int)) else rec_score.item())
            test_f1.append(f1_score if isinstance(f1_score, (float, int)) else f1_score.item())
        else:
            losses.append(loss_clf+loss_MI)
            losses_MI.append(loss_MI)
            losses_clf.append(loss_clf)
            test_prec.append(prec_score)
            test_rec.append(rec_score)
            test_f1.append(f1_score)
    losses=np.array(losses)
    losses_clf=np.array(losses_clf)
    losses_MI=np.array(losses_MI)
    test_prec=np.array(test_prec)
    test_rec=np.array(test_rec)
    test_f1=np.array(test_f1)
    print("losses:{},losses_MI:{},losses_clf:{},Prec:{},Rec:{},F1:{}".format(np.mean(losses,axis=0),np.mean(losses_MI,axis=0),np.mean(losses_clf,axis=0),np.mean(test_prec,axis=0),np.mean(test_rec,axis=0),np.mean(test_f1,axis=0)))

#writer.add_embedding(model.sensor_emb,metadata=sensors)

# file=open('9.19_log.txt','a')
# file.write('---------------------------------------------------------------------------\n')
# file.write('Dataset:{}       '.format(config['dataset']))
# if config['MI']:
#     file.write('With MI.    ')
# else:
#     file.write('Without MI    ')
# file.write("Time: {}\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
# file.write("Experiment config:{} \n".format(config))
# file.write("Training result:loss:{},loss_clf:{},loss_MI:{},prec:{},rec:{},f1:{}\n".format(train_total_loss[-1],train_total_loss_clf[-1],train_total_loss_MI[-1],train_total_prec[-1],train_total_rec[-1],train_total_f1[-1]))
# file.write("Test result:loss:{},loss_clf:{},loss_MI:{},prec:{},rec:{},f1:{}\n".format(np.mean(losses,axis=0),np.mean(losses_clf,axis=0),np.mean(losses_MI,axis=0),np.mean(test_prec,axis=0),np.mean(test_rec,axis=0),np.mean(test_f1,axis=0)))
# file.close()
