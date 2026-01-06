import torch
import numpy as np
import pandas as pd
from model import Model
from process import TimeDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data import WeightedRandomSampler
import argparse

def train_simple_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-epoches', type=int, default=1)  # 只训练1个epoch
    parser.add_argument('-slide_win', type=int, default=5)
    parser.add_argument('-slide_stride', type=int, default=5)
    parser.add_argument('-dataset', type=str, default='kddcup99')
    parser.add_argument('-class_number', type=int, default=2)
    parser.add_argument('-device', type=str, default='cpu')  # 使用CPU更快
    parser.add_argument('-sensor_emb_dim', type=int, default=64)
    parser.add_argument('-train_size_ratio', type=float, default=0.8)
    parser.add_argument('-eval_size_ratio', type=float, default=0.3)
    parser.add_argument('-node_hidden_dim', type=int, default=256)
    parser.add_argument('-dev_hidden_dim1', type=int, default=64)
    parser.add_argument('-dev_hidden_dim2', type=int, default=32)
    parser.add_argument('-topk', type=int, default=15)
    parser.add_argument('-epsilon', type=float, default=0.85)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-l', type=int, default=5000)
    parser.add_argument('-margin', type=int, default=5)
    parser.add_argument('-num_pers', type=int, default=16)
    parser.add_argument('-graph_hops', type=int, default=2)
    parser.add_argument('-threshold', type=float, default=1.96)
    parser.add_argument('-bias', type=bool, default=True)
    parser.add_argument('-batch_norm', type=bool, default=True)
    parser.add_argument('-lambd', type=float, default=0.6)
    parser.add_argument('-delta', type=float, default=1e-5)
    parser.add_argument('-weight_decay', type=float, default=1e-1)
    parser.add_argument('-max_iter', type=int, default=10)
    parser.add_argument('-dropout', type=bool, default=True)
    parser.add_argument('-MI', type=bool, default=True)
    parser.add_argument('-patience', type=int, default=10)
    parser.add_argument('-random_seed', type=int, default=1)
    parser.add_argument('-test', type=bool, default=False)
    parser.add_argument('-loss', type=float, default=0.8)
    parser.add_argument('-iterative_flag', type=bool, default=False)
    args = parser.parse_args([])

    config = {
        'delta': args.delta, 'epsilon': args.epsilon, 'topk': args.topk, 'lambda': args.lambd,
        'weight_decay': args.weight_decay, 'sensor_emb_dim': args.sensor_emb_dim,
        'slide_win': args.slide_win, 'slide_stride': args.slide_stride, 'batch_size': args.batch_size,
        'loss': args.loss, 'random_seed': args.random_seed, 'dataset': args.dataset,
        'class number': args.class_number, 'l': args.l, 'train_size_ratio': args.train_size_ratio,
        'eval_size_ratio': args.eval_size_ratio, 'node_dim': 2 * args.sensor_emb_dim,
        'epoches': args.epoches, 'device': torch.device(args.device), 'dev_hidden_dim1': args.dev_hidden_dim1,
        'dev_hidden_dim2': args.dev_hidden_dim2, 'max_iter': args.max_iter,
        'node_hidden_dim': args.node_hidden_dim, 'graph_hops': args.graph_hops,
        'num_pers': args.num_pers, 'margin': args.margin, 'threshold': args.threshold,
        'lr': args.lr, 'patience': args.patience, 'bias': args.bias, 'batch_norm': args.batch_norm,
        'test': args.test, 'dropout': args.dropout, 'MI': args.MI, 'iterative_flag': args.iterative_flag
    }

    # 加载数据
    f = open('./data/{}/list.txt'.format(config['dataset']))
    sensors = f.readlines()
    f.close()
    config['number of sensors'] = len(sensors)

    data = pd.read_csv('./data/{}/test.csv'.format(config['dataset']))
    train_data = pd.read_csv('./data/{}/train.csv'.format(config['dataset']))
    data = data.drop(columns='Timestamp')
    train_data = train_data.drop(columns='Timestamp')
    
    data = TimeDataset(data, config)
    train_dataset = TimeDataset(train_data, config)
    
    train_size = int(len(train_data))
    eval_size = int(len(data) * config['eval_size_ratio'])
    eval_dataset, test_dataset = random_split(data, [eval_size, len(data) - eval_size])

    # 创建训练数据加载器
    train_anomaly = 0
    train_normal = 0
    for x, label in train_dataset:
        if label == 1:
            train_anomaly += 1
        else:
            train_normal += 1
    
    train_weights = [1/(train_anomaly*2) if label == 1 else 1/train_normal for x, label in train_dataset]
    train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=train_size, replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler)

    # 创建模型
    model = Model(config).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # 训练一个epoch
    print("Training model for 1 epoch...")
    for epoch in range(1, config['epoches'] + 1):
        for i, train_data in enumerate(train_loader):
            train_x = train_data[0].to(config['device'])
            train_labels = train_data[1].to(config['device'])
            score, loss_MI, loss_clf, prec_score, rec_score, f1_score = model(train_x, train_labels)
            
            if config['MI']:
                loss = loss_MI + loss_clf
            else:
                loss = loss_clf
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}: loss={loss.item():.4f}, loss_clf={loss_clf.item():.4f}, loss_MI={loss_MI.item():.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'checkpoint.pt')
    print("Model saved to checkpoint.pt")
    
    return model, config

if __name__ == "__main__":
    train_simple_model()
