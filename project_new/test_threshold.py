import torch
import numpy as np
import pandas as pd
from model import Model
from process import TimeDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data import WeightedRandomSampler
import argparse

def test_with_threshold(threshold_value):
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-slide_win', type=int, default=5)
    parser.add_argument('-slide_stride', type=int, default=5)
    parser.add_argument('-dataset', type=str, default='kddcup99')
    parser.add_argument('-class_number', type=int, default=2)
    parser.add_argument('-device', type=str, default='cuda')
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
    parser.add_argument('-threshold', type=float, default=threshold_value)  # 使用传入的阈值
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
    parser.add_argument('-test', type=bool, default=True)  # 设置为测试模式
    parser.add_argument('-loss', type=float, default=0.8)
    parser.add_argument('-iterative_flag', type=bool, default=False)
    args = parser.parse_args([])  # 空列表表示使用默认值

    config = {
        'delta': args.delta, 'epsilon': args.epsilon, 'topk': args.topk, 'lambda': args.lambd,
        'weight_decay': args.weight_decay, 'sensor_emb_dim': args.sensor_emb_dim,
        'slide_win': args.slide_win, 'slide_stride': args.slide_stride, 'batch_size': args.batch_size,
        'loss': args.loss, 'random_seed': args.random_seed, 'dataset': args.dataset,
        'class number': args.class_number, 'l': args.l, 'train_size_ratio': args.train_size_ratio,
        'eval_size_ratio': args.eval_size_ratio, 'node_dim': 2 * args.sensor_emb_dim,
        'epoches': 1, 'device': torch.device('cpu'), 'dev_hidden_dim1': args.dev_hidden_dim1,
        'dev_hidden_dim2': args.dev_hidden_dim2, 'max_iter': args.max_iter,
        'node_hidden_dim': args.node_hidden_dim, 'graph_hops': args.graph_hops,
        'num_pers': args.num_pers, 'margin': args.margin, 'threshold': args.threshold,
        'lr': args.lr, 'patience': args.patience, 'bias': args.bias, 'batch_norm': args.batch_norm,
        'test': args.test, 'dropout': args.dropout, 'MI': args.MI, 'iterative_flag': args.iterative_flag
    }

    # 加载数据
    f = open('project_new/data/{}/list.txt'.format(config['dataset']))
    sensors = f.readlines()
    f.close()
    config['number of sensors'] = len(sensors)

    data = pd.read_csv('project_new/data/{}/test.csv'.format(config['dataset']))
    train_data = pd.read_csv('project_new/data/{}/train.csv'.format(config['dataset']))
    data = data.drop(columns='Timestamp')
    train_data = train_data.drop(columns='Timestamp')
    
    data = TimeDataset(data, config)
    train_dataset = TimeDataset(train_data, config)
    
    train_size = int(len(train_data))
    eval_size = int(len(data) * config['eval_size_ratio'])
    eval_dataset, test_dataset = random_split(data, [eval_size, len(data) - eval_size])

    # 创建测试数据加载器
    test_anomaly = 0
    test_normal = 0
    for x, label in test_dataset:
        if label == 1:
            test_anomaly += 1
        else:
            test_normal += 1
    
    test_weights = [1/(test_anomaly*2) if label == 1 else 1/test_normal for x, label in test_dataset]
    test_sampler = WeightedRandomSampler(weights=test_weights, num_samples=len(data)-eval_size, replacement=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], sampler=test_sampler)

    # 加载模型
    model = Model(config).to(config['device'])
    
    # 加载预训练权重（如果存在）
    try:
        model.load_state_dict(torch.load('checkpoint.pt', map_location=config['device']))
        print(f"Loaded pretrained model for threshold={threshold_value}")
    except:
        print(f"No pretrained model found for threshold={threshold_value}, using random initialization")

    # 测试
    config['test'] = True
    with torch.no_grad():
        test_prec = []
        test_rec = []
        test_f1 = []
        for i, d in enumerate(test_loader):
            x = d[0].to(config['device'])
            labels = d[1].to(config['device'])
            score, loss_MI, loss_clf, prec_score, rec_score, f1_score = model(x, labels)
            test_prec.append(prec_score if isinstance(prec_score, (float, int)) else prec_score.item())
            test_rec.append(rec_score if isinstance(rec_score, (float, int)) else rec_score.item())
            test_f1.append(f1_score if isinstance(f1_score, (float, int)) else f1_score.item())
        
        test_prec = np.array(test_prec)
        test_rec = np.array(test_rec)
        test_f1 = np.array(test_f1)
        
        print(f"Threshold={threshold_value}: Prec={np.mean(test_prec):.4f}, Rec={np.mean(test_rec):.4f}, F1={np.mean(test_f1):.4f}")
        return np.mean(test_prec), np.mean(test_rec), np.mean(test_f1)

if __name__ == "__main__":
    # 测试不同的阈值
    thresholds = [1.96, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0]
    results = []
    
    for thresh in thresholds:
        prec, rec, f1 = test_with_threshold(thresh)
        results.append((thresh, prec, rec, f1))
    
    print("\n=== 阈值测试结果汇总 ===")
    for thresh, prec, rec, f1 in results:
        print(f"阈值={thresh}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
