import torch
import numpy as np
import pandas as pd
from model import Model
from process import TimeDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data import WeightedRandomSampler
import argparse
import matplotlib.pyplot as plt

# 数据集特定阈值配置
DATASET_THRESHOLDS = {
    'kddcup99': 2.1576,  # 根据理论最佳阈值
    'swat': 0.0,         # 根据实际分析调整（理论最佳阈值-0.0404，但假阳性率太高）
    'wadi': 1.8          # 需要根据实际分析调整
}

def analyze_score_distribution(dataset_name='kddcup99', threshold_value=None):
    # 如果未提供阈值，使用数据集特定的阈值
    if threshold_value is None:
        if dataset_name in DATASET_THRESHOLDS:
            threshold_value = DATASET_THRESHOLDS[dataset_name]
            print(f"使用数据集 '{dataset_name}' 的默认阈值: {threshold_value}")
        else:
            threshold_value = -1.0
            print(f"数据集 '{dataset_name}' 未配置阈值，使用默认值: {threshold_value}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-slide_win', type=int, default=5)
    parser.add_argument('-slide_stride', type=int, default=5)
    parser.add_argument('-dataset', type=str, default=dataset_name)
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
    parser.add_argument('-threshold', type=float, default=threshold_value)
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
    parser.add_argument('-test', type=bool, default=True)
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
        'epoches': 1, 'device': torch.device('cpu'), 'dev_hidden_dim1': args.dev_hidden_dim1,
        'dev_hidden_dim2': args.dev_hidden_dim2, 'max_iter': args.max_iter,
        'node_hidden_dim': args.node_hidden_dim, 'graph_hops': args.graph_hops,
        'num_pers': args.num_pers, 'margin': args.margin, 'threshold': args.threshold,
        'lr': args.lr, 'patience': args.patience, 'bias': args.bias, 'batch_norm': args.batch_norm,
        'test': args.test, 'dropout': args.dropout, 'MI': args.MI, 'iterative_flag': args.iterative_flag
    }

    # 加载数据
    f = open('data/{}/list.txt'.format(config['dataset']))
    sensors = f.readlines()
    f.close()
    config['number of sensors'] = len(sensors)

    data = pd.read_csv('data/{}/test.csv'.format(config['dataset']))
    train_data = pd.read_csv('data/{}/train.csv'.format(config['dataset']))
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
    
    # 加载预训练权重
    try:
        model.load_state_dict(torch.load('checkpoint.pt', map_location=config['device']))
        print(f"Loaded pretrained model for analysis")
    except:
        print(f"No pretrained model found")
        return

    # 分析得分分布
    config['test'] = True
    with torch.no_grad():
        normal_scores = []
        anomaly_scores = []
        normal_deviations = []
        anomaly_deviations = []
        
        for i, d in enumerate(test_loader):
            x = d[0].to(config['device'])
            labels = d[1].to(config['device'])
            
            # 获取模型输出
            score, loss_MI, loss_clf, prec_score, rec_score, f1_score = model(x, labels)
            
            # 计算偏差（根据measure_net.py中的逻辑）
            ref = torch.normal(mean=0., std=torch.full([5000], 1.)).to(config['device'])
            deviation = (score - torch.mean(ref)) / torch.std(ref)
            
            # 分离正常和异常样本的得分
            for j in range(len(labels)):
                if labels[j] == 0:  # 正常样本
                    normal_scores.append(score[j].item())
                    normal_deviations.append(deviation[j].item())
                else:  # 异常样本
                    anomaly_scores.append(score[j].item())
                    anomaly_deviations.append(deviation[j].item())
        
        # 转换为numpy数组
        normal_scores = np.array(normal_scores)
        anomaly_scores = np.array(anomaly_scores)
        normal_deviations = np.array(normal_deviations)
        anomaly_deviations = np.array(anomaly_deviations)
        
        # 打印统计信息
        print("=== 正常样本和异常样本得分分析 ===")
        print(f"正常样本数量: {len(normal_scores)}")
        print(f"异常样本数量: {len(anomaly_scores)}")
        print("\n--- 原始得分 (score) ---")
        print(f"正常样本平均得分: {np.mean(normal_scores):.6f}")
        print(f"异常样本平均得分: {np.mean(anomaly_scores):.6f}")
        print(f"得分差异 (异常-正常): {np.mean(anomaly_scores) - np.mean(normal_scores):.6f}")
        print(f"正常样本得分标准差: {np.std(normal_scores):.6f}")
        print(f"异常样本得分标准差: {np.std(anomaly_scores):.6f}")
        
        print("\n--- 标准化偏差 (deviation) ---")
        print(f"正常样本平均偏差: {np.mean(normal_deviations):.6f}")
        print(f"异常样本平均偏差: {np.mean(anomaly_deviations):.6f}")
        print(f"偏差差异 (异常-正常): {np.mean(anomaly_deviations) - np.mean(normal_deviations):.6f}")
        print(f"正常样本偏差标准差: {np.std(normal_deviations):.6f}")
        print(f"异常样本偏差标准差: {np.std(anomaly_deviations):.6f}")
        
        # 分析阈值效果
        print(f"\n--- 当前阈值效果分析 (阈值={threshold_value}) ---")
        # 计算基于当前阈值的预测
        normal_pred_anomaly = np.sum(normal_deviations > threshold_value)
        anomaly_pred_anomaly = np.sum(anomaly_deviations > threshold_value)
        
        print(f"正常样本中被预测为异常的比例: {normal_pred_anomaly/len(normal_deviations)*100:.2f}% ({normal_pred_anomaly}/{len(normal_deviations)})")
        print(f"异常样本中被预测为异常的比例: {anomaly_pred_anomaly/len(anomaly_deviations)*100:.2f}% ({anomaly_pred_anomaly}/{len(anomaly_deviations)})")
        
        # 计算理论上的最佳阈值（基于偏差分布）
        print("\n--- 理论最佳阈值分析 ---")
        # 使用ROC曲线思想：找到使Youden指数最大的阈值
        all_deviations = np.concatenate([normal_deviations, anomaly_deviations])
        all_labels = np.concatenate([np.zeros_like(normal_deviations), np.ones_like(anomaly_deviations)])
        
        thresholds = np.linspace(np.min(all_deviations)-1, np.max(all_deviations)+1, 100)
        youden_indices = []
        
        for thresh in thresholds:
            preds = (all_deviations > thresh).astype(int)
            tp = np.sum((preds == 1) & (all_labels == 1))
            tn = np.sum((preds == 0) & (all_labels == 0))
            fp = np.sum((preds == 1) & (all_labels == 0))
            fn = np.sum((preds == 0) & (all_labels == 1))
            
            if tp + fn > 0 and tn + fp > 0:
                tpr = tp / (tp + fn)  # 召回率
                fpr = fp / (fp + tn)  # 假阳性率
                youden = tpr - fpr
                youden_indices.append((thresh, youden, tpr, fpr))
        
        if youden_indices:
            youden_indices.sort(key=lambda x: x[1], reverse=True)
            best_thresh, best_youden, best_tpr, best_fpr = youden_indices[0]
            print(f"理论最佳阈值: {best_thresh:.4f}")
            print(f"对应Youden指数: {best_youden:.4f}")
            print(f"理论召回率: {best_tpr:.4f}")
            print(f"理论假阳性率: {best_fpr:.4f}")
        
        # 绘制得分分布直方图
        try:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(normal_scores, bins=50, alpha=0.5, label='正常样本', color='blue')
            plt.hist(anomaly_scores, bins=50, alpha=0.5, label='异常样本', color='red')
            plt.xlabel('原始得分 (score)')
            plt.ylabel('频数')
            plt.title('正常vs异常样本原始得分分布')
            plt.legend()
            plt.axvline(x=np.mean(normal_scores), color='blue', linestyle='--', alpha=0.7)
            plt.axvline(x=np.mean(anomaly_scores), color='red', linestyle='--', alpha=0.7)
            
            plt.subplot(1, 2, 2)
            plt.hist(normal_deviations, bins=50, alpha=0.5, label='正常样本', color='blue')
            plt.hist(anomaly_deviations, bins=50, alpha=0.5, label='异常样本', color='red')
            plt.xlabel('标准化偏差 (deviation)')
            plt.ylabel('频数')
            plt.title('正常vs异常样本标准化偏差分布')
            plt.legend()
            plt.axvline(x=threshold_value, color='green', linestyle='-', label=f'当前阈值={threshold_value}')
            plt.axvline(x=np.mean(normal_deviations), color='blue', linestyle='--', alpha=0.7)
            plt.axvline(x=np.mean(anomaly_deviations), color='red', linestyle='--', alpha=0.7)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('score_distribution.png', dpi=150)
            print(f"\n得分分布图已保存为: score_distribution.png")
            
        except Exception as e:
            print(f"\n无法生成图表: {e}")

if __name__ == "__main__":
    # 使用当前阈值进行分析
    analyze_score_distribution(threshold_value=-1.0)
