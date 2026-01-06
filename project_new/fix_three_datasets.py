import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def analyze_dataset(dataset_name):
    """分析数据集的数据分布"""
    data_path = f'project_new/data/{dataset_name}/'
    
    if not os.path.exists(data_path + 'train.csv'):
        print(f"警告: {dataset_name}的训练文件不存在")
        return None, None, None, None
    
    try:
        train_data = pd.read_csv(data_path + 'train.csv')
        test_data = pd.read_csv(data_path + 'test.csv')
        
        print(f"\n=== {dataset_name} 数据集分析 ===")
        print(f"训练集形状: {train_data.shape}")
        
        if 'attack' in train_data.columns:
            train_counts = train_data['attack'].value_counts()
            print(f"训练集标签分布:")
            print(train_counts)
            train_anomaly_ratio = train_data['attack'].mean() * 100
            print(f"训练集异常比例: {train_anomaly_ratio:.2f}%")
        else:
            print("训练集中没有'attack'列")
            train_counts = None
            train_anomaly_ratio = None
        
        print(f"\n测试集形状: {test_data.shape}")
        if 'attack' in test_data.columns:
            test_counts = test_data['attack'].value_counts()
            print(f"测试集标签分布:")
            print(test_counts)
            test_anomaly_ratio = test_data['attack'].mean() * 100
            print(f"测试集异常比例: {test_anomaly_ratio:.2f}%")
        else:
            print("测试集中没有'attack'列")
            test_counts = None
            test_anomaly_ratio = None
        
        return train_data, test_data, train_anomaly_ratio, test_anomaly_ratio
        
    except Exception as e:
        print(f"分析{dataset_name}时出错: {e}")
        return None, None, None, None

def fix_dataset_split(dataset_name, test_size=0.3):
    """修复数据集划分，确保训练集中包含异常样本"""
    data_path = f'project_new/data/{dataset_name}/'
    
    # 分析原始数据
    train_data, test_data, train_ratio, test_ratio = analyze_dataset(dataset_name)
    
    if train_data is None or test_data is None:
        print(f"无法处理{dataset_name}数据集")
        return
    
    # 检查是否需要修复
    needs_fix = False
    if train_ratio == 0:
        print(f"⚠️ 问题: {dataset_name}训练集中没有异常样本!")
        needs_fix = True
    elif train_ratio == 100:
        print(f"⚠️ 问题: {dataset_name}训练集中全是异常样本!")
        needs_fix = True
    elif abs(train_ratio - test_ratio) > 20:  # 差异超过20%
        print(f"⚠️ 问题: {dataset_name}训练集和测试集异常比例差异过大!")
        needs_fix = True
    
    if not needs_fix:
        print(f"✓ {dataset_name}数据集划分正常，无需修复")
        return
    
    print(f"\n正在修复{dataset_name}数据集划分...")
    
    # 合并所有数据
    all_data = pd.concat([train_data, test_data], ignore_index=True)
    print(f"合并后总数据形状: {all_data.shape}")
    
    if 'attack' in all_data.columns:
        print(f"总数据标签分布:")
        print(all_data['attack'].value_counts())
        total_anomaly_ratio = all_data['attack'].mean() * 100
        print(f"总数据异常比例: {total_anomaly_ratio:.2f}%")
    
    # 准备特征和标签
    if 'Timestamp' in all_data.columns:
        X = all_data.drop(columns=['attack', 'Timestamp'])
        has_timestamp = True
    else:
        X = all_data.drop(columns=['attack'])
        has_timestamp = False
    
    y = all_data['attack']
    
    # 分层划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        stratify=y,  # 分层抽样，保持标签比例
        random_state=42
    )
    
    # 重新构建数据框
    if has_timestamp:
        # 使用原始时间戳或生成新的
        train_timestamps = all_data.loc[X_train.index, 'Timestamp'] if 'Timestamp' in all_data.columns else pd.Series(range(len(X_train)))
        test_timestamps = all_data.loc[X_test.index, 'Timestamp'] if 'Timestamp' in all_data.columns else pd.Series(range(len(X_test)))
        
        new_train_data = pd.concat([
            train_timestamps.reset_index(drop=True), 
            X_train.reset_index(drop=True), 
            y_train.reset_index(drop=True)
        ], axis=1)
        
        new_test_data = pd.concat([
            test_timestamps.reset_index(drop=True), 
            X_test.reset_index(drop=True), 
            y_test.reset_index(drop=True)
        ], axis=1)
        
        # 确保列顺序正确
        new_train_data = new_train_data[['Timestamp'] + list(X_train.columns) + ['attack']]
        new_test_data = new_test_data[['Timestamp'] + list(X_test.columns) + ['attack']]
    else:
        new_train_data = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        new_test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    
    # 创建备份目录
    backup_dir = data_path + 'backup/'
    os.makedirs(backup_dir, exist_ok=True)
    
    # 备份原始文件
    train_data.to_csv(backup_dir + f'train_original_{dataset_name}.csv', index=False)
    test_data.to_csv(backup_dir + f'test_original_{dataset_name}.csv', index=False)
    
    # 保存新文件
    new_train_data.to_csv(data_path + 'train.csv', index=False)
    new_test_data.to_csv(data_path + 'test.csv', index=False)
    
    print(f"\n✓ {dataset_name}数据集修复完成")
    print(f"原始文件已备份到: {backup_dir}")
    
    # 验证修复结果
    print(f"\n验证修复后的{dataset_name}数据集:")
    new_train_ratio = new_train_data['attack'].mean() * 100
    new_test_ratio = new_test_data['attack'].mean() * 100
    
    print(f"新训练集形状: {new_train_data.shape}")
    print(f"新训练集异常比例: {new_train_ratio:.2f}%")
    print(f"新测试集形状: {new_test_data.shape}")
    print(f"新测试集异常比例: {new_test_ratio:.2f}%")
    
    return new_train_data, new_test_data

def main():
    """主函数：处理三个数据集"""
    datasets = ['kddcup99', 'swat', 'wadi']
    
    print("="*60)
    print("开始处理三个数据集: kddcup99, swat, wadi")
    print("="*60)
    
    for dataset in datasets:
        print(f"\n{'='*40}")
        print(f"处理数据集: {dataset}")
        print('='*40)
        
        # 先分析
        train_data, test_data, train_ratio, test_ratio = analyze_dataset(dataset)
        
        if train_data is not None and test_data is not None:
            # 检查是否需要修复
            if train_ratio == 0 or train_ratio == 100 or abs(train_ratio - test_ratio) > 20:
                print(f"\n检测到问题，正在修复{dataset}...")
                fix_dataset_split(dataset)
            else:
                print(f"\n✓ {dataset}数据集正常，无需修复")
        else:
            print(f"\n✗ 无法处理{dataset}数据集")
    
    print("\n" + "="*60)
    print("所有数据集处理完成!")
    print("="*60)
    
    # 最终验证
    print("\n最终验证所有数据集:")
    for dataset in datasets:
        print(f"\n{dataset}最终状态:")
        analyze_dataset(dataset)

if __name__ == "__main__":
    main()
