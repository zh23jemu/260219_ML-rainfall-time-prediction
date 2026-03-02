import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, Dataset
from config import Config
from datapreprocess import load_battery_data, splitDataset


class TSDataset(Dataset):
    """
    时间序列数据集
    """
    def __init__(self, data, y):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class TSDomainDataset(Dataset):
    """
    带域标签的时间序列数据集
    """
    def __init__(self, data, y, domain_labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(y, dtype=torch.float32)
        self.domain_labels = torch.tensor(domain_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.domain_labels[index]


def prepDataloader(data, label, mode, batch_size, n_jobs=0):
    """
    准备数据加载器

    参数：
    - data: 输入数据
    - label: 标签数据
    - mode: 模式 ('train' 或 'test')
    - batch_size: 批量大小
    - n_jobs: 工作进程数

    返回：
    - DataLoader 对象
    """
    dataset = TSDataset(data, label)
    return DataLoader(dataset, batch_size, shuffle=(mode == 'train'), num_workers=n_jobs, pin_memory=True)


def prepDataloader2(data, label, domain_labels, mode, batch_size, n_jobs=0):
    """
    准备带域标签的数据加载器

    参数：
    - data: 输入数据
    - label: 标签数据
    - domain_labels: 域标签
    - mode: 模式 ('train' 或 'test')
    - batch_size: 批量大小
    - n_jobs: 工作进程数

    返回：
    - DataLoader 对象
    """
    dataset = TSDomainDataset(data, label, domain_labels)
    return DataLoader(dataset, batch_size, shuffle=(mode == 'train'), num_workers=n_jobs, pin_memory=True)


def load_and_prepare_data():
    """
    加载并准备数据

    返回：
    - train_data: 训练数据
    - train_label: 训练标签
    - test_data: 测试数据
    - test_label: 测试标签
    - domain_labels: 域标签
    """
    # 确保数据目录存在
    Config.ensure_directories()

    # 检查数据文件是否存在，如果不存在则生成
    data_file = os.path.join(Config.data_dir, 'data.csv')
    if not os.path.exists(data_file):
        from data_generator import save_data
        save_data(Config.data_dir)

    # 加载电池数据
    train_x0, train_y0 = load_battery_data(Config.battery_list[0], Config.feature_size)
    train_x1, train_y1 = load_battery_data(Config.battery_list[1], Config.feature_size)
    train_x2, train_y2 = load_battery_data(Config.battery_list[2], Config.feature_size)
    test_x0, test_y0 = load_battery_data(Config.battery_list[3], Config.feature_size)

    # 合并训练数据
    train_data = np.concatenate((train_x0, train_x1, train_x2), axis=0)
    train_label = np.concatenate((train_y0, train_y1, train_y2), axis=0)

    # 创建域标签
    domain_labels = np.zeros(len(train_data))
    domain_labels[:len(train_x0)] = 0
    domain_labels[len(train_x0):len(train_x0)+len(train_x1)] = 1
    domain_labels[len(train_x0)+len(train_x1):] = 2

    return train_data, train_label, test_x0, test_y0, domain_labels


if __name__ == '__main__':
    data_path = r'data.csv'
    data = pd.read_csv(data_path,encoding="utf-8")
    train_data, test_data, train_y, test_y = splitDataset(data, 40, 10, 1, 0.3)
    tr_set = prepDataloader(train_data, train_y, 'train', 128)
    tt_set = prepDataloader(test_data, test_y, 'test', 128)
    for i, (x, y) in enumerate(tr_set):
        # print(x.shape)
        x = x.view(-1, 40, 1)
        y = y.view(-1, 10, 1)
        print(x.shape, y.shape)
    pass