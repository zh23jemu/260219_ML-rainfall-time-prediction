import torch
import os

class Config:
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型参数
    hidden_dim = 64
    num_layers = 2
    feature_size = 40
    heads = 4

    # 训练参数
    batch_size = 32
    n_epoches = 5  # 减少训练轮数以加快测试
    optimizer = 'Adam'
    optim_hparas = {'lr': 0.001, 'weight_decay': 1e-5}

    # 数据参数
    Rated_Capacity = 1.0
    seed = 1000

    # 路径配置
    data_dir = './data'
    model_save_dir = './models'

    # 确保目录存在
    @staticmethod
    def ensure_directories():
        if not os.path.exists(Config.data_dir):
            os.makedirs(Config.data_dir)
        if not os.path.exists(Config.model_save_dir):
            os.makedirs(Config.model_save_dir)

    # 电池列表
    battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
