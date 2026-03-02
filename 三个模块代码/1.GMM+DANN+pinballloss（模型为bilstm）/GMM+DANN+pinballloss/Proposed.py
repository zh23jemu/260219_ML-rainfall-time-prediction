import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

# 忽略所有警告
warnings.filterwarnings("ignore")

from config import Config
from datapreprocess import relative_error
from LSTM_Attention import BiLSTMAttention
from datagenerate import prepDataloader, prepDataloader2, load_and_prepare_data
from loss.adv_loss import AdversarialLoss


class Pro(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, domain_num, heads=4, smooth=False):
        super().__init__()
        # 使用BiLSTM+多头注意力作为特征提取器
        self.feature_extractor = BiLSTMAttention(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            output_size=1,  # 这个参数在特征提取时不重要
            heads=heads
        )

        # 移除BiLSTMAttention的最后一层，只保留特征提取部分
        self.feature_extractor.fc0 = nn.Identity()

        # 添加MLP分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # 域对抗损失 - 注意这里input_dim要与hidden_dim匹配
        self.adv_loss = AdversarialLoss(domain_num=domain_num, smooth=smooth)
        # 修改domain_classifier的input_dim
        self.adv_loss.domain_classifier = torch.nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, domain_num),
            nn.BatchNorm1d(domain_num),
            nn.Softmax(dim=1)
        )
        self.smooth = smooth

    def forward(self, x):
        # 确保数据类型是float32
        x = x.float()

        # 调整输入形状为BiLSTM期望的形状 [batch_size, seq_len, input_size]
        # 我们将特征视为序列，每个时间步是一个特征
        x = x.view(x.size(0), x.size(1), 1)

        # 特征提取
        features = self.feature_extractor(x)

        # 预测
        pred = self.classifier(features)

        return features, pred

    def cal_loss(self, all_fc, s_preds, source_y, domain_labels, epoch_ratio=None):
        return RMSE_loss(s_preds, source_y) + \
            self.adv_loss(all_fc, domain_labels, epoch_ratio)


def RMSE_loss(labels, predicts):
    criterion = nn.MSELoss()
    return torch.sqrt(criterion(predicts, labels))


def get_domain_label(fcs1, fcs2, fcs3, s_y1, s_y2, s_y3, s_p1, s_p2, s_p3):
    domain_labels = torch.zeros([len(fcs1) + len(fcs2) + len(fcs3), 3]).to(fcs1.device)
    domain_labels[:len(fcs1), 0] = 1
    domain_labels[len(fcs1):(len(fcs1)+len(fcs2)), 1] = 1
    domain_labels[(len(fcs1)+len(fcs2)):, 2] = 1
    all_fc = torch.cat((fcs1, fcs2, fcs3), axis=0)
    all_preds = torch.cat((s_p1, s_p2, s_p3), axis=0)
    all_class_labels = torch.cat((s_y1, s_y2, s_y3), axis=0).reshape(-1, 1)
    return domain_labels, all_fc, all_preds, all_class_labels


def test(model, tt_set, device):
    """
    测试模型性能

    参数：
    - model: 模型
    - tt_set: 测试数据集
    - device: 设备

    返回：
    - rmse: 均方根误差
    - re: 相对误差
    """
    model.eval()
    predicts, targets = [], []
    for x, y in tt_set:
        x = x.to(device)
        targets.append(y)
        with torch.no_grad():
            _, predict = model(x)
            predicts.append(predict.detach().cpu())
    targets = torch.cat(targets, dim=0).numpy()
    preds = torch.cat(predicts, dim=0).numpy()
    rmse = np.sqrt(mean_squared_error(targets, preds))
    re = relative_error(y_test=targets, y_predict=preds, threshold=Config.Rated_Capacity * 0.7)
    return rmse, re


def train(tr_set, tt_set, model, device, model_save_dir=None):
    """
    训练模型

    参数：
    - tr_set: 训练数据集
    - tt_set: 测试数据集
    - model: 模型
    - device: 设备
    - model_save_dir: 模型保存目录
    """
    # 创建优化器
    optimizer = getattr(torch.optim, Config.optimizer)(model.parameters(), **Config.optim_hparas)

    # 初始化最佳RMSE和分数列表
    min_rmse = 1e10
    scores = []

    # 训练循环
    for epoch in range(Config.n_epoches):
        model.train()
        total_loss = 0
        batch_count = 0

        # 批次训练
        for x, y, domain_labels in tr_set:
            optimizer.zero_grad()
            x, y, domain_labels = x.to(device), y.to(device), domain_labels.to(device)

            # 前向传播
            fs, preds = model(x)

            # 计算损失
            loss = model.cal_loss(fs, preds, y, domain_labels, epoch / Config.n_epoches)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        # 测试模型
        rmse, rul_score = test(model, tt_set, device)
        scores.append(rul_score)

        # 保存检查点
        check_point = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # 如果当前模型是最佳模型，则保存
        if rmse < min_rmse:
            min_rmse = rmse
            if model_save_dir is not None:
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                torch.save(check_point, os.path.join(model_save_dir, 'LSTM_Attention_DANN.pth'))

        # 打印训练信息
        print('Epoch [{}/{}], Loss: {:.4f}, Test RMSE: {:.4f}, Test RE: {:.4f}, Min RMSE: {:.4f}'.
              format(epoch + 1, Config.n_epoches, total_loss / batch_count, rmse, rul_score, min_rmse))

    print(f'最小相对误差: {np.array(scores).min():.4f}')





if __name__ == '__main__':
    # 确保目录存在
    Config.ensure_directories()

    # 加载并准备数据
    train_data, train_label, test_x0, test_y0, domain_labels = load_and_prepare_data()

    # 将域标签转换为one-hot编码
    n_components = 3  # 我们有3个域
    domain_labels_onehot = np.zeros((len(domain_labels), n_components))
    for i in range(len(domain_labels)):
        domain_labels_onehot[i, int(domain_labels[i])] = 1

    # 准备数据加载器
    tr_dataset = prepDataloader2(train_data, train_label, domain_labels_onehot, 'train', Config.batch_size*3)
    tt_dataset = prepDataloader(test_x0, test_y0, 'test', Config.batch_size)

    # 创建模型
    model = Pro(
        input_size=Config.feature_size,
        hidden_dim=Config.hidden_dim,
        num_layers=Config.num_layers,
        domain_num=n_components,
        heads=Config.heads,
        smooth=False
    ).to(Config.device)

    print("模型结构:")
    print(model)

    # 训练模型
    train(tr_dataset, tt_dataset, model, Config.device, Config.model_save_dir)

    print("训练完成!")
