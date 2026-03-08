import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from sklearn.metrics import mean_squared_error
from sklearn.mixture import GaussianMixture

# 忽略所有警告
warnings.filterwarnings("ignore")

# 反向层函数，用于梯度反转
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# Lambda调度器，用于调整梯度反转的强度
class LambdaSheduler(nn.Module):
    def __init__(self, gamma=10.0, max_iter=1000, **kwargs):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb

    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # 确保embed_size可以被heads整除
        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"
        
        self.values_linear = nn.Linear(embed_size, embed_size)
        self.keys_linear = nn.Linear(embed_size, embed_size)
        self.queries_linear = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # 线性变换
        values = self.values_linear(values)
        keys = self.keys_linear(keys)
        queries = self.queries_linear(query)
        
        # 分割成多头
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        
        # 计算注意力得分
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        # 应用掩码（如果有）
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # 注意力权重
        attention = torch.softmax(energy / (self.head_dim ** (1/2)), dim=3)
        
        # 加权求和
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        # 最终线性层
        out = self.fc_out(out)
        return out

# 域分类器
class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, domain_num):
        super(DomainClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, domain_num),
            nn.BatchNorm1d(domain_num),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.layers(x)

# 域对抗损失
class AdversarialLoss(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, domain_num=3, gamma=1.0, max_iter=1000, use_lambda_scheduler=True):
        super(AdversarialLoss, self).__init__()
        self.domain_classifier = DomainClassifier(input_dim, hidden_dim, domain_num)
        self.domain_num = domain_num
        
        self.use_lambda_scheduler = use_lambda_scheduler
        if self.use_lambda_scheduler:
            self.lambda_scheduler = LambdaSheduler(gamma, max_iter)
        self.lambd = 1.0
    
    def forward(self, features, domain_labels, epoch_ratio=None):
        # 梯度反转
        reversed_features = ReverseLayerF.apply(features, self.lambd)
        
        # 域分类
        domain_preds = self.domain_classifier(reversed_features)
        
        # 计算损失
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(domain_preds, domain_labels.float())
        
        return loss

# LSTM+多头注意力+DANN模型
class LSTM_Attention_DANN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, domain_num, heads=4, dropout=0.25):
        super(LSTM_Attention_DANN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=1,  # 输入是单变量时间序列
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 双向LSTM的输出维度是hidden_size*2
        self.bidirectional_size = hidden_size * 2
        
        # 多头注意力层
        self.attention = MultiHeadAttention(
            embed_size=self.bidirectional_size,
            heads=heads
        )
        
        # 标签分类器（MLP）
        self.label_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bidirectional_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 域对抗损失
        self.adv_loss = AdversarialLoss(
            input_dim=self.bidirectional_size,
            hidden_dim=32,
            domain_num=domain_num
        )
    
    def forward(self, x):
        # 确保输入形状正确 [batch_size, seq_len, 1]
        if x.dim() == 2:
            x = x.unsqueeze(2)
        
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 多头注意力层
        attention_out = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 取最后一个时间步的特征
        features = attention_out[:, -1, :]
        
        # 标签预测
        predictions = self.label_classifier(features)
        
        return features, predictions
    
    def cal_loss(self, features, predictions, targets, domain_labels, epoch_ratio=None):
        # 标签预测损失（MSE）
        mse_loss = nn.MSELoss()(predictions, targets)
        
        # 域对抗损失
        domain_loss = self.adv_loss(features, domain_labels, epoch_ratio)
        
        # 总损失
        total_loss = mse_loss + domain_loss
        
        return total_loss, mse_loss, domain_loss

# 训练函数
def train_model(model, train_loader, test_loader, optimizer, device, num_epochs=100, model_save_dir=None):
    best_rmse = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_mse = 0
        total_domain = 0
        batch_count = 0
        
        for x, y, domain_labels in train_loader:
            x, y, domain_labels = x.to(device), y.to(device), domain_labels.to(device)
            
            # 前向传播
            features, predictions = model(x)
            
            # 计算损失
            loss, mse_loss, domain_loss = model.cal_loss(
                features, predictions, y, domain_labels, epoch / num_epochs
            )
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_domain += domain_loss.item()
            batch_count += 1
        
        # 测试
        test_rmse, test_mae = evaluate_model(model, test_loader, device)
        
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {total_loss/batch_count:.4f}, '
              f'MSE: {total_mse/batch_count:.4f}, '
              f'Domain: {total_domain/batch_count:.4f}, '
              f'Test RMSE: {test_rmse:.4f}, '
              f'Test MAE: {test_mae:.4f}')
        
        # 保存最佳模型
        if test_rmse < best_rmse and model_save_dir is not None:
            best_rmse = test_rmse
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_model.pth'))
            print(f'Model saved with RMSE: {best_rmse:.4f}')
    
    return model

# 评估函数
def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            _, pred = model(x)
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = np.mean(np.abs(targets - predictions))
    
    return rmse, mae

# 使用GMM进行域发现
def discover_domains(data, min_components=3, max_components=10, random_state=42):
    best_bic = float('inf')
    best_gmm = None
    best_n_components = 0
    
    for n_components in range(min_components, max_components + 1):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=random_state
        )
        gmm.fit(data)
        bic = gmm.bic(data)
        
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n_components = n_components
    
    domain_labels = best_gmm.predict(data)
    print(f"Best number of domains: {best_n_components}")
    
    return domain_labels, best_n_components
