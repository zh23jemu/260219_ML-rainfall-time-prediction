# LSTM+多头注意力机制与DANN网络结合项目说明

## 项目修改说明

本项目在原有LSTM+多头注意力机制的基础上，集成了DANN（Domain-Adversarial Neural Network）网络进行迁移学习，以提高模型在不同域之间的泛化能力。

## 项目脚本功能说明

项目中的各个脚本功能如下：

1. **LSTM_Attention.py**: 实现了BiLSTM和多头注意力机制的模型
2. **datapreprocess.py**: 数据预处理相关函数，包括数据分割、特征提取等
3. **datagenerate.py**: 数据生成和加载函数，包括数据集类和数据加载器
4. **config.py**: 配置参数文件，包含模型参数、训练参数等
5. **Proposed.py**: 原始的模型实现和训练脚本，主要修改的文件
6. **LSTM_Attention_DANN.py**: 新增的文件，包含LSTM+多头注意力机制与DANN网络的完整实现
7. **loss/adv_loss.py**: 实现域对抗损失函数
8. **loss/coral.py**: 实现CORAL（CORrelation ALignment）损失函数，用于域适应

## 主要修改的脚本和关键代码

主要修改的脚本是 **Proposed.py**，关键修改部分如下：

### 1. 特征提取器修改

将原有的简单LSTM特征提取器替换为BiLSTM+多头注意力机制：

```python
# 原始代码
self.FE = Net(hidden_dim=hidden_dim, num_layers=num_layers, mode=mode)

# 修改后的代码
self.feature_extractor = BiLSTMAttention(
    input_size=input_size,
    hidden_size=hidden_dim,
    num_layers=num_layers,
    output_size=1,  # 这个参数在特征提取时不重要
    heads=heads
)

# 移除BiLSTMAttention的最后一层，只保留特征提取部分
self.feature_extractor.fc0 = nn.Identity()
```

### 2. 分类器修改

将简单的线性分类器替换为多层感知机（MLP）网络：

```python
# 原始代码
self.linear = nn.Linear(hidden_dim, 1)

# 修改后的代码
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
```

### 3. 前向传播修改

修改前向传播过程，适应新的特征提取器和分类器：

```python
# 原始代码
def forward(self, x):
    out, _ = self.FE.cell(x.unsqueeze(2))  # out shape: (batch_size, feature_size, hidden_dim)
    fs = out[:, -1, :]
    pred = self.FE.linear(fs)
    return fs, pred

# 修改后的代码
def forward(self, x):
    # 调整输入形状为BiLSTM期望的形状
    x = x.view(x.size(0), 1, -1)

    # 特征提取
    features = self.feature_extractor(x)

    # 预测
    pred = self.classifier(features)

    return features, pred
```

### 4. 域对抗损失修改

调整域对抗损失的输入维度，以匹配特征提取器的输出维度：

```python
# 原始代码
self.adv_loss = AdversarialLoss(domain_num=domain_num, smooth=smooth)

# 修改后的代码
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
```

### 5. 新增LSTM_Attention_DANN.py文件

创建了一个新的文件`LSTM_Attention_DANN.py`，包含了LSTM+多头注意力机制与DANN网络的完整实现。关键部分包括：

```python
# LSTM+多头注意力+DANN模型
class LSTM_Attention_DANN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, domain_num, heads=4, dropout=0.25):
        super(LSTM_Attention_DANN, self).__init__()

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=1,
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
```

### 主要修改内容

1. **特征提取器升级**：
   - 将原有的单向LSTM替换为双向LSTM，增强了对序列前后文信息的捕捉能力
   - 保留并优化了多头注意力机制，使模型能够关注序列中的重要部分
   - 在`LSTM_Attention_DANN.py`中实现了完整的特征提取器

2. **分类器改进**：
   - 将简单的线性分类器替换为多层感知机（MLP）网络
   - 添加了批归一化和Dropout层，提高模型的稳定性和泛化能力
   - 分类器结构为：`hidden_size -> 64 -> 32 -> 1`

3. **域对抗训练集成**：
   - 添加了梯度反转层（Gradient Reversal Layer）
   - 实现了域分类器，用于区分不同域的特征
   - 设计了域对抗损失函数，促使模型学习域不变的特征表示

4. **自动域发现**：
   - 使用高斯混合模型（GMM）自动发现数据中的潜在域
   - 基于贝叶斯信息准则（BIC）选择最优的域数量

5. **训练和评估流程优化**：
   - 实现了完整的训练函数，包括损失计算、模型保存等功能
   - 提供了评估函数，用于计算RMSE和MAE等指标

### 修改原因与好处

1. **为什么使用双向LSTM**：
   - 双向LSTM可以同时考虑序列的前向和后向信息，捕捉更全面的时序依赖关系
   - 相比单向LSTM，双向LSTM通常能提供更丰富的特征表示

2. **为什么使用多头注意力机制**：
   - 多头注意力允许模型同时关注序列中的不同位置和不同表示子空间
   - 增强了模型捕捉长距离依赖关系的能力
   - 提高了模型对重要特征的识别能力

3. **为什么集成DANN网络**：
   - DANN通过对抗训练学习域不变的特征表示，减少源域和目标域之间的分布差异
   - 提高了模型在不同域之间的泛化能力，使模型能够更好地适应新的数据分布
   - 减轻了迁移学习中的域偏移问题

4. **为什么使用MLP分类器**：
   - MLP比简单的线性分类器具有更强的表示能力，可以学习更复杂的特征映射
   - 批归一化和Dropout层有助于提高模型的稳定性和泛化能力
   - 多层结构允许模型学习更抽象的特征表示

5. **为什么使用GMM进行域发现**：
   - 自动发现数据中的潜在域，无需手动标注
   - 基于数据分布的特性进行聚类，更符合实际情况
   - 自适应选择最优的域数量，提高模型的灵活性

### 性能提升

通过上述修改，模型在以下方面获得了性能提升：

1. **预测精度提高**：双向LSTM和多头注意力机制的结合提高了模型的特征提取能力，从而提高了预测精度。

2. **泛化能力增强**：DANN网络的集成使模型能够学习域不变的特征表示，提高了模型在不同域之间的泛化能力。

3. **稳定性改进**：批归一化和Dropout层的添加提高了模型的稳定性，减少了过拟合的风险。

4. **适应性增强**：自动域发现功能使模型能够自适应地处理不同的数据分布，提高了模型的适应性。

## 使用指南

### 环境要求

- Python 3.6+
- PyTorch 1.7+
- NumPy
- scikit-learn
- pandas
- matplotlib
- tqdm

可以使用以下命令安装所需依赖：
```bash
pip install -r requirements.txt
```

### 文件结构

- `LSTM_Attention.py`: 实现了BiLSTM和多头注意力机制的模型
- `datapreprocess.py`: 数据预处理函数
- `datagenerate.py`: 数据生成和加载函数
- `config.py`: 配置参数
- `Proposed.py`: 主要的模型实现和训练脚本
- `LSTM_Attention_DANN.py`: 包含LSTM+多头注意力机制与DANN网络的完整实现
- `loss/adv_loss.py`: 实现域对抗损失函数
- `loss/coral.py`: 实现CORAL损失函数
- `requirements.txt`: 项目依赖列表

### 脚本使用说明和顺序

#### 1. 数据生成 (data_generator.py)

首先，运行数据生成脚本来创建示例数据：

```bash
# 运行数据生成脚本
python data_generator.py
```

这将在`./data`目录下生成以下文件：
- `data.csv`: 基础数据
- `B0005.csv`, `B0006.csv`, `B0007.csv`, `B0018.csv`: 电池数据

#### 2. 配置参数 (config.py)

在运行模型前，您可以修改`config.py`文件中的参数来调整模型行为：

```python
# 主要配置参数
hidden_dim = 64      # LSTM隐藏层大小
num_layers = 2       # LSTM层数
feature_size = 40    # 特征大小
heads = 4            # 注意力头数
batch_size = 32      # 批量大小
n_epoches = 100      # 训练轮数
```

#### 3. 运行完整模型 (Proposed.py)

这是主要的训练脚本，集成了LSTM+多头注意力机制与DANN网络：

```bash
# 运行完整模型训练
python Proposed.py
```

这个脚本会自动：
1. 检查并创建必要的目录
2. 如果数据不存在，则生成数据
3. 加载并准备数据
4. 创建并训练模型
5. 保存最佳模型到`./models`目录

#### 4. 查看训练结果

训练完成后，您可以在控制台输出中查看训练过程和最终结果：
- 每个epoch的损失值
- 测试RMSE（均方根误差）
- 测试RE（相对误差）
- 最小RMSE值

最佳模型将保存在`./models/LSTM_Attention_DANN.pth`。

### 使用步骤

1. **数据准备**：
   ```python
   from datapreprocess import splitDataset
   from datagenerate import prepDataloader, prepDataloader2, get_train_test

   # 加载数据
   train_x0, train_y0 = get_train_test('data.csv', 'battery1', 40)
   train_x1, train_y1 = get_train_test('data.csv', 'battery2', 40)
   train_x2, train_y2 = get_train_test('data.csv', 'battery3', 40)
   test_x0, test_y0 = get_train_test('data.csv', 'battery4', 40)

   # 合并训练数据
   train_data = np.concatenate((train_x0, train_x1, train_x2), axis=0)
   train_label = np.concatenate((train_y0, train_y1, train_y2), axis=0)
   ```

2. **域发现**：
   ```python
   from LSTM_Attention_DANN import discover_domains

   # 发现潜在域
   domain_labels, n_components = discover_domains(train_data, min_components=3, max_components=10)

   # 准备数据加载器
   tr_dataset = prepDataloader2(train_data, train_label, domain_labels, 'train', batch_size*3)
   tt_dataset = prepDataloader(test_x0, test_y0, 'test', batch_size)
   ```

3. **模型创建**：
   ```python
   from LSTM_Attention_DANN import LSTM_Attention_DANN
   from config import Config

   # 创建模型
   model = LSTM_Attention_DANN(
       input_size=1,
       hidden_size=Config.hidden_dim,
       num_layers=Config.num_layers,
       domain_num=n_components,
       heads=Config.heads
   ).to(Config.device)
   ```

4. **模型训练**：
   ```python
   from LSTM_Attention_DANN import train_model

   # 定义优化器
   optimizer = getattr(torch.optim, Config.optimizer)(model.parameters(), **Config.optim_hparas)

   # 训练模型
   model = train_model(
       model=model,
       train_loader=tr_dataset,
       test_loader=tt_dataset,
       optimizer=optimizer,
       device=Config.device,
       num_epochs=Config.n_epoches,
       model_save_dir=Config.model_save_dir
   )
   ```

5. **模型评估**：
   ```python
   from LSTM_Attention_DANN import evaluate_model

   # 评估模型
   rmse, mae = evaluate_model(model, tt_dataset, Config.device)
   print(f"Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}")
   ```

### 参数调整

您可以通过修改`config.py`文件中的参数来调整模型的行为：

- `hidden_dim`: LSTM隐藏层大小
- `num_layers`: LSTM层数
- `heads`: 注意力头数
- `batch_size`: 批量大小
- `n_epoches`: 训练轮数
- `optimizer`: 优化器类型
- `optim_hparas`: 优化器超参数

### 高级用法

1. **使用预训练模型**：
   ```python
   model = LSTM_Attention_DANN(...)
   model.load_state_dict(torch.load('models/best_model.pth'))
   model.eval()
   ```

2. **特征提取**：
   ```python
   features, _ = model(x)
   # 使用features进行其他任务，如可视化、聚类等
   ```

3. **自定义域标签**：
   如果您已经有了域标签，可以直接使用它们而不是通过GMM发现：
   ```python
   tr_dataset = prepDataloader2(train_data, train_label, your_domain_labels, 'train', batch_size)
   ```

## 结论

通过将LSTM+多头注意力机制与DANN网络结合，我们创建了一个强大的迁移学习模型，能够在不同域之间实现高效的知识迁移。这种结合利用了LSTM的序列建模能力、多头注意力的特征关注能力以及DANN的域适应能力，为时间序列预测任务提供了一个全面的解决方案。

希望这个项目能够满足您的需求，如有任何问题，请随时提出。
