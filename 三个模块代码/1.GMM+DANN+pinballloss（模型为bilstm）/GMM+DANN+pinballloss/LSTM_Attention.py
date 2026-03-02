import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.attention = ScaledDotProductAttention(embed_size, heads)

    def forward(self, values, keys, query, mask=None):
        return self.attention(values, keys, query, mask)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(ScaledDotProductAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.values_linear = nn.Linear(embed_size, embed_size)
        self.keys_linear = nn.Linear(embed_size, embed_size)
        self.queries_linear = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len = values.shape[1]
        key_len = keys.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, key_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=-1)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])

        out = out.reshape(N, query.shape[1], self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, heads=4):
        super(BiLSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        # 使用GRU而不是LSTM，因为GRU参数更少，训练更快
        # 注意：这里的input_size是1，因为我们将特征视为序列
        self.lstm = nn.GRU(
            input_size=1,  # 输入是单变量时间序列
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # 多头注意力层
        self.multihead_attention = MultiHeadAttention(embed_size=hidden_size, heads=heads)

        self.fc0 = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(hidden_size, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 确保数据类型是float32
        x = x.float()

        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # 多头注意力层
        attention_output = self.multihead_attention(lstm_out, lstm_out, lstm_out)

        # 取最后一个时间步的特征
        output = attention_output[:, -1, :]

        # 如果fc0不是Identity，则应用fc0
        if not isinstance(self.fc0, nn.Identity):
            output = self.fc0(output)

        return output


