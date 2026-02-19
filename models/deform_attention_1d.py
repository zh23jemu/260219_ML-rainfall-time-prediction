import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def trunc_normal_(tensor, mean=0., std=0.02):
    # PyTorch 自带 trunc_normal_（新版本），这里做兼容
    if hasattr(nn.init, "trunc_normal_"):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std)
    # fallback: normal
    return nn.init.normal_(tensor, mean=mean, std=std)

class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class DeformAtten1D(nn.Module):
    """
    1D Deformable Self-Attention：在时间轴学习偏移并重采样 K,V，
    用于对齐“周边->中心”的动态传播滞后。
    输入 x: [B, L, C]
    输出: [B, L, C]
    """
    def __init__(self, seq_len, d_model, n_heads=8, dropout=0.1, kernel=7, n_groups=4, no_off=False, rpb=True):
        super().__init__()
        assert d_model % n_heads == 0
        assert d_model % n_groups == 0
        assert n_heads % n_groups == 0

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.no_off = no_off
        self.offset_range_factor = kernel
        self.head_dim = d_model // n_heads
        self.group_dim = d_model // n_groups
        self.group_heads = n_heads // n_groups
        self.scale = self.head_dim ** -0.5
        self.rpb = rpb

        self.proj_q = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj_k = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj_v = nn.Conv1d(d_model, d_model, kernel_size=1)

        pad = kernel // 2
        self.proj_offset = nn.Sequential(
            nn.Conv1d(self.group_dim, self.group_dim, kernel_size=kernel, padding=pad),
            nn.Conv1d(self.group_dim, 1, kernel_size=1)
        )

        self.proj_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(torch.zeros(1, d_model, seq_len))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, attn_mask=None):
        B, L, C = x.shape
        assert L == self.seq_len, f"seq_len mismatch: got {L}, expect {self.seq_len}"

        # (B,L,C) -> (B,C,L)
        x_c = x.permute(0, 2, 1)

        q = self.proj_q(x_c)
        k = self.proj_k(x_c)
        v = self.proj_v(x_c)

        if self.rpb:
            v = v + self.relative_position_bias_table

        # group for offsets
        grouped_q = rearrange(q, "b (g c) l -> (b g) c l", g=self.n_groups)  # (Bg, Cg, L)
        offset = self.proj_offset(grouped_q)  # (Bg,1,L)

        if (self.offset_range_factor >= 0) and (not self.no_off):
            offset = offset.tanh() * self.offset_range_factor

        # build sampling grid: normalized to [-1,1]
        # base positions: 0..L-1
        pos = torch.arange(L, device=x.device, dtype=x.dtype).view(1, 1, L)  # (1,1,L)
        # add offset
        sample_pos = pos + offset  # (Bg,1,L)
        # normalize: [-1,1]
        sample_grid = 2.0 * (sample_pos / max(L - 1, 1)) - 1.0  # (Bg,1,L)
        sample_grid = sample_grid.permute(0, 2, 1).unsqueeze(2)  # (Bg,L,1,1)

        # sample k,v in grouped manner
        x_grouped = rearrange(x_c, "b (g c) l -> (b g) c l", g=self.n_groups)  # (Bg,Cg,L)
        # grid_sample expects 4D: (N,C,H,W). here treat (L,1) as (H,W)
        x_grouped_4d = x_grouped.unsqueeze(-1)  # (Bg,Cg,L,1)
        k_grouped = rearrange(k, "b (g c) l -> (b g) c l", g=self.n_groups).unsqueeze(-1)
        v_grouped = rearrange(v, "b (g c) l -> (b g) c l", g=self.n_groups).unsqueeze(-1)

        k_s = F.grid_sample(k_grouped, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(-1)  # (Bg,Cg,L)
        v_s = F.grid_sample(v_grouped, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(-1)

        # merge groups back
        k_s = rearrange(k_s, "(b g) c l -> b (g c) l", b=B, g=self.n_groups)
        v_s = rearrange(v_s, "(b g) c l -> b (g c) l", b=B, g=self.n_groups)

        # multi-head attention on time axis
        qh = rearrange(q, "b (h d) l -> (b h) l d", h=self.n_heads, d=self.head_dim)
        kh = rearrange(k_s, "b (h d) l -> (b h) l d", h=self.n_heads, d=self.head_dim)
        vh = rearrange(v_s, "b (h d) l -> (b h) l d", h=self.n_heads, d=self.head_dim)

        attn = torch.einsum("b i d, b j d -> b i j", qh, kh) * self.scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("b i j, b j d -> b i d", attn, vh)  # (Bh,L,D)
        out = rearrange(out, "(b h) l d -> b l (h d)", b=B, h=self.n_heads, d=self.head_dim)
        out = self.proj_out(out)
        return out

class DeformBlock(nn.Module):
    def __init__(self, seq_len, d_model, n_heads=8, n_groups=4, kernel=7, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = DeformAtten1D(seq_len=seq_len, d_model=d_model, n_heads=n_heads, dropout=dropout,
                                 kernel=kernel, n_groups=n_groups)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x
