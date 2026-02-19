import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# --------- Mamba import (optional) ----------
try:
    from mamba_ssm.modules.mamba_simple import Mamba as _Mamba
    MAMBA_AVAILABLE = True
except Exception:
    _Mamba = None
    MAMBA_AVAILABLE = False
    warnings.warn("mamba_ssm not found. Using GRU fallback. Install with: pip install mamba-ssm")

class _MambaFallback(nn.Module):
    """GRU fallback that matches Mamba input/output shape: (B,T,C)->(B,T,C)"""
    def __init__(self, d_model: int):
        super().__init__()
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        y, _ = self.gru(x)
        return y

class ExBiMamba(nn.Module):
    """外部双向 Mamba：正向 + 反向，拼接后线性投影回 d_model。"""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if MAMBA_AVAILABLE:
            self.forward_mamba = _Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.backward_mamba = _Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.forward_mamba = _MambaFallback(d_model=d_model)
            self.backward_mamba = _MambaFallback(d_model=d_model)

        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        y_f = self.forward_mamba(x)                       # (B,T,C)
        y_b = self.backward_mamba(torch.flip(x, dims=[1]))
        y_b = torch.flip(y_b, dims=[1])
        y = torch.cat([y_f, y_b], dim=-1)                 # (B,T,2C)
        return self.proj(y)                               # (B,T,C)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class ConformerConv(nn.Module):
    """轻量 Conformer-style conv module: (B,T,C)->(B,T,C)"""
    def __init__(self, d_model: int, kernel_size: int = 31, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        inner = d_model * expansion
        pad = kernel_size // 2
        self.pre_norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Conv1d(d_model, inner, kernel_size=1),
            nn.GLU(dim=1),                                 # halves channels
            nn.Conv1d(inner // 2, inner // 2, kernel_size=kernel_size, padding=pad, groups=inner // 2),
            nn.BatchNorm1d(inner // 2),
            nn.SiLU(),
            nn.Conv1d(inner // 2, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        # (B,T,C)->(B,C,T)->conv->(B,C,T)->(B,T,C)
        y = self.pre_norm(x).transpose(1, 2)
        y = self.net(y)
        return y.transpose(1, 2)

class ConBiMambaBlock(nn.Module):
    """
    结构：FFN(1/2) -> ExBiMamba -> Conv -> FFN(1/2) + LN
    """
    def __init__(self, d_model: int, dropout: float = 0.1, conv_kernel: int = 31):
        super().__init__()
        self.ff1 = FeedForward(d_model, expansion=4, dropout=dropout)
        self.mamba = ExBiMamba(d_model)
        self.conv = ConformerConv(d_model, kernel_size=conv_kernel, dropout=dropout)
        self.ff2 = FeedForward(d_model, expansion=4, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.mamba(self.norm(x))
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)

class ConBiMambaEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 3, dropout: float = 0.1, conv_kernel: int = 31):
        super().__init__()
        self.layers = nn.ModuleList([
            ConBiMambaBlock(d_model=d_model, dropout=dropout, conv_kernel=conv_kernel)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
