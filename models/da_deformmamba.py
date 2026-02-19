import math
import torch
import torch.nn as nn
from .deform_attention_1d import DeformBlock
from .conbimamba import ConBiMambaEncoder
from .grad_reverse import grad_reverse

class DADeformMamba(nn.Module):
    """
    输入 x: [B, L, V]
    输出：
      pred:
        - point: [B, H]
        - quantile: [B, H, Q]
      domain_logits: [B, Kd]
      feat: [B, d_model]
    """
    def __init__(self, V_in: int, L: int, H: int, num_domains: int,
                 d_model: int = 128,
                 deform_layers: int = 2, deform_heads: int = 8, deform_groups: int = 4, deform_kernel: int = 7,
                 conbimamba_layers: int = 3,
                 dropout: float = 0.1,
                 task_mode: str = "quantile",
                 quantiles = (0.1, 0.5, 0.9),
                 domain_hidden: int = 64):
        super().__init__()
        self.L = L
        self.H = H
        self.V_in = V_in
        self.d_model = d_model
        self.num_domains = num_domains
        self.task_mode = task_mode
        self.quantiles = list(quantiles)

        # 空间变量融合：每个时间步把 V 维投影到 d_model
        self.in_proj = nn.Linear(V_in, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, L, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.drop = nn.Dropout(dropout)

        # Deform encoder（时间对齐）
        self.deform = nn.ModuleList([
            DeformBlock(seq_len=L, d_model=d_model, n_heads=deform_heads, n_groups=deform_groups,
                       kernel=deform_kernel, dropout=dropout)
            for _ in range(deform_layers)
        ])

        # Mamba encoder（长依赖编码）
        self.mamba = ConBiMambaEncoder(d_model=d_model, num_layers=conbimamba_layers, dropout=dropout, conv_kernel=31)

        # pooling：取最后时间步（也可改成 mean pooling）
        self.norm_out = nn.LayerNorm(d_model)

        # prediction head
        if task_mode == "point":
            self.head = nn.Linear(d_model, H)
        elif task_mode == "quantile":
            Q = len(self.quantiles)
            self.head = nn.Linear(d_model, H * Q)
        else:
            raise ValueError(f"Unknown task_mode: {task_mode}")

        # domain classifier head (logits)
        self.domain_head = nn.Sequential(
            nn.Linear(d_model, domain_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(domain_hidden, num_domains)
        )

    def forward(self, x, grl_lambda: float = 1.0):
        """
        x: [B,L,V]
        """
        z = self.in_proj(x) + self.pos_emb
        z = self.drop(z)

        for blk in self.deform:
            z = blk(z)

        z = self.mamba(z)

        feat = self.norm_out(z[:, -1, :])  # [B,d]
        # task pred
        out = self.head(feat)
        if self.task_mode == "quantile":
            Q = len(self.quantiles)
            out = out.view(out.size(0), self.H, Q)

        # domain pred with GRL
        feat_rev = grad_reverse(feat, grl_lambda)
        domain_logits = self.domain_head(feat_rev)

        return out, domain_logits, feat
