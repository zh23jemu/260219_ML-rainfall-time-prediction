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
                 spatial_deform_enabled: bool = False,
                 spatial_deform_layers: int = 1,
                 spatial_deform_heads: int = 4,
                 spatial_deform_groups: int = 4,
                 spatial_deform_kernel: int = 5,
                 dropout: float = 0.1,
                 require_mamba: bool = False,
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
        self.use_spatial_deform = bool(spatial_deform_enabled and V_in > 1)

        # 时间分支：每个时间步把 V 维投影到 d_model
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
        self.mamba = ConBiMambaEncoder(
            d_model=d_model,
            num_layers=conbimamba_layers,
            dropout=dropout,
            conv_kernel=31,
            require_mamba=require_mamba,
        )

        self.temporal_norm_out = nn.LayerNorm(d_model)

        # 空间分支：对城市维(V)做 deform + BiMamba，显式建模空间传播关系
        if self.use_spatial_deform:
            self.spatial_in_proj = nn.Linear(L, d_model)
            self.spatial_pos_emb = nn.Parameter(torch.zeros(1, V_in, d_model))
            nn.init.normal_(self.spatial_pos_emb, std=0.02)
            self.spatial_drop = nn.Dropout(dropout)
            self.spatial_deform = nn.ModuleList([
                DeformBlock(
                    seq_len=V_in,
                    d_model=d_model,
                    n_heads=spatial_deform_heads,
                    n_groups=spatial_deform_groups,
                    kernel=spatial_deform_kernel,
                    dropout=dropout
                )
                for _ in range(spatial_deform_layers)
            ])
            self.spatial_mamba = ConBiMambaEncoder(
                d_model=d_model,
                num_layers=max(1, conbimamba_layers // 2),
                dropout=dropout,
                conv_kernel=15,
                require_mamba=require_mamba,
            )
            self.spatial_norm_out = nn.LayerNorm(d_model)

            self.fuse = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model),
            )

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

        feat_temporal = self.temporal_norm_out(z[:, -1, :])  # [B,d]

        if self.use_spatial_deform:
            # x: [B,L,V] -> [B,V,L]，将每个城市的L长度序列映射为城市token
            z_sp = self.spatial_in_proj(x.transpose(1, 2)) + self.spatial_pos_emb
            z_sp = self.spatial_drop(z_sp)
            for blk in self.spatial_deform:
                z_sp = blk(z_sp)
            z_sp = self.spatial_mamba(z_sp)
            feat_spatial = self.spatial_norm_out(z_sp.mean(dim=1))
            feat = self.fuse(torch.cat([feat_temporal, feat_spatial], dim=-1))
        else:
            feat = feat_temporal

        # task pred
        out = self.head(feat)
        if self.task_mode == "quantile":
            Q = len(self.quantiles)
            out = out.view(out.size(0), self.H, Q)

        # domain pred with GRL
        feat_rev = grad_reverse(feat, grl_lambda)
        domain_logits = self.domain_head(feat_rev)

        return out, domain_logits, feat
