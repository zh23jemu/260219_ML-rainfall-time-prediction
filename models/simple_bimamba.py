import torch
import torch.nn as nn

from .conbimamba import ExBiMamba
from .grad_reverse import grad_reverse


class SimpleBiMambaForecast(nn.Module):
    """
    Lightweight single-branch BiMamba forecaster.

    Input:
      x: [B, L, V]
    Output:
      pred:
        - point: [B, H]
        - quantile: [B, H, Q]
      domain_logits: [B, Kd]
      feat: [B, d_model]
    """

    def __init__(
        self,
        V_in: int,
        L: int,
        H: int,
        num_domains: int,
        d_model: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        require_mamba: bool = False,
        task_mode: str = "quantile",
        quantiles=(0.1, 0.5, 0.9),
        domain_hidden: int = 64,
    ):
        super().__init__()
        self.H = H
        self.task_mode = task_mode
        self.quantiles = list(quantiles)

        self.in_proj = nn.Linear(V_in, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, L, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.drop = nn.Dropout(dropout)

        self.encoder = nn.ModuleList(
            [ExBiMamba(d_model=d_model, require_mamba=require_mamba) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.pool_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        if task_mode == "point":
            self.head = nn.Linear(d_model, H)
        elif task_mode == "quantile":
            self.head = nn.Linear(d_model, H * len(self.quantiles))
        else:
            raise ValueError(f"Unknown task_mode: {task_mode}")

        self.domain_head = nn.Sequential(
            nn.Linear(d_model, domain_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(domain_hidden, max(1, num_domains)),
        )

    def forward(self, x, grl_lambda: float = 1.0):
        z = self.drop(self.in_proj(x) + self.pos_emb)
        for layer in self.encoder:
            z = z + layer(self.norm(z))
        z = self.norm(z)

        feat_last = z[:, -1, :]
        feat_mean = z.mean(dim=1)
        feat = self.pool_proj(torch.cat([feat_last, feat_mean], dim=-1))

        pred = self.head(feat)
        if self.task_mode == "quantile":
            pred = pred.view(pred.size(0), self.H, len(self.quantiles))

        feat_rev = grad_reverse(feat, grl_lambda)
        domain_logits = self.domain_head(feat_rev)
        return pred, domain_logits, feat
