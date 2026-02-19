import torch
import torch.nn.functional as F

def pinball_loss(y_true, y_pred, tau: float):
    # y_true,y_pred: [..., H]
    u = y_true - y_pred
    return torch.mean(torch.maximum(tau * u, (tau - 1) * u))

def huber_pinball_loss(y_true, y_pred, tau: float, delta: float = 1.0):
    # Huberized pinball (smooth)
    u = y_true - y_pred
    abs_u = torch.abs(u)
    huber = torch.where(abs_u <= delta, 0.5 * (u**2) / delta, abs_u - 0.5 * delta)
    # pinball weighting
    weight = torch.where(u >= 0, tau, 1 - tau)
    return torch.mean(weight * huber)

def quantile_loss(y_true, y_pred_q, quantiles, use_huber=False, delta=1.0):
    # y_pred_q: [B, H, Q]
    loss = 0.0
    for qi, tau in enumerate(quantiles):
        pred = y_pred_q[..., qi]
        if use_huber:
            loss = loss + huber_pinball_loss(y_true, pred, float(tau), delta)
        else:
            loss = loss + pinball_loss(y_true, pred, float(tau))
    return loss / len(quantiles)

def domain_ce_loss(domain_logits, domain_labels):
    # logits: [B, Kd], labels: [B]
    return F.cross_entropy(domain_logits, domain_labels)

def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)
